#!/usr/bin/env bash
# One immutable metadata snapshot -> HDFS -> Pig/YARN -> Spark/GraphX/YARN -> Hive.
set -euo pipefail
source "$HOME/.hadoop_env"
export PATH="$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$SPARK_HOME/bin:$PIG_HOME/bin:$HIVE_HOME/bin:$PATH"
export TERM=dumb SPARK_LOCAL_IP=127.0.0.1
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-} -Djline.terminal=jline.UnsupportedTerminal"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAPSHOT_DIR="$(realpath "${1:?Supply the local collected snapshot directory}")"
export METADATA_SNAPSHOT_ID="$(sha256sum "$SNAPSHOT_DIR/metadata_snapshot.json" | cut -d ' ' -f 1)"
export METADATA_EXPECTED_RECORDS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["record_count"])' "$SNAPSHOT_DIR/metadata_snapshot.json")"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_$$"
REPORT="$PROJECT_ROOT/report/metadata_pipeline/$RUN_ID"
export METADATA_HDFS="/data/llm_risk_topology/metadata/$METADATA_SNAPSHOT_ID/$RUN_ID"
TABLE_SUFFIX="metadata_$RUN_ID"
mkdir -p "$REPORT"
python3 "$PROJECT_ROOT/python/monitor_wsl_resources.py" "$REPORT/resources.json" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true' EXIT
printf 'stage\tseconds\tstatus\n' > "$REPORT/timings.tsv"
printf 'snapshot_id=%s\nhdfs_root=%s\nreport_dir=%s\n' "$METADATA_SNAPSHOT_ID" "$METADATA_HDFS" "$REPORT" | tee "$REPORT/run.env"
step() {
  local name="$1"; shift
  local started="$SECONDS"
  echo "RUN $name"
  if /usr/bin/time -v -o "$REPORT/$name.resources.txt" timeout --kill-after=15s 3600s "$@" </dev/null > "$REPORT/$name.log" 2>&1; then
    echo "PASS $name" | tee -a "$REPORT/results.txt"
    printf '%s\t%s\tPASS\n' "$name" "$((SECONDS - started))" >> "$REPORT/timings.tsv"
  else
    tail -n 35 "$REPORT/$name.log"
    echo "FAIL $name" | tee -a "$REPORT/results.txt"
    printf '%s\t%s\tFAIL\n' "$name" "$((SECONDS - started))" >> "$REPORT/timings.tsv"
    exit 1
  fi
}
# Verify bytes before uploading; no service restarts or formatting.
python3 - "$SNAPSHOT_DIR" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1])
m=json.loads((p/'metadata_snapshot.json').read_text())
assert m['status']=='frozen' and m['record_count']==m['requested_records']
for artifact in m['artifacts']:
    h=hashlib.sha256()
    with (p/artifact['path']).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): h.update(block)
    assert h.hexdigest()==artifact['sha256']
PY
step hdfs_health hdfs dfsadmin -report
hdfs dfs -mkdir -p "$METADATA_HDFS/raw"
step hdfs_upload hdfs dfs -put "$SNAPSHOT_DIR/models.tsv" "$SNAPSHOT_DIR/models.jsonl.gz" "$SNAPSHOT_DIR/metadata_snapshot.json" "$METADATA_HDFS/raw/"
step pig_extract pig -x mapreduce -param "PARSER=$PROJECT_ROOT/python/extract_lineage.py" -param "INPUT=$METADATA_HDFS/raw/models.tsv" -param "OUTPUT=$METADATA_HDFS/pig" -f "$PROJECT_ROOT/pig/extract_metadata.pig"
step spark_graphx spark-shell --master yarn --deploy-mode client --num-executors 1 --executor-cores 2 --executor-memory 1536m --driver-memory 1536m --conf spark.ui.enabled=false --conf spark.sql.shuffle.partitions=8 --conf spark.default.parallelism=8 --conf spark.executor.memoryOverhead=384 -i "$PROJECT_ROOT/spark/scala/metadata_pipeline.scala"
grep -q 'METADATA_PIPELINE_SPARK_OK' "$REPORT/spark_graphx.log"
hdfs dfs -cat "$METADATA_HDFS/processed/summary/part-*" > "$REPORT/summary.json"
if [[ "$METADATA_EXPECTED_RECORDS" -lt 10000 ]]; then
  hdfs dfs -cat "$METADATA_HDFS/processed/edge_contracts/part-*" > "$REPORT/edges.jsonl"
fi
for artifact in distributed_validation reference_relations reference_hubs; do
  hdfs dfs -cat "$METADATA_HDFS/processed/$artifact/part-*" > "$REPORT/$artifact.jsonl"
done
hdfs dfs -cat "$METADATA_HDFS/processed/edge_sample/part-*" > "$REPORT/edges_sample.jsonl"
cat > "$REPORT/queries.hql" <<EOF
SET hive.execution.engine=mr;
SET hive.fetch.task.conversion=none;
SET hive.server2.in.place.progress=false;
CREATE DATABASE IF NOT EXISTS llm_risk_topology;
CREATE EXTERNAL TABLE llm_risk_topology.${TABLE_SUFFIX}_models (model_id STRING, revision STRING, card_json STRING) STORED AS PARQUET LOCATION '$METADATA_HDFS/processed/models';
CREATE EXTERNAL TABLE llm_risk_topology.${TABLE_SUFFIX}_edges (schema_version STRING, edge_id STRING, snapshot_id STRING, child_model_id STRING, parent_model_id STRING, relation STRING, lineage_status STRING, evidence ARRAY<STRUCT<url:STRING,revision:STRING,field:STRING,value:STRING>>) STORED AS PARQUET LOCATION '$METADATA_HDFS/processed/edges';
CREATE EXTERNAL TABLE llm_risk_topology.${TABLE_SUFFIX}_metrics (vertex_id BIGINT, model_id STRING, component_id BIGINT, upstream_pagerank DOUBLE, observed_children BIGINT, declared_parents BIGINT, in_sample BOOLEAN, snapshot_id STRING) STORED AS PARQUET LOCATION '$METADATA_HDFS/processed/node_metrics';
SELECT 'models', COUNT(*) FROM llm_risk_topology.${TABLE_SUFFIX}_models;
SELECT 'edges', COUNT(*) FROM llm_risk_topology.${TABLE_SUFFIX}_edges;
SELECT 'vertices', COUNT(*) FROM llm_risk_topology.${TABLE_SUFFIX}_metrics;
SELECT 'relation', relation, COUNT(*) FROM llm_risk_topology.${TABLE_SUFFIX}_edges GROUP BY relation;
SELECT 'hub', model_id, observed_children, upstream_pagerank FROM llm_risk_topology.${TABLE_SUFFIX}_metrics ORDER BY observed_children DESC, model_id LIMIT 10;
EOF
step hive_queries beeline -u jdbc:hive2://localhost:10000 -n "$USER" --silent=true --showHeader=false --outputformat=tsv2 -f "$REPORT/queries.hql"
cat "$REPORT/summary.json"
echo "Evidence: $REPORT"
