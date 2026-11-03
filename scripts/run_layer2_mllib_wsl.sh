#!/usr/bin/env bash
# Layer 2 on the cluster: staged responses -> HDFS -> Spark MLlib (LSH k-NN, K-Means) -> Hive tables and queries.
# Usage: run_layer2_mllib_wsl.sh LOCAL_STAGED_DIR (contains responses.parquet from python/stage_layer2.py)
set -euo pipefail
source "$HOME/.hadoop_env"
export PATH="$JAVA_HOME/bin:$HADOOP_HOME/bin:$SPARK_HOME/bin:$HIVE_HOME/bin:$PATH"
export TERM=dumb SPARK_LOCAL_IP=127.0.0.1
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-} -Djline.terminal=jline.UnsupportedTerminal"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGED="$(realpath "${1:?Local folder with responses.parquet}")"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_$$"
export LAYER2_HDFS="/data/llm_risk_topology/layer2/$RUN_ID"
export LAYER2_OUTPUT="$LAYER2_HDFS/processed"
REPORT="$PROJECT_ROOT/report/layer2_mllib/$RUN_ID"
mkdir -p "$REPORT"
printf 'staged=%s\nhdfs=%s\n' "$STAGED" "$LAYER2_HDFS" | tee "$REPORT/run.env"
printf 'stage\tseconds\tstatus\n' > "$REPORT/timings.tsv"
step() {
  local name="$1"; shift
  local started="$SECONDS"
  echo "RUN $name"
  if timeout --kill-after=15s 7200s "$@" </dev/null > "$REPORT/$name.log" 2>&1; then
    printf '%s\t%s\tPASS\n' "$name" "$((SECONDS - started))" >> "$REPORT/timings.tsv"
    echo "PASS $name"
  else
    tail -n 40 "$REPORT/$name.log"
    printf '%s\t%s\tFAIL\n' "$name" "$((SECONDS - started))" >> "$REPORT/timings.tsv"
    echo "FAIL $name"
    exit 1
  fi
}
hdfs dfs -mkdir -p "$LAYER2_HDFS/responses"
step hdfs_upload hdfs dfs -put "$STAGED/responses.parquet" "$LAYER2_HDFS/responses/"
step spark_mllib spark-shell --master yarn --deploy-mode client --num-executors 1 --executor-cores 2 \
  --executor-memory 2g --driver-memory 2g --conf spark.ui.enabled=false --conf spark.sql.shuffle.partitions=16 \
  --conf spark.executor.memoryOverhead=512 -i "$PROJECT_ROOT/spark/scala/response_mllib.scala"
grep -q 'LAYER2_MLLIB_OK' "$REPORT/spark_mllib.log"
grep 'LAYER2_MODEL_DONE' "$REPORT/spark_mllib.log" | tee "$REPORT/models.txt"
hdfs dfs -cat "$LAYER2_OUTPUT/summary/part-*" > "$REPORT/summary.jsonl"
T="llm_risk_topology.layer2_${RUN_ID}"
cat > "$REPORT/queries.hql" <<EOF
SET hive.execution.engine=mr;
SET hive.fetch.task.conversion=none;
CREATE EXTERNAL TABLE ${T}_responses (response_id STRING, model_id STRING, prompt_id STRING, split STRING,
  group_id STRING, source STRING, intent STRING, variant_type STRING, output_tokens BIGINT, output_truncated BOOLEAN,
  refusal BOOLEAN, response_harmful BOOLEAN, prompt_harmful BOOLEAN, refusal_rules BOOLEAN, text_chars BIGINT)
  STORED AS PARQUET LOCATION '$LAYER2_HDFS/responses';
CREATE EXTERNAL TABLE ${T}_clusters (response_id STRING, model_id STRING, prompt_id STRING, split STRING,
  source STRING, intent STRING, variant_type STRING, refusal BOOLEAN, response_harmful BOOLEAN, refusal_rules BOOLEAN,
  cluster INT)
  STORED AS PARQUET LOCATION '$LAYER2_OUTPUT/clusters';
SELECT 'rates', model_id, source, intent, variant_type, COUNT(*),
  ROUND(AVG(CAST(COALESCE(refusal, refusal_rules) AS INT)), 3), ROUND(AVG(CAST(response_harmful AS INT)), 3)
  FROM ${T}_responses GROUP BY model_id, source, intent, variant_type ORDER BY model_id, source, intent, variant_type;
SELECT 'cluster', model_id, cluster, COUNT(*), ROUND(AVG(CAST(COALESCE(refusal, refusal_rules) AS INT)), 3),
  ROUND(AVG(CAST(response_harmful AS INT)), 3)
  FROM ${T}_clusters GROUP BY model_id, cluster ORDER BY model_id, cluster;
EOF
step hive_queries beeline -u jdbc:hive2://localhost:10000 -n "$USER" --silent=true --showHeader=false \
  --outputformat=tsv2 -f "$REPORT/queries.hql"
grep -aE "^(null)?(rates|cluster)" "$REPORT/hive_queries.log" | sed 's/^null//' > "$REPORT/hive_results.tsv"
cat "$REPORT/timings.tsv"
echo "LAYER2_PIPELINE_OK"
echo "Evidence: $REPORT"
