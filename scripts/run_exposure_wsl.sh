#!/usr/bin/env bash
# E_q on a finished metadata run: Spark job -> HDFS -> Hive table, checked against the Python reference.
# Usage: run_exposure_wsl.sh METADATA_HDFS_RUN_ROOT [REPLICATES] [SEED] [RELATIONS]
set -euo pipefail
source "$HOME/.hadoop_env"
export PATH="$JAVA_HOME/bin:$HADOOP_HOME/bin:$SPARK_HOME/bin:$HIVE_HOME/bin:$PATH"
export TERM=dumb SPARK_LOCAL_IP=127.0.0.1
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-} -Djline.terminal=jline.UnsupportedTerminal"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export METADATA_HDFS="${1:?Supply the HDFS root of a finished metadata run}"
export EXPOSURE_REPLICATES="${2:-200}" EXPOSURE_SEED="${3:-42}"
export EXPOSURE_RELATIONS="${4:-fine_tune,adapter,merge,quantization,other}"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_$$"
export EXPOSURE_HDFS="$METADATA_HDFS/exposure/$RUN_ID"
REPORT="$PROJECT_ROOT/report/exposure/$RUN_ID"
mkdir -p "$REPORT"
printf 'metadata_hdfs=%s\nexposure_hdfs=%s\nreplicates=%s\nseed=%s\nrelations=%s\n' "$METADATA_HDFS" \
  "$EXPOSURE_HDFS" "$EXPOSURE_REPLICATES" "$EXPOSURE_SEED" "$EXPOSURE_RELATIONS" | tee "$REPORT/run.env"
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
step spark_exposure spark-shell --master yarn --deploy-mode client --num-executors 1 --executor-cores 2 \
  --executor-memory 2g --driver-memory 2g --conf spark.ui.enabled=false --conf spark.sql.shuffle.partitions=16 \
  --conf spark.default.parallelism=16 --conf spark.executor.memoryOverhead=512 \
  -i "$PROJECT_ROOT/spark/scala/exposure.scala"
grep -q 'EXPOSURE_SPARK_OK' "$REPORT/spark_exposure.log"
hdfs dfs -cat "$EXPOSURE_HDFS/node_exposure_csv/part-*" > "$REPORT/node_exposure_spark.csv"
hdfs dfs -cat "$EXPOSURE_HDFS/summary/part-*" > "$REPORT/summary.json"
# Reference on the exact edge rows the Spark job read.
hdfs dfs -cat "$METADATA_HDFS/processed/edge_contracts/part-*" > "$REPORT/edges.jsonl"
step python_reference python3 "$PROJECT_ROOT/python/exposure.py" --edges "$REPORT/edges.jsonl" \
  --output "$REPORT/node_exposure_python.csv" --replicates "$EXPOSURE_REPLICATES" --seed "$EXPOSURE_SEED" \
  --relations "$EXPOSURE_RELATIONS"
step compare python3 "$PROJECT_ROOT/python/compare_exposure.py" --spark "$REPORT/node_exposure_spark.csv" \
  --reference "$REPORT/node_exposure_python.csv" --output "$REPORT/comparison.json"
TABLE="llm_risk_topology.exposure_${RUN_ID}"
cat > "$REPORT/queries.hql" <<EOF
SET hive.execution.engine=mr;
SET hive.fetch.task.conversion=none;
CREATE DATABASE IF NOT EXISTS llm_risk_topology;
CREATE EXTERNAL TABLE $TABLE (model_id STRING, q DOUBLE, reach BIGINT, e_q_mean DOUBLE, e_q_se DOUBLE,
  replicates INT, seed BIGINT, relations STRING) STORED AS PARQUET LOCATION '$EXPOSURE_HDFS/node_exposure';
SELECT 'rows', COUNT(*) FROM $TABLE;
SELECT 'top_e_q_0.3', model_id, reach, ROUND(e_q_mean, 3), ROUND(e_q_se, 3) FROM $TABLE
  WHERE q = 0.3 ORDER BY e_q_mean DESC, model_id LIMIT 10;
SELECT 'top_reach', model_id, reach FROM $TABLE WHERE q = 1.0 ORDER BY reach DESC, model_id LIMIT 10;
EOF
step hive_queries beeline -u jdbc:hive2://localhost:10000 -n "$USER" --silent=true --showHeader=false \
  --outputformat=tsv2 -f "$REPORT/queries.hql"
cat "$REPORT/timings.tsv"
echo "Evidence: $REPORT"
