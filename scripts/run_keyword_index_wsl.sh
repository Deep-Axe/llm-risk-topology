#!/usr/bin/env bash
# MapReduce keyword index over model cards (Hadoop streaming on YARN) -> Hive, joined with lineage and E_q.
# Usage: run_keyword_index_wsl.sh METADATA_HDFS_RUN_ROOT METADATA_TABLE_PREFIX EXPOSURE_TABLE
set -euo pipefail
source "$HOME/.hadoop_env"
export PATH="$JAVA_HOME/bin:$HADOOP_HOME/bin:$HIVE_HOME/bin:$PATH"
export TERM=dumb
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-} -Djline.terminal=jline.UnsupportedTerminal"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METADATA_HDFS="${1:?HDFS root of a finished metadata run}"
METADATA_TABLE="${2:?Hive table prefix of that run, e.g. llm_risk_topology.metadata_20261029_040751_44540}"
EXPOSURE_TABLE="${3:?Hive E_q table, e.g. llm_risk_topology.exposure_20261103_172946_6153}"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_$$"
OUTPUT="$METADATA_HDFS/keyword_index/$RUN_ID"
REPORT="$PROJECT_ROOT/report/keyword_index/$RUN_ID"
mkdir -p "$REPORT"
printf 'metadata_hdfs=%s\noutput=%s\n' "$METADATA_HDFS" "$OUTPUT" | tee "$REPORT/run.env"
STREAMING_JAR="$(ls "$HADOOP_HOME"/share/hadoop/tools/lib/hadoop-streaming-*.jar | head -1)"
started=$SECONDS
hadoop jar "$STREAMING_JAR" \
  -D mapreduce.job.name="keyword_index_$RUN_ID" -D mapreduce.job.reduces=2 \
  -files "$PROJECT_ROOT/mr/keyword_mapper.py,$PROJECT_ROOT/mr/keyword_reducer.py" \
  -mapper "python3 keyword_mapper.py" -reducer "python3 keyword_reducer.py" \
  -input "$METADATA_HDFS/raw/models.tsv" -output "$OUTPUT/index" > "$REPORT/mapreduce.log" 2>&1
echo "mapreduce_seconds=$((SECONDS - started))" | tee -a "$REPORT/run.env"
grep -E 'Map input records|Map output records|Reduce output records|Launched (map|reduce) tasks' \
  "$REPORT/mapreduce.log" | sed 's/^[[:space:]]*//' | tee "$REPORT/counters.txt"
hdfs dfs -cat "$OUTPUT/index/part-*" > "$REPORT/keyword_index.tsv"
TABLE="llm_risk_topology.keyword_index_$RUN_ID"
cat > "$REPORT/queries.hql" <<EOF
SET hive.execution.engine=mr;
SET hive.fetch.task.conversion=none;
SET hive.auto.convert.join=false;
CREATE EXTERNAL TABLE IF NOT EXISTS $TABLE (keyword STRING, model_id STRING, category STRING, matched_fields STRING)
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t' STORED AS TEXTFILE LOCATION '$OUTPUT/index';
SELECT 'category', category, COUNT(DISTINCT model_id) FROM $TABLE GROUP BY category;
SELECT 'keyword', keyword, COUNT(*) AS n FROM $TABLE GROUP BY keyword ORDER BY n DESC LIMIT 15;
SELECT 'safety_removed_children', e.parent_model_id, COUNT(DISTINCT e.child_model_id) AS children,
  COUNT(DISTINCT k.model_id) AS flagged
  FROM ${METADATA_TABLE}_edges e
  LEFT JOIN (SELECT DISTINCT model_id FROM $TABLE WHERE category = 'safety_removed') k ON e.child_model_id = k.model_id
  GROUP BY e.parent_model_id ORDER BY flagged DESC, children DESC LIMIT 15;
SELECT 'top_e_q_with_flags', x.model_id, x.reach, ROUND(x.e_q_mean, 1), COALESCE(f.flagged, 0)
  FROM (SELECT model_id, reach, e_q_mean FROM $EXPOSURE_TABLE WHERE q = 0.3) x
  LEFT JOIN (SELECT e.parent_model_id AS model_id, COUNT(DISTINCT k.model_id) AS flagged
             FROM ${METADATA_TABLE}_edges e
             JOIN (SELECT DISTINCT model_id FROM $TABLE WHERE category = 'safety_removed') k
               ON e.child_model_id = k.model_id
             GROUP BY e.parent_model_id) f ON x.model_id = f.model_id
  ORDER BY x.e_q_mean DESC LIMIT 15;
EOF
started=$SECONDS
beeline -u jdbc:hive2://localhost:10000 -n "$USER" --silent=true --showHeader=false --outputformat=tsv2 \
  -f "$REPORT/queries.hql" > "$REPORT/hive_queries.log" 2>&1
echo "hive_seconds=$((SECONDS - started))" | tee -a "$REPORT/run.env"
grep -aE "^(null)?(category|keyword|safety_removed_children|top_e_q_with_flags)" "$REPORT/hive_queries.log" \
  | sed 's/^null//' > "$REPORT/hive_results.tsv"
cat "$REPORT/hive_results.tsv"
echo "KEYWORD_INDEX_OK"
echo "Evidence: $REPORT"
