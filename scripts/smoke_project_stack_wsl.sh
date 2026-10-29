#!/usr/bin/env bash
# Small runtime checks in an isolated namespace; no service restart or HDFS format.
set -uo pipefail
source "$HOME/.hadoop_env"
export PATH="$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$SPARK_HOME/bin:$PIG_HOME/bin:$HIVE_HOME/bin:$PATH"
export TERM=dumb
export SPARK_LOCAL_IP=127.0.0.1
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-} -Djline.terminal=jline.UnsupportedTerminal"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_$$"
REPORT_DIR="$PROJECT_ROOT/report/stack_checks/$RUN_ID"
WORK_DIR="$(mktemp -d /tmp/risk-topology-smoke.XXXXXX)"
HDFS_TEST="/tmp/risk_topology_smoke/$RUN_ID"
HIVE_TABLE="risk_topology_smoke_$RUN_ID"
mkdir -p "$REPORT_DIR"
printf 'check\tstatus\texit_code\n' > "$REPORT_DIR/results.tsv"
printf 'run_id=%s\nwork_dir=%s\nhdfs_path=%s\nhive_table=%s\n' "$RUN_ID" "$WORK_DIR" "$HDFS_TEST" "$HIVE_TABLE" > "$REPORT_DIR/run.env"
printf 'model-a\tbase-a\nmodel-b\tbase-a\nmodel-c\tbase-b\n' > "$WORK_DIR/models.tsv"
failed=0

check() {
  local name="$1" limit="$2"; shift 2
  printf 'RUN %s\n' "$name"
  timeout --kill-after=10s "$limit" "$@" </dev/null > "$REPORT_DIR/$name.log" 2>&1
  local result=$?
  if [[ "$result" -eq 0 ]]; then
    printf '%s\tPASS\t0\n' "$name" >> "$REPORT_DIR/results.tsv"
    printf 'PASS %s\n' "$name"
  else
    printf '%s\tFAIL\t%s\n' "$name" "$result" >> "$REPORT_DIR/results.tsv"
    printf 'FAIL %s (exit %s):\n' "$name" "$result"
    tail -n 18 "$REPORT_DIR/$name.log"
    failed=1
  fi
}

"$JAVA_HOME/bin/java" -version > "$REPORT_DIR/java.log" 2>&1
jps -l > "$REPORT_DIR/daemons.log" 2>&1
check hdfs_health 30s hdfs dfsadmin -report
check yarn_nodes 30s yarn node -list
check hdfs_mkdir 30s hdfs dfs -mkdir -p "$HDFS_TEST/input"
check hdfs_upload 30s hdfs dfs -put "$WORK_DIR/models.tsv" "$HDFS_TEST/input/models.tsv"
hdfs dfs -cat "$HDFS_TEST/input/models.tsv" > "$WORK_DIR/roundtrip.tsv" 2> "$REPORT_DIR/hdfs_read.log"
check hdfs_roundtrip 10s cmp "$WORK_DIR/models.tsv" "$WORK_DIR/roundtrip.tsv"
check mapreduce_yarn 150s hadoop jar "$HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-examples-3.3.6.jar" wordcount "$HDFS_TEST/input" "$HDFS_TEST/wordcount"
hdfs dfs -cat "$HDFS_TEST/wordcount/part-r-*" > "$REPORT_DIR/mapreduce_output.tsv" 2>> "$REPORT_DIR/mapreduce_yarn.log"

cat > "$WORK_DIR/extract.pig" <<EOF
A = LOAD '$HDFS_TEST/input/models.tsv' USING PigStorage('\t') AS (model_id:chararray, base_model:chararray);
B = GROUP A BY base_model;
C = FOREACH B GENERATE group AS base_model, COUNT(A) AS children;
STORE C INTO '$HDFS_TEST/pig_hubs' USING PigStorage('\t');
EOF
check pig_mapreduce 150s pig -x mapreduce -f "$WORK_DIR/extract.pig"
hdfs dfs -cat "$HDFS_TEST/pig_hubs/part-*" > "$REPORT_DIR/pig_output.tsv" 2>> "$REPORT_DIR/pig_mapreduce.log"

cat > "$WORK_DIR/spark.scala" <<EOF
import org.apache.spark.graphx._
import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import org.apache.spark.ml.linalg.Vectors
try {
  val models = spark.read.option("sep", "\\t").schema("model_id STRING, base_model STRING").csv("$HDFS_TEST/input")
  assert(models.count() == 3L)
  models.write.parquet("$HDFS_TEST/parquet")
  assert(spark.read.parquet("$HDFS_TEST/parquet").count() == 3L)
  val graph = Graph(sc.parallelize(Seq((1L,"base"),(2L,"child-a"),(3L,"child-b"))), sc.parallelize(Seq(Edge(2L,1L,1),Edge(3L,1L,1))))
  assert(graph.connectedComponents().vertices.map(_._2).distinct().count() == 1L)
  assert(graph.pageRank(0.001).vertices.count() == 3L)
  val vectors = Seq((1,Vectors.dense(1.0,0.0)),(2,Vectors.dense(0.99,0.1)),(3,Vectors.dense(0.0,1.0))).toDF("id","features")
  val lsh = new BucketedRandomProjectionLSH().setInputCol("features").setOutputCol("hashes").setBucketLength(2.0).setNumHashTables(3).setSeed(42L).fit(vectors)
  assert(lsh.approxSimilarityJoin(vectors,vectors,0.5).count() >= 3L)
  println("STACK_SPARK_GRAPHX_MLLIB_OK rows=3 components=1 pagerank_vertices=3")
  spark.stop()
  System.exit(0)
} catch {
  case error: Throwable => error.printStackTrace(); System.exit(1)
}
EOF
check spark_graphx_mllib 150s spark-shell --master 'local[2]' --driver-memory 768m --conf spark.ui.enabled=false -i "$WORK_DIR/spark.scala"
if ! grep -q 'STACK_SPARK_GRAPHX_MLLIB_OK' "$REPORT_DIR/spark_graphx_mllib.log"; then
  printf 'spark_assertion_marker\tFAIL\t1\n' >> "$REPORT_DIR/results.tsv"
  printf 'FAIL spark_assertion_marker\n'
  failed=1
fi

cat > "$WORK_DIR/check.hql" <<EOF
SET hive.execution.engine=mr;
SET hive.fetch.task.conversion=none;
CREATE EXTERNAL TABLE $HIVE_TABLE (model_id STRING, base_model STRING) STORED AS PARQUET LOCATION '$HDFS_TEST/parquet';
SELECT base_model, COUNT(*) AS children FROM $HIVE_TABLE GROUP BY base_model ORDER BY base_model;
DROP TABLE $HIVE_TABLE;
EOF
check hive_hdfs_parquet 180s beeline -u "jdbc:hive2://localhost:10000" -n "$USER" --silent=true --showHeader=false --outputformat=tsv2 -f "$WORK_DIR/check.hql"

printf '\nReport: %s\n' "$REPORT_DIR"
cat "$REPORT_DIR/results.tsv"
printf 'Temporary HDFS and local smoke data retained at the paths in run.env.\n'
exit "$failed"
