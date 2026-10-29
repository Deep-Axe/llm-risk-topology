#!/usr/bin/env bash
# Prove HDFS, YARN, MapReduce, Hive, Pig, and Spark — not just `hdfs dfs -ls`.
set -uo pipefail

# shellcheck disable=SC1090
source "$HOME/.hadoop_env"
export PATH="$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$SPARK_HOME/bin:$PIG_HOME/bin:$HIVE_HOME/bin:$PATH"

fail=0
check() {
  local name="$1"
  shift
  echo "== $name =="
  if "$@"; then
    echo "OK $name"
  else
    echo "FAIL $name" >&2
    fail=1
  fi
  echo
}

check "java 11" bash -c '"$JAVA_HOME/bin/java" -version 2>&1 | grep -E "11\\.0"'

echo "== jps daemons =="
jps -l
if jps -l | grep -q NameNode && jps -l | grep -q DataNode && jps -l | grep -q ResourceManager && jps -l | grep -q NodeManager; then
  echo "OK jps daemons"
else
  echo "FAIL jps daemons" >&2
  fail=1
fi
echo

check "HDFS ls" hdfs dfs -ls /

echo "== HDFS put/cat =="
echo stack-ok > /tmp/stack_ok.txt
if hdfs dfs -put -f /tmp/stack_ok.txt /tmp/stack_ok.txt && hdfs dfs -cat /tmp/stack_ok.txt | grep -q stack-ok; then
  echo "OK HDFS put/cat"
else
  echo "FAIL HDFS put/cat" >&2
  fail=1
fi
echo

echo "== YARN node =="
yarn node -list
if yarn node -list 2>/dev/null | grep -qi RUNNING; then
  echo "OK YARN node"
else
  echo "FAIL YARN node" >&2
  fail=1
fi
echo

echo "== MapReduce pi (YARN) =="
if hadoop jar "$HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-examples-3.3.6.jar" pi 2 20; then
  echo "OK mapreduce-pi"
else
  echo "FAIL mapreduce-pi" >&2
  fail=1
fi
echo

echo "== Hive 4.0.1 =="
if beeline -u "jdbc:hive2://127.0.0.1:10000" -n "$USER" --silent=true --outputformat=tsv2 -e "SELECT 1 AS ok;"; then
  echo "OK hive"
else
  echo "FAIL hive" >&2
  fail=1
fi
echo

echo "== Pig local =="
if pig -x local -e "A = LOAD '/tmp/stack_ok.txt' AS (s:chararray); DUMP A;"; then
  echo "OK pig"
else
  echo "FAIL pig (Pig 0.17 on Java 11 is unofficial; HDFS/Hive/Spark still count)" >&2
  fail=1
fi
echo

echo "== Spark =="
if spark-submit --version; then
  echo "OK spark-submit"
else
  echo "FAIL spark-submit" >&2
  fail=1
fi
echo

if [[ "$fail" -ne 0 ]]; then
  echo "One or more checks failed." >&2
  exit 1
fi
echo "All stack checks passed."
