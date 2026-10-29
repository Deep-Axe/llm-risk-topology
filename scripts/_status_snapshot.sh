#!/usr/bin/env bash
source "$HOME/.hadoop_env"
export PATH="$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$SPARK_HOME/bin:$PIG_HOME/bin:$HIVE_HOME/bin:$PATH"
echo "JAVA=$("$JAVA_HOME/bin/java" -version 2>&1 | head -1)"
echo "HADOOP=$(hadoop version 2>/dev/null | head -1)"
spark-submit --version 2>&1 | grep -E 'version 3' | head -1
echo "PIG=$(pig -version 2>/dev/null | head -1)"
echo "HIVE_HOME=$HIVE_HOME"
echo "=== JPS ==="
jps -l
echo "=== HDFS ==="
hdfs dfs -ls /
echo "=== YARN ==="
yarn node -list 2>/dev/null | tail -n 5
echo "=== HISTORY ==="
mapred job -list all 2>/dev/null | head -n 8 || true
