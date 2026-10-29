#!/usr/bin/env bash
# Format (once), start HDFS + YARN, create warehouse dirs, init Hive metastore.
set -euo pipefail

# shellcheck disable=SC1090
source "$HOME/.hadoop_env"

if [[ -z "${JAVA_HOME:-}" || ! -x "$JAVA_HOME/bin/java" ]]; then
  echo "JAVA_HOME is not a JDK 11 install. Re-run setup_hadoop_wsl.sh first." >&2
  exit 1
fi

echo "java: $("$JAVA_HOME/bin/java" -version 2>&1 | head -1)"

NN_CURRENT="$HOME/hadoop-data/namenode/current"
if [[ ! -d "$NN_CURRENT" ]]; then
  echo "== format NameNode =="
  mkdir -p "$HOME/hadoop-data/namenode" "$HOME/hadoop-data/datanode"
  hdfs namenode -format -force -nonInteractive
fi

# Hadoop maps SIGHUP to shutdown. Closing the WSL pty would kill --daemon
# processes; start each one in a new session with no controlling terminal.
start_svc() {
  local pattern="$1"; shift
  if jps -l 2>/dev/null | grep -q "$pattern"; then
    echo "already running $pattern"
    return 0
  fi
  mkdir -p "$HADOOP_HOME/logs"
  local log="$HADOOP_HOME/logs/${pattern}.nohup"
  setsid -- "$@" </dev/null >>"$log" 2>&1 &
  echo "started $pattern"
}

echo "== start HDFS =="
start_svc NameNode hdfs namenode
start_svc DataNode hdfs datanode
start_svc SecondaryNameNode hdfs secondarynamenode || true

echo "== start YARN =="
start_svc ResourceManager yarn resourcemanager
start_svc NodeManager yarn nodemanager
start_svc JobHistoryServer mapred historyserver

echo "== wait for HDFS =="
for _ in $(seq 1 30); do
  if hdfs dfsadmin -safemode wait >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
hdfs dfsadmin -safemode leave >/dev/null 2>&1 || true

echo "== HDFS dirs =="
hdfs dfs -mkdir -p /tmp /tmp/hive /user/"$USER" /user/hive/warehouse /data/hf/cards /data/embeddings
hdfs dfs -chmod 1777 /tmp /tmp/hive
hdfs dfs -chmod g+w /user/hive/warehouse || true

echo "== Hive metastore (Derby) =="
mkdir -p "$HOME/hive-metastore" "$HOME/hive-tmp"
if [[ ! -d "$HOME/hive-metastore/metastore_db" ]]; then
  "$HIVE_HOME/bin/schematool" -dbType derby -initSchema
else
  echo "metastore already initialized"
fi

echo "== HiveServer2 =="
start_svc HiveServer2 "$HIVE_HOME/bin/hiveserver2"

echo
echo "daemons:"
jps -l | grep -E 'NameNode|DataNode|SecondaryNameNode|ResourceManager|NodeManager|JobHistoryServer|HiveServer2|RunJar' || jps
echo
echo "Stack is up. Run: bash /mnt/c/Users/ahuja/Desktop/minor_big_data/scripts/verify_hadoop_wsl.sh"
