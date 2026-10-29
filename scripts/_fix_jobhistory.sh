#!/usr/bin/env bash
set -euo pipefail
source "$HOME/.hadoop_env"
export PATH="$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PIG_HOME/bin:$PATH"
CONF="$HADOOP_HOME/etc/hadoop/mapred-site.xml"

python3 - <<'PY'
from pathlib import Path
p = Path("/home/ahujade/opt/hadoop-3.3.6/etc/hadoop/mapred-site.xml")
t = p.read_text()
needed = {
    "mapreduce.jobhistory.address": "127.0.0.1:10020",
    "mapreduce.jobhistory.webapp.address": "127.0.0.1:19888",
    "mapreduce.jobhistory.bind-host": "127.0.0.1",
}
for name, value in needed.items():
    if f"<name>{name}</name>" in t:
        continue
    t = t.replace(
        "</configuration>",
        f"""  <property>
    <name>{name}</name>
    <value>{value}</value>
  </property>
</configuration>
""",
        1,
    )
p.write_text(t)
print("mapred-site jobhistory properties present")
PY

# Restart JobHistory so it advertises localhost, not 0.0.0.0.
jps -l | awk '/JobHistoryServer/{print $1}' | xargs -r kill
sleep 2
mkdir -p "$HADOOP_HOME/logs"
setsid -- mapred historyserver </dev/null >>"$HADOOP_HOME/logs/JobHistoryServer.nohup" 2>&1 &
for _ in $(seq 1 20); do
  if jps -l | grep -q JobHistoryServer; then
    echo "JobHistoryServer up"
    break
  fi
  sleep 1
done
jps -l | grep JobHistory || true
ss -lnt | grep -E '10020|19888' || true

echo "== pig yarn smoke =="
hdfs dfs -mkdir -p /tmp/pig_jhs_check
printf 'mistral\tllama\nllama\tnone\nqwen\tllama\n' > /tmp/pig_jhs_models.tsv
hdfs dfs -put -f /tmp/pig_jhs_models.tsv /tmp/pig_jhs_check/models.tsv
rm -f /tmp/pig_jhs_run.log
set +e
pig -x mapreduce -e "A = LOAD '/tmp/pig_jhs_check/models.tsv' USING PigStorage('\t') AS (model:chararray, base:chararray); B = GROUP A BY base; C = FOREACH B GENERATE group, COUNT(A); DUMP C;" > /tmp/pig_jhs_run.log 2>&1
pig_rc=$?
set -e
echo "pig_exit=$pig_rc"
grep -E '0\.0\.0\.0:10020|UnsetHostnameOrPort|Redirecting to job history|Success!|FAIL' /tmp/pig_jhs_run.log | tail -n 30
echo '--- pig stdout tail ---'
tail -n 25 /tmp/pig_jhs_run.log
exit "$pig_rc"
