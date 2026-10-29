#!/usr/bin/env bash
set -euo pipefail
source "$HOME/.hadoop_env"
export PATH="$JAVA_HOME/bin:$HADOOP_HOME/bin:$HIVE_HOME/bin:$PATH"

SITE="$HIVE_HOME/conf/hive-site.xml"
if ! grep -q 'hive.server2.thrift.port' "$SITE"; then
  python3 -c '
from pathlib import Path
p = Path("'"$SITE"'")
t = p.read_text()
extra = """  <property>
    <name>hive.server2.thrift.bind.host</name>
    <value>127.0.0.1</value>
  </property>
  <property>
    <name>hive.server2.thrift.port</name>
    <value>10000</value>
  </property>
  <property>
    <name>hive.server2.enable.doAs</name>
    <value>false</value>
  </property>
  <property>
    <name>hive.server2.authentication</name>
    <value>NONE</value>
  </property>
</configuration>
"""
p.write_text(t.replace("</configuration>", extra, 1))
print("hive-site patched")
'
else
  echo "hive-site already has HS2 props"
fi

mkdir -p "$HADOOP_HOME/logs"
if jps -l | grep -q HiveServer2; then
  echo "already running HiveServer2"
else
  setsid -- "$HIVE_HOME/bin/hiveserver2" </dev/null >>"$HADOOP_HOME/logs/HiveServer2.nohup" 2>&1 &
  echo "started HiveServer2"
fi

for _ in $(seq 1 45); do
  if grep -q "Started HiveServer2" "$HADOOP_HOME/logs/HiveServer2.nohup" 2>/dev/null; then
    echo "hs2_ready"
    break
  fi
  if ss -lnt 2>/dev/null | grep -q ':10000'; then
    echo "hs2_listen"
    break
  fi
  sleep 2
done

jps -l | grep -E 'HiveServer2|RunJar' || true
tail -n 25 "$HADOOP_HOME/logs/HiveServer2.nohup" || true
echo "---QUERY---"
beeline -u "jdbc:hive2://127.0.0.1:10000" -n "$USER" --silent=true --outputformat=tsv2 -e "SELECT 1 AS ok;"
echo "hive_exit=$?"
