#!/usr/bin/env bash
# User-space Hadoop stack on WSL Ubuntu. No sudo.
# Java 11 + Hadoop 3.3.6 + Spark 3.5.5 + Pig 0.17 + Hive 4.0.1
# Hive 4.0.1 is the newest Hive that still runs on JDK 11 (Java 8 bytecode).
# Hive 4.1+ requires JDK 17; Hive 4.2+ requires JDK 21.
set -euo pipefail

OPT="${HADOOP_OPT:-$HOME/opt}"
SRC="$OPT/src"
DATA="$HOME/hadoop-data"
HIVE_META="$HOME/hive-metastore"
mkdir -p "$OPT" "$SRC" "$DATA" "$HIVE_META" "$HOME/hive-tmp"

JAVA_TGZ="$SRC/OpenJDK11U-jdk_x64_linux_hotspot_11.0.25_9.tar.gz"
HADOOP_TGZ="$SRC/hadoop-3.3.6.tar.gz"
SPARK_TGZ="$SRC/spark-3.5.5-bin-hadoop3.tgz"
PIG_TGZ="$SRC/pig-0.17.0.tar.gz"
HIVE_TGZ="$SRC/apache-hive-4.0.1-bin.tar.gz"

JAVA_HOME_DIR="$OPT/jdk-11"
HADOOP_HOME="$OPT/hadoop-3.3.6"
SPARK_HOME="$OPT/spark-3.5.5-bin-hadoop3"
PIG_HOME="$OPT/pig-0.17.0"
HIVE_HOME="$OPT/apache-hive-4.0.1-bin"

download() {
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then
    echo "already have $(basename "$dest") ($(du -h "$dest" | awk '{print $1}'))"
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  local part="${dest}.part"
  echo "GET $url"
  curl -L --fail --retry 8 --retry-all-errors --retry-delay 3 -C - --progress-bar -o "$part" "$url"
  mv -f "$part" "$dest"
}

extract_once() {
  local archive="$1" dest="$2" inner="$3"
  if [[ -d "$dest" && -n "$(ls -A "$dest" 2>/dev/null || true)" ]]; then
    echo "already extracted $(basename "$dest")"
    return 0
  fi
  echo "extract $(basename "$archive") -> $dest"
  rm -rf "$dest" "${dest}.extracting"
  mkdir -p "${dest}.extracting"
  tar -xzf "$archive" -C "${dest}.extracting"
  mv "${dest}.extracting/$inner" "$dest"
  rm -rf "${dest}.extracting"
}

write_xml() {
  local path="$1"
  cat >"$path"
}

echo "== downloads =="
download "https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.25%2B9/OpenJDK11U-jdk_x64_linux_hotspot_11.0.25_9.tar.gz" "$JAVA_TGZ"
download "https://archive.apache.org/dist/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz" "$HADOOP_TGZ"
download "https://archive.apache.org/dist/spark/spark-3.5.5/spark-3.5.5-bin-hadoop3.tgz" "$SPARK_TGZ"
download "https://archive.apache.org/dist/pig/pig-0.17.0/pig-0.17.0.tar.gz" "$PIG_TGZ"
download "https://archive.apache.org/dist/hive/hive-4.0.1/apache-hive-4.0.1-bin.tar.gz" "$HIVE_TGZ"

echo "== extract =="
extract_once "$JAVA_TGZ" "$JAVA_HOME_DIR" "jdk-11.0.25+9"
extract_once "$HADOOP_TGZ" "$HADOOP_HOME" "hadoop-3.3.6"
extract_once "$SPARK_TGZ" "$SPARK_HOME" "spark-3.5.5-bin-hadoop3"
extract_once "$PIG_TGZ" "$PIG_HOME" "pig-0.17.0"
extract_once "$HIVE_TGZ" "$HIVE_HOME" "apache-hive-4.0.1-bin"

export JAVA_HOME="$JAVA_HOME_DIR"
export PATH="$JAVA_HOME/bin:$PATH"

echo "== hadoop configs =="
CONF="$HADOOP_HOME/etc/hadoop"
printf 'localhost\n' >"$CONF/workers"

# Keep the vendor hadoop-env.sh; pin JAVA_HOME and Java 11 --add-opens.
sed -i '/^# >>> risk-topology/,/^# <<< risk-topology/d' "$CONF/hadoop-env.sh"
cat >>"$CONF/hadoop-env.sh" <<EOF
# >>> risk-topology
export JAVA_HOME=$JAVA_HOME_DIR
export HADOOP_HEAPSIZE=\${HADOOP_HEAPSIZE:-512}
export HADOOP_OPTS="--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED --add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED \${HADOOP_OPTS:-}"
# <<< risk-topology
EOF

write_xml "$CONF/core-site.xml" <<EOF
<?xml version="1.0"?>
<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://localhost:9000</value>
  </property>
  <property>
    <name>hadoop.tmp.dir</name>
    <value>file://$DATA</value>
  </property>
</configuration>
EOF

write_xml "$CONF/hdfs-site.xml" <<EOF
<?xml version="1.0"?>
<configuration>
  <property>
    <name>dfs.replication</name>
    <value>1</value>
  </property>
  <property>
    <name>dfs.namenode.name.dir</name>
    <value>file://$DATA/namenode</value>
  </property>
  <property>
    <name>dfs.datanode.data.dir</name>
    <value>file://$DATA/datanode</value>
  </property>
  <property>
    <name>dfs.permissions.enabled</name>
    <value>false</value>
  </property>
  <property>
    <name>dfs.namenode.datanode.registration.ip-hostname-check</name>
    <value>false</value>
  </property>
</configuration>
EOF

write_xml "$CONF/mapred-site.xml" <<EOF
<?xml version="1.0"?>
<configuration>
  <property>
    <name>mapreduce.framework.name</name>
    <value>yarn</value>
  </property>
  <property>
    <name>mapreduce.application.classpath</name>
    <value>$HADOOP_HOME/share/hadoop/mapreduce/*:$HADOOP_HOME/share/hadoop/mapreduce/lib/*</value>
  </property>
  <property>
    <name>yarn.app.mapreduce.am.env</name>
    <value>HADOOP_MAPRED_HOME=$HADOOP_HOME</value>
  </property>
  <property>
    <name>mapreduce.map.env</name>
    <value>HADOOP_MAPRED_HOME=$HADOOP_HOME</value>
  </property>
  <property>
    <name>mapreduce.reduce.env</name>
    <value>HADOOP_MAPRED_HOME=$HADOOP_HOME</value>
  </property>
  <property>
    <name>mapreduce.map.memory.mb</name>
    <value>512</value>
  </property>
  <property>
    <name>mapreduce.reduce.memory.mb</name>
    <value>512</value>
  </property>
  <property>
    <name>yarn.app.mapreduce.am.resource.mb</name>
    <value>512</value>
  </property>
  <property>
    <name>mapreduce.jobhistory.address</name>
    <value>127.0.0.1:10020</value>
  </property>
  <property>
    <name>mapreduce.jobhistory.webapp.address</name>
    <value>127.0.0.1:19888</value>
  </property>
  <property>
    <name>mapreduce.jobhistory.bind-host</name>
    <value>127.0.0.1</value>
  </property>
</configuration>
EOF

write_xml "$CONF/yarn-site.xml" <<EOF
<?xml version="1.0"?>
<configuration>
  <property>
    <name>yarn.nodemanager.aux-services</name>
    <value>mapreduce_shuffle</value>
  </property>
  <property>
    <name>yarn.nodemanager.env-whitelist</name>
    <value>JAVA_HOME,HADOOP_COMMON_HOME,HADOOP_HDFS_HOME,HADOOP_CONF_DIR,CLASSPATH_PREPEND_DISTCACHE,HADOOP_YARN_HOME,HADOOP_HOME,PATH,LANG,TZ,HADOOP_MAPRED_HOME</value>
  </property>
  <property>
    <name>yarn.nodemanager.resource.memory-mb</name>
    <value>4096</value>
  </property>
  <property>
    <name>yarn.scheduler.minimum-allocation-mb</name>
    <value>256</value>
  </property>
  <property>
    <name>yarn.scheduler.maximum-allocation-mb</name>
    <value>4096</value>
  </property>
  <property>
    <name>yarn.nodemanager.vmem-check-enabled</name>
    <value>false</value>
  </property>
  <property>
    <name>yarn.resourcemanager.hostname</name>
    <value>localhost</value>
  </property>
</configuration>
EOF

echo "== spark / pig / hive configs =="
cat >"$SPARK_HOME/conf/spark-env.sh" <<EOF
export JAVA_HOME=$JAVA_HOME_DIR
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export SPARK_LOCAL_IP=127.0.0.1
export SPARK_DRIVER_MEMORY=1g
EOF
chmod +x "$SPARK_HOME/conf/spark-env.sh"

cat >"$HIVE_HOME/conf/hive-env.sh" <<EOF
export JAVA_HOME=$JAVA_HOME_DIR
export HADOOP_HOME=$HADOOP_HOME
export HIVE_CONF_DIR=$HIVE_HOME/conf
export HADOOP_HEAPSIZE=512
export HADOOP_OPTS="--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED \${HADOOP_OPTS:-}"
EOF
chmod +x "$HIVE_HOME/conf/hive-env.sh"

write_xml "$HIVE_HOME/conf/hive-site.xml" <<EOF
<?xml version="1.0"?>
<configuration>
  <property>
    <name>javax.jdo.option.ConnectionURL</name>
    <value>jdbc:derby:;databaseName=$HIVE_META/metastore_db;create=true</value>
  </property>
  <property>
    <name>javax.jdo.option.ConnectionDriverName</name>
    <value>org.apache.derby.jdbc.EmbeddedDriver</value>
  </property>
  <property>
    <name>hive.metastore.warehouse.dir</name>
    <value>/user/hive/warehouse</value>
  </property>
  <property>
    <name>hive.metastore.schema.verification</name>
    <value>true</value>
  </property>
  <property>
    <name>datanucleus.schema.autoCreateAll</name>
    <value>false</value>
  </property>
  <property>
    <name>hive.metastore.uris</name>
    <value></value>
  </property>
  <property>
    <name>hive.execution.engine</name>
    <value>mr</value>
  </property>
  <property>
    <name>hive.exec.scratchdir</name>
    <value>/tmp/hive</value>
  </property>
  <property>
    <name>hive.exec.local.scratchdir</name>
    <value>$HOME/hive-tmp</value>
  </property>
  <property>
    <name>system:java.io.tmpdir</name>
    <value>$HOME/hive-tmp</value>
  </property>
  <property>
    <name>hive.uploaded.resources.dest.dir</name>
    <value>$HOME/hive-tmp/resources</value>
  </property>
  <property>
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
EOF

ENV_FILE="$HOME/.hadoop_env"
cat >"$ENV_FILE" <<EOF
export JAVA_HOME=$JAVA_HOME_DIR
export HADOOP_HOME=$HADOOP_HOME
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export SPARK_HOME=$SPARK_HOME
export PIG_HOME=$PIG_HOME
export HIVE_HOME=$HIVE_HOME
export HADOOP_MAPRED_HOME=$HADOOP_HOME
export HADOOP_COMMON_HOME=$HADOOP_HOME
export HADOOP_HDFS_HOME=$HADOOP_HOME
export YARN_HOME=$HADOOP_HOME
export HADOOP_OPTS="--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED"
export PATH="\$JAVA_HOME/bin:\$HADOOP_HOME/bin:\$HADOOP_HOME/sbin:\$SPARK_HOME/bin:\$PIG_HOME/bin:\$HIVE_HOME/bin:\$PATH"
EOF

if ! grep -q 'source \$HOME/.hadoop_env' "$HOME/.bashrc" 2>/dev/null; then
  printf '\n# Hadoop stack\n[ -f "$HOME/.hadoop_env" ] && source "$HOME/.hadoop_env"\n' >>"$HOME/.bashrc"
fi

echo
echo "Installed under $OPT"
echo "  Java  $( "$JAVA_HOME_DIR/bin/java" -version 2>&1 | head -1 )"
echo "  Hive  4.0.1 (runs on JDK 11; 4.1+ needs JDK 17)"
echo
echo "Next (from Ubuntu, not from Windows cmd):"
echo "  source ~/.hadoop_env"
echo "  bash /mnt/c/Users/ahuja/Desktop/minor_big_data/scripts/start_hadoop_wsl.sh"
echo "  bash /mnt/c/Users/ahuja/Desktop/minor_big_data/scripts/verify_hadoop_wsl.sh"
