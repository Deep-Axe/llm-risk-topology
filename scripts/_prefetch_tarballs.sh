#!/usr/bin/env bash
set -euo pipefail
SRC="$HOME/opt/src"
mkdir -p "$SRC"

dl() {
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then
    echo "have $(basename "$dest")"
    return 0
  fi
  echo "START $(basename "$dest")"
  curl -L --fail --retry 8 --retry-all-errors --retry-delay 3 -C - --progress-bar \
    -o "${dest}.part" "$url"
  mv -f "${dest}.part" "$dest"
  echo "DONE $(basename "$dest")"
}

# Hadoop is already being fetched by the frozen installer curl — do not touch it.
dl "https://archive.apache.org/dist/spark/spark-3.5.5/spark-3.5.5-bin-hadoop3.tgz" "$SRC/spark-3.5.5-bin-hadoop3.tgz" &
dl "https://archive.apache.org/dist/pig/pig-0.17.0/pig-0.17.0.tar.gz" "$SRC/pig-0.17.0.tar.gz" &
dl "https://archive.apache.org/dist/hive/hive-4.0.1/apache-hive-4.0.1-bin.tar.gz" "$SRC/apache-hive-4.0.1-bin.tar.gz" &
wait
echo "prefetch finished"
ls -lh "$SRC"
