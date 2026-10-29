"""Inspect bounded prompt-dataset slices; never download complete dataset shards.

Run from the repository root with .venv/Scripts/python.exe.
Evidence and sample content go in the locally excluded report/ directory.
HF viewer samples describe its current conversion, not a pinned historical export.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import gzip
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from urllib.parse import urlparse

import requests
from huggingface_hub import get_token
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
HF = [
    "allenai/wildjailbreak", "lmsys/lmsys-chat-1m",
    "PKU-Alignment/BeaverTails", "PKU-Alignment/PKU-SafeRLHF",
    "allenai/wildguardmix", "OpenSafetyLab/Salad-Data",
    "Anthropic/hh-rlhf", "LibrAI/do-not-answer", "abisee/cnn_dailymail",
]
GITHUB = {
    "paul-rottger/xstest": ["xstest_prompts.csv"],
    "centerforaisafety/HarmBench": [
        "data/behavior_datasets/harmbench_behaviors_text_all.csv",
        "data/behavior_datasets/harmbench_behaviors_text_val.csv",
        "data/behavior_datasets/harmbench_behaviors_text_test.csv",
        "data/behavior_datasets/harmbench_behaviors_multimodal_all.csv",
    ],
    "llm-attacks/llm-attacks": [
        "data/advbench/harmful_behaviors.csv", "data/advbench/harmful_strings.csv",
    ],
    "liu00222/Open-Prompt-Injection": [],
}
LIMIT = 2_000_000


def fetch(url, params=None):
    """Bound response bytes, close early; never log credential headers."""
    headers = {"User-Agent": "minor-big-data-schema-inspector/1"}
    if urlparse(url).hostname in {"huggingface.co", "datasets-server.huggingface.co"}:
        token = get_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
    with requests.get(url, params=params, headers=headers, timeout=(10, 30), stream=True) as r:
        r.raise_for_status()
        chunks, size = [], 0
        for chunk in r.iter_content(16384):
            size += len(chunk)
            if size > LIMIT:
                raise ValueError(f"Response exceeds {LIMIT} byte inspection limit")
            chunks.append(chunk)
        return b"".join(chunks).decode("utf-8-sig"), r.headers


def get_json(url, params=None):
    return json.loads(fetch(url, params)[0])


def attempt(fn):
    try:
        return fn()
    except requests.RequestException as exc:
        response = exc.response
        request = exc.request
        host = urlparse(request.url).hostname if request is not None else "unknown host"
        status = f"HTTP {response.status_code}" if response is not None else type(exc).__name__
        # Redirect errors can contain signed download URLs; do not save those.
        return {"error": f"{status} from {host}"}
    except (ValueError, KeyError) as exc:
        return {"error": str(exc)}


def hf_source(repo, count):
    meta = get_json(f"https://huggingface.co/api/datasets/{repo}")
    sha = meta["sha"]
    result = {
        "source": repo, "provider": "huggingface", "revision": sha,
        "url": f"https://huggingface.co/datasets/{repo}",
        "gated": meta.get("gated"), "card_data": meta.get("cardData", {}),
        "files": [s["rfilename"] for s in meta.get("siblings", [])],
        "card": attempt(lambda: fetch(f"https://huggingface.co/datasets/{repo}/resolve/{sha}/README.md")[0]),
        "viewer_revision_note": "Viewer conversion sampled at inspection time; not guaranteed pinned to repository SHA.",
        "samples": [],
    }
    splits = attempt(lambda: get_json("https://datasets-server.huggingface.co/splits", {"dataset": repo}))
    result["viewer_splits"] = splits
    for split in splits.get("splits", []):
        params = {"dataset": repo, "config": split["config"], "split": split["split"], "offset": 0, "length": count}
        data = attempt(lambda: get_json("https://datasets-server.huggingface.co/rows", params))
        result["samples"].append({"config": split["config"], "split": split["split"], **data})
    if repo == "allenai/wildjailbreak" and not any(
        s.get("config") == "train" and s.get("rows") for s in result["samples"]
    ):
        result["samples"].append({"path": "train/train.tsv", "method": "bounded streamed CSV head",
                                  **attempt(lambda: csv_head(f"https://huggingface.co/datasets/{repo}/resolve/{sha}/train/train.tsv", count))})
    if repo == "allenai/wildjailbreak" and not any(s.get("config") == "eval" and s.get("rows") for s in result["samples"]):
        result["samples"].append({"path": "eval/eval.tsv", "method": "bounded streamed CSV head",
                                  **attempt(lambda: csv_head(f"https://huggingface.co/datasets/{repo}/resolve/{sha}/eval/eval.tsv", count))})
    if repo == "Anthropic/hh-rlhf":
        path = "red-team-attempts/red_team_attempts.jsonl.gz"
        result["samples"].append({"path": path, "method": "bounded streamed gzip JSONL head",
                                  **attempt(lambda: jsonl_gzip_head(f"https://huggingface.co/datasets/{repo}/resolve/{sha}/{path}", count))})
    if repo in {"lmsys/lmsys-chat-1m", "allenai/wildguardmix"}:
        parquet_files = [p for p in result["files"] if p.endswith(".parquet")]
        # One schema per WildGuard split; a single LMSYS shard suffices for schema.
        if repo.startswith("lmsys/"):
            parquet_files = parquet_files[:1]
        result["parquet_footers"] = [
            {"path": p, **attempt(lambda p=p: parquet_footer(f"https://huggingface.co/datasets/{repo}/resolve/{sha}/{p}"))}
            for p in parquet_files
        ]
    return result


def parquet_footer(url):
    """Read at most 64 KiB of a Parquet footer, with no row-group download."""
    headers = {"Range": "bytes=-65536", "User-Agent": "minor-big-data-schema-inspector/1"}
    token = get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with requests.get(url, headers=headers, timeout=(10, 30), stream=True) as r:
        r.raise_for_status()
        if r.status_code != 206:
            raise ValueError("Server did not honor Range; refused bulk file download")
        data = bytearray()
        for chunk in r.iter_content(8192):
            data.extend(chunk)
            if len(data) > 65536:
                raise ValueError("Footer exceeds 64 KiB bound")
    if data[-4:] != b"PAR1":
        raise ValueError("Missing Parquet footer magic")
    size = int.from_bytes(data[-8:-4], "little")
    if size + 8 > len(data):
        raise ValueError("Parquet metadata larger than bounded footer window")
    metadata = pq.read_metadata(pa.BufferReader(b"PAR1" + bytes(data[-size-8:])))
    schema = metadata.schema.to_arrow_schema()
    return {"fields": [{"name": f.name, "type": str(f.type), "nullable": f.nullable} for f in schema],
            "shard_rows": metadata.num_rows, "bytes_read": len(data),
            "evidence_note": "Pinned file footer schema only; no data rows read."}


def jsonl_gzip_head(url, count):
    with requests.get(url, timeout=(10, 30), stream=True) as r:
        r.raise_for_status()
        rows, total, pending = [], 0, ""
        started, array = False, False
        decoder = json.JSONDecoder()
        with gzip.GzipFile(fileobj=r.raw) as gz:
            while len(rows) < count:
                line = gz.readline(LIMIT - total + 1)
                total += len(line)
                if total > LIMIT:
                    raise ValueError("JSONL head exceeds decompressed byte limit")
                if not line:
                    if pending.strip():
                        raise ValueError("Incomplete JSON record at end of stream")
                    break
                pending += line.decode("utf-8")
                # Despite a .jsonl suffix, some upstream files contain a JSON array.
                if not started and pending.strip():
                    pending = pending.lstrip()
                    array = pending.startswith("[")
                    if array:
                        pending = pending[1:]
                    started = True
                while pending.strip() and len(rows) < count:
                    pending = pending.lstrip(", \r\n\t") if array else pending.lstrip()
                    if array and pending.startswith("]"):
                        pending = ""
                        break
                    if not pending:
                        break
                    try:
                        row, end = decoder.raw_decode(pending)
                    except json.JSONDecodeError:
                        break
                    rows.append(row)
                    pending = pending[end:]
        return {"fields": list(rows[0]) if rows else [], "rows": rows, "decompressed_bytes": total}


def csv_head(url, count):
    """Parse only header and requested rows, closing a potentially large stream."""
    headers = {"User-Agent": "minor-big-data-schema-inspector/1"}
    if urlparse(url).hostname == "huggingface.co" and get_token():
        headers["Authorization"] = f"Bearer {get_token()}"
    with requests.get(url, headers=headers, timeout=(10, 30), stream=True) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        # Bounded binary reads also cover multiline CSV fields.
        buffer = bytearray()
        for chunk in r.iter_content(4096):
            buffer.extend(chunk)
            if len(buffer) > LIMIT:
                raise ValueError("CSV preview exceeds inspection byte limit")
            text = buffer.decode("utf-8-sig", errors="replace")
            reader = csv.DictReader(io.StringIO(text), delimiter="\t" if url.endswith('.tsv') else ',')
            rows = []
            try:
                for row in reader:
                    rows.append(row)
                    if len(rows) >= count + 1:
                        return {"fields": reader.fieldnames, "rows": rows[:count], "bytes_read": len(buffer)}
            except csv.Error:
                continue
        return {"fields": reader.fieldnames, "rows": rows[:count], "bytes_read": len(buffer)}


def github_source(repo, paths, count):
    commit = get_json(f"https://api.github.com/repos/{repo}/commits/HEAD")["sha"]
    tree = get_json(f"https://api.github.com/repos/{repo}/git/trees/{commit}", {"recursive": "1"})
    base = f"https://raw.githubusercontent.com/{repo}/{commit}/"
    result = {"source": repo, "provider": "github", "revision": commit,
              "tree_sha": tree["sha"], "url": f"https://github.com/{repo}",
              "card": attempt(lambda: fetch(base + "README.md")[0]), "samples": []}
    for path in paths:
        result["samples"].append({"path": path, **attempt(lambda: csv_head(base + path, count))})
    if repo.endswith("Open-Prompt-Injection"):
        selected = [x["path"] for x in tree["tree"] if x["type"] == "blob" and
                    (x["path"].startswith("configs/task_configs/") or
                     x["path"].startswith("OpenPromptInjection/tasks/") or
                     "instruction" in x["path"].lower()) and x.get("size", LIMIT) < 100_000]
        result["task_artifacts"] = {p: attempt(lambda p=p: fetch(base + p)[0]) for p in selected}
        result["evidence_note"] = "Task configurations and loader code inspected; no underlying task corpus downloaded. Current commit includes post-2024 extensions."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--only", nargs="*", help="Exact source IDs; omitted means every catalog entry")
    parser.add_argument("--output", type=Path, default=ROOT / "report" / "dataset_schemas")
    args = parser.parse_args()
    if not 1 <= args.rows <= 10:
        parser.error("--rows must be between 1 and 10")
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(repo, hf_source, (repo, args.rows)) for repo in HF]
    jobs += [(repo, github_source, (repo, paths, args.rows)) for repo, paths in GITHUB.items()]
    jobs = [job for job in jobs if not args.only or job[0] in args.only]
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = {pool.submit(attempt, lambda fn=fn, a=a: fn(*a)): repo for repo, fn, a in jobs}
        for future in as_completed(tasks):
            repo = tasks[future]
            result = {"source": repo, "inspected_at": datetime.now(timezone.utc).isoformat(),
                      "requested_rows_per_partition": args.rows, "max_response_bytes": LIMIT,
                      "sampling_note": "First rows for structure inspection only; not representative or evaluation data.",
                      **future.result()}
            path = args.output / (repo.replace("/", "__") + ".json")
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            sampled = sum(len(s.get("rows", [])) for s in result.get("samples", []))
            errors = [s.get("error") for s in result.get("samples", []) if s.get("error")]
            print(json.dumps({"source": repo, "sampled_rows": sampled, "partitions": len(result.get("samples", [])),
                              "errors": errors or result.get("error") or result.get("viewer_splits", {}).get("error")}, ensure_ascii=False), flush=True)
    write_inventory(args.output)


def write_inventory(output):
    """Publish only provenance and schemas to the trackable manifest, never rows."""
    inventory = []
    for path in sorted(output.glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        entry = {k: d.get(k) for k in ["source", "provider", "url", "revision", "inspected_at"]}
        entry["stretch"] = d["source"] in {"abisee/cnn_dailymail", "liu00222/Open-Prompt-Injection"}
        entry["sampled_rows"] = sum(len(s.get("rows", [])) for s in d.get("samples", []))
        entry["partitions"] = []
        for s in d.get("samples", []):
            entry["partitions"].append({
                **{k: s[k] for k in ["config", "split", "path", "method", "num_rows_total", "features", "fields", "error"] if k in s},
                "sampled_rows": len(s.get("rows", [])),
            })
        entry["declared_dataset_info"] = d.get("card_data", {}).get("dataset_info")
        entry["declared_configs"] = d.get("card_data", {}).get("configs")
        entry["license"] = d.get("card_data", {}).get("license")
        entry["parquet_footers"] = d.get("parquet_footers", [])
        entry["viewer_revision_note"] = d.get("viewer_revision_note")
        entry["task_artifact_paths"] = list(d.get("task_artifacts", {}))
        entry["errors"] = [d[k]["error"] for k in ["viewer_splits", "card"] if isinstance(d.get(k), dict) and "error" in d[k]]
        inventory.append(entry)
    destination = ROOT / "manifests" / "dataset_schema_inventory.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps({"sources": inventory, "note": "Schema inspection only. First-row samples are not representative. Viewer exports are not SHA-pinned; source files/cards use recorded revisions."}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
