# LLM Risk Topology

Does spectral boundary proximity in a response-embedding graph predict held-out behavior changes better than
simpler baselines? A separate model-lineage analysis studies exposure through declared and inferred Hub relationships;
lineage edges are metadata evidence, not proof that behavior transfers.

The stack has two paths:

- **Hugging Face collector → HDFS → Pig:** collect a pinned metadata snapshot, normalize records and edges.
- **Spark/GraphX → Hive:** compute lineage statistics and exposure estimates, then store queryable tables.
- **Frozen prompts → transformers or vLLM:** generate checkpoint-pinned responses with crash-resumable shards.
- **WildGuard and refusal rules:** label fresh responses; export a separate sample for human checking.
- **MiniLM → HDFS → Spark MLlib → Hive:** embed responses, stage Parquet, and run distributed clustering summaries.
- **Python analysis:** build response graphs, score spectral boundaries and baselines, and compare held-out outcomes
  using prompt groups for splits and resampling. The response spectral analysis is separate from lineage PageRank.

## Running the stages

Use Python 3.11 with `pip install -r requirements.txt` and a PyTorch installation suitable for your hardware.
The Hadoop/Pig/Spark/Hive commands require the configured WSL stack and `~/.hadoop_env`.
Run Python commands from the repository root; run the Bash commands from that root inside WSL.
Replace uppercase placeholders with the paths or identifiers from your run.

```sh
# Collect into a new snapshot folder; use --resume with the same arguments after interruption.
python python/collect_metadata.py --limit 1000 --output data/raw/metadata/NEW_SNAPSHOT
bash scripts/run_metadata_pipeline_wsl.sh data/raw/metadata/NEW_SNAPSHOT
# Use the HDFS root printed by the metadata pipeline.
bash scripts/run_exposure_wsl.sh METADATA_HDFS_RUN_ROOT

# Validate the existing frozen prompt set; do not regenerate over it.
python python/data_contracts.py validate data/processed/prompts/pilot_v5/experiment.bundle.json --verify-artifacts

# Laptop validation example. Keep the registered settings for comparisons.
python python/generate_responses.py --prompts data/processed/prompts/pilot_v5 --models manifests/models/laptop_qwen.bundle.json --model Qwen/Qwen2.5-1.5B-Instruct --backend transformers --split validation --batch-size 16 --max-new-tokens 256 --seed 42
python python/evaluate_responses.py label --run RUN_FOLDER --labeler refusal_rules

# Local embedding and cluster staging of a completed run.
python python/stage_layer2.py --runs RUN_FOLDER --output data/processed/analysis/layer2_flat_MODEL
bash scripts/run_layer2_mllib_wsl.sh data/processed/analysis/layer2_flat_MODEL

# After the saved Kaggle run ends: download just its results archive and ingest it.
python scripts/kaggle/ingest_scale8.py --kernel deepamahuja/llm-risk-topology-wildguard-v7 --version KERNEL_VERSION --output data/processed/responses/NEW_INGEST
python scripts/kaggle/behavior_summary.py --output report/pilot
python python/run_layer2.py --flat data/processed/analysis/layer2_flat_MODEL
python -m unittest discover -s tests
```

Kaggle preparation and execution code lives in `scripts/kaggle/`. GPU notebooks use their own pinned requirements.
Enable Internet, the requested GPU, and the Hugging Face secret in the web editor before saving a run.
The ingestion command requires local Kaggle credentials and a fresh output folder. It validates checksums and contracts,
stages completed models through the existing cluster pipeline, and preserves incomplete-run inventory for recovery.
Use `--archive PATH --scratch` instead of `--kernel` for an offline rehearsal that leaves production staging alone.

## Data, models, and layout

The frozen pilot uses existing SALAD-Data, WildJailbreak and XSTest prompts, preserves original/variant links, and splits
whole related-prompt groups. WildGuard training overlap is tracked separately. Public prompts can still have appeared
in assistant training. Historical dataset labels do not describe our newly generated answers.

Model manifests pin checkpoints and templates for Qwen, Mistral/Zephyr, Llama and their selected fine-tunes, plus the
small laptop models. Official Meta Llama weights and WildGuard require the relevant Hugging Face access terms and token.
Attach a read token as the notebook's Hugging Face secret; keep credentials out of source files and logs.
Bulk datasets, response shards, model weights and local reports are not distributed in Git.

- `python/`: collection, prompt preparation, contracts, generation, evaluation and analysis.
- `scripts/`, `pig/`, `mr/`, `spark/`: orchestration and distributed jobs.
- `schemas/`, `manifests/`, `fixtures/`, `tests/`: contracts, pinned experiment records and checks.
- `data/`: local raw and processed artifacts; `report/` and `project_notes/`: local evidence and planning.

Behavioral results are a pilot. Test-set answers are reserved; validation results and phrase-rule labels do not establish
the main hypothesis, evaluator accuracy, or generalization to unseen model families.
