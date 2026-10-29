"""Sanity-check the risk-topology environment."""
from __future__ import annotations

import os
import sys


def main() -> int:
    print(f"python        {sys.version.split()[0]}  ({sys.executable})")
    print(f"JAVA_HOME     {os.environ.get('JAVA_HOME', '(unset)')}")

    missing: list[str] = []
    versions: dict[str, str] = {}

    checks = [
        ("huggingface_hub", "huggingface_hub"),
        ("datasets", "datasets"),
        ("transformers", "transformers"),
        ("sentence_transformers", "sentence_transformers"),
        ("torch", "torch"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pyarrow", "pyarrow"),
        ("networkx", "networkx"),
        ("sklearn", "sklearn"),
        ("pyspark", "pyspark"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("tqdm", "tqdm"),
        ("yaml", "yaml"),
        ("jsonlines", "jsonlines"),
    ]

    for label, mod in checks:
        try:
            m = __import__(mod)
            versions[label] = getattr(m, "__version__", "ok")
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{label}: {exc}")

    for label, ver in versions.items():
        print(f"{label:<24} {ver}")

    try:
        import torch

        print(f"torch.cuda               {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"torch.device             {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

    try:
        import pyspark
        from pyspark.sql import SparkSession

        expected_spark = "3.5.5"
        if pyspark.__version__ != expected_spark:
            missing.append(
                f"pyspark {pyspark.__version__} != WSL Spark {expected_spark}; "
                "pin the Windows client before using it against the WSL runtime"
            )
        print("spark jobs               GraphX/Pregel/MLlib stay in WSL Spark 3.5.5")

        spark = (
            SparkSession.builder.master("local[1]")
            .appName("env-check")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate()
        )
        n = spark.range(3).count()
        print(f"pyspark local            ok (range count={n})")
        spark.stop()
    except Exception as exc:  # noqa: BLE001
        missing.append(f"pyspark runtime: {exc}")

    if missing:
        print("\nMISSING / FAILED")
        for line in missing:
            print(f"  - {line}")
        return 1

    print("\nall imports ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
