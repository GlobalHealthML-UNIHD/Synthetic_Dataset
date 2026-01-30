from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path

from generator import generate


def _load_scene(scene_module: str) -> dict:
    mod = importlib.import_module(scene_module)
    if not hasattr(mod, "SCENE"):
        raise AttributeError(f"Scene module '{scene_module}' does not define SCENE")
    scene = mod.SCENE
    if not isinstance(scene, dict):
        raise TypeError(f"SCENE in '{scene_module}' must be a dict, got {type(scene)}")
    return scene


def _safe_name(s: str) -> str:
    # safe folder name
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in s).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="Python module path, e.g. toy_data_set.scenes.scene_example")
    ap.add_argument(
        "--outroot",
        default="" \
        "synthetic_data_set/experiments",
        help="Root folder for outputs (default: synthetic_data_set/experiments)",
    )
    ap.add_argument(
        "--no-cluster",
        action="store_true",
        help="Only generate data.csv and manifest.json; skip clustering pipeline",
    )
    args = ap.parse_args()

    # 1) Load scene
    scene = _load_scene(args.scene)
    scene_name = _safe_name(scene.get("name", args.scene.split(".")[-1]))

    # 2) Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outroot) / f"{scene_name}__{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    # 3) Generate and save data
    df = generate(scene)
    csv_path = outdir / "data.csv"
    df.to_csv(csv_path, index=False)

    # 4) Save manifest (scene + paths)
    manifest = {
        "scene_module": args.scene,
        "scene_name": scene_name,
        "timestamp": timestamp,
        "output_dir": str(outdir),
        "data_csv": str(csv_path),
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "scene": scene,  # full scene snapshot for reproducibility
    }
    with open(outdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Scene: {scene_name}")
    print(f"[OK] Wrote dataset: {csv_path}  shape={df.shape}")
    print(f"[OK] Wrote manifest: {outdir / 'manifest.json'}")

    # 5) Run clustering pipeline (optional)
    if args.no_cluster:
        print("[SKIP] --no-cluster set, not running clustering pipeline.")
        return

    try:
        from toy_cluster import run_pipeline
    except Exception as e:
        raise ImportError(
            "Could not import run_pipeline from toy_cluster.py.\n"
            "Fix: in toy_cluster.py, wrap your existing script into a function:\n"
            "  def run_pipeline(data_csv, outdir): ...\n"
        ) from e

    run_pipeline(data_csv=csv_path, outdir=outdir)
    print(f"[DONE] Results in: {outdir}")


if __name__ == "__main__":
    main()
