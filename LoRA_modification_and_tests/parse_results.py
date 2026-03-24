import json
import re
from pathlib import Path

import pandas as pd

BASE_ROOT = Path(__file__).parent / "results" / "benchmark"
OUTPUT_XLSX = BASE_ROOT / "benchmark_summary.xlsx"

BENCHMARK_TYPES = ["lora", "partial_lora", "original"]


def parse_stats_dir(stats_dir: Path) -> dict:
    records = {}
    for json_file in sorted(stats_dir.glob("*.json")):
        name = json_file.stem

        train_match = re.match(r"train_step(\d+)_rank0", name)
        val_match = re.match(r"val_step(\d+)", name)

        if train_match:
            split, step = "train", int(train_match.group(1))
        elif val_match:
            split, step = "val", int(val_match.group(1))
        else:
            print(f"  Skipping unrecognised file: {json_file.name}")
            continue

        with open(json_file) as f:
            data = json.load(f)

        records[(split, step)] = data

    return records


# --- NEW: experiment normalization logic ---
def normalize_experiment(bench_type: str, scene: str, experiment: str) -> str:
    if bench_type == "lora":
        replacements = {
            "lora_d_high": "lora_d_high_memory_opt",
            "lora_d_high_original": "lora_d_high_cosmo_branch",
            "lora_d_low": "lora_d_low_memory_opt",
            "lora_d_low_original": "lora_d_low_cosmo_branch",
            "lora_d_mid": "lora_d_mid_memory_opt",
            "lora_d_mid_original": "lora_d_mid_cosmo_branch",
        }
        return replacements.get(experiment, experiment)

    elif bench_type == "partial_lora":
        # remove scene prefix before first underscore
        # e.g. gardenspheres_colors -> colors
        parts = experiment.split("_", 1)
        suffix = parts[1] if len(parts) > 1 else experiment
        return f"partial_lora_{suffix}"

    elif bench_type == "original":
        return experiment

    return experiment


def collect_rows(benchmark_root: Path, bench_type: str):
    rows = []

    for scene_dir in sorted(benchmark_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene = scene_dir.name

        for exp_dir in sorted(scene_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            experiment = normalize_experiment(
                bench_type, scene, exp_dir.name
            )

            stats_dir = exp_dir / "stats"
            if not stats_dir.exists():
                continue

            records = parse_stats_dir(stats_dir)
            if not records:
                continue

            row = {
                "scene": scene,
                "experiment": experiment,
                "type": bench_type,  # helpful for filtering later
            }

            for (split, step), metrics in sorted(records.items()):
                for metric, value in metrics.items():
                    col = f"{split}/step{step}/{metric}"
                    row[col] = value

            rows.append(row)

    return rows


def main():
    all_rows = []

    for bench_type in BENCHMARK_TYPES:
        root = BASE_ROOT / bench_type
        if not root.exists():
            print(f"Skipping missing: {root}")
            continue

        rows = collect_rows(root, bench_type)
        all_rows.extend(rows)

    if not all_rows:
        print("No data found.")
        return

    df = pd.DataFrame(all_rows)

    # Sort rows
    df = df.sort_values(["scene", "type", "experiment"]).reset_index(drop=True)

    # Column ordering
    fixed_cols = ["scene", "type", "experiment"]
    stat_cols = sorted(
        [c for c in df.columns if c not in fixed_cols],
        key=lambda c: (
            0 if c.startswith("train") else 1,
            int(re.search(r"step(\d+)", c).group(1)),
            c,
        ),
    )

    df = df[fixed_cols + stat_cols]

    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="all")

        # Auto-fit columns
        sheet = writer.sheets["all"]
        for col in sheet.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            sheet.column_dimensions[col[0].column_letter].width = min(max_len + 2, 40)

    print(f"Written {len(df)} rows → {OUTPUT_XLSX}")
    print(f"Columns: {stat_cols}")


if __name__ == "__main__":
    main()