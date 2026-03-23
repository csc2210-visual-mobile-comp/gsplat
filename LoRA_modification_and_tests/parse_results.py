"""
Parse benchmark results from:
  results/benchmark/lora/<scene>/<experiment>/stats/<stat_file>.json

and write an Excel table where:
  - Rows = (scene, experiment)
  - Columns grouped by split (train / val) and step
"""

import json
import re
from pathlib import Path

import pandas as pd

BENCHMARK_ROOT = Path(__file__).parent / "results" / "benchmark" / "lora"
OUTPUT_XLSX = Path(__file__).parent / "results" / "benchmark_summary.xlsx"


def parse_stats_dir(stats_dir: Path) -> dict:
    """Return {(split, step): {metric: value}} for all JSON files in stats_dir."""
    records = {}
    for json_file in sorted(stats_dir.glob("*.json")):
        name = json_file.stem  # e.g. "train_step1999_rank0" or "val_step6999"

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


def main():
    rows = []

    for scene_dir in sorted(BENCHMARK_ROOT.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene = scene_dir.name

        for exp_dir in sorted(scene_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            experiment = exp_dir.name
            stats_dir = exp_dir / "stats"
            if not stats_dir.exists():
                print(f"No stats dir: {exp_dir}")
                continue

            records = parse_stats_dir(stats_dir)
            if not records:
                continue

            row = {"scene": scene, "experiment": experiment}
            for (split, step), metrics in sorted(records.items()):
                for metric, value in metrics.items():
                    col = f"{split}/step{step}/{metric}"
                    row[col] = value

            rows.append(row)

    if not rows:
        print("No data found.")
        return

    df = pd.DataFrame(rows)

    # Sort rows by scene then experiment
    df = df.sort_values(["scene", "experiment"]).reset_index(drop=True)

    # Put scene/experiment first, then columns sorted by split→step→metric
    fixed_cols = ["scene", "experiment"]
    stat_cols = sorted(
        [c for c in df.columns if c not in fixed_cols],
        key=lambda c: (
            0 if c.startswith("train") else 1,   # train before val
            int(re.search(r"step(\d+)", c).group(1)),
            c,
        ),
    )
    df = df[fixed_cols + stat_cols]

    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="all")

        # Also write one sheet per scene for easier comparison
        for scene, group in df.groupby("scene"):
            group = group.drop(columns=["scene"]).reset_index(drop=True)
            group.to_excel(writer, index=False, sheet_name=scene[:31])  # sheet name limit

        # Auto-fit column widths
        for sheet in writer.sheets.values():
            for col in sheet.columns:
                max_len = max(len(str(cell.value or "")) for cell in col)
                sheet.column_dimensions[col[0].column_letter].width = min(max_len + 2, 40)

    print(f"Written {len(rows)} rows → {OUTPUT_XLSX}")
    print(f"Columns: {stat_cols}")


if __name__ == "__main__":
    main()
