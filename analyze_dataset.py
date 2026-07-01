"""
Member 3 – Dataset Audit Script
Run this from the directory containing your .npz files:
    python analyze_dataset.py --data_dir /path/to/npz/files --split splits.json
"""
import argparse
import json
import os
import numpy as np
from collections import defaultdict


def analyze(data_dir: str, split_path: str):
    with open(split_path) as f:
        splits = json.load(f)

    results = {}
    for split_name, files in splits.items():
        labels, shapes, mins, maxs = [], [], [], []
        missing, zero_patches = [], []

        for fname in files:
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                missing.append(fname)
                continue
            try:
                d = np.load(fpath)
                patch = d["patch"].astype(np.float32)
                lbl = int(d["label"])
                labels.append(lbl)
                shapes.append(patch.shape)
                mins.append(float(patch.min()))
                maxs.append(float(patch.max()))
                if patch.max() == 0.0:
                    zero_patches.append(fname)
            except Exception as e:
                print(f"  ERROR loading {fname}: {e}")

        n_total = len(labels)
        n_malignant = sum(labels)
        n_benign = n_total - n_malignant

        results[split_name] = {
            "total": n_total,
            "malignant": n_malignant,
            "benign": n_benign,
            "malignant_pct": 100 * n_malignant / max(n_total, 1),
            "missing_files": missing,
            "zero_patches": zero_patches,
            "patch_shape": shapes[0] if shapes else "N/A",
            "intensity_range": (min(mins), max(maxs)) if mins else "N/A",
        }

    # Print report
    print("=" * 60)
    print("DATASET AUDIT REPORT")
    print("=" * 60)
    total_all = sum(r["total"] for r in results.values())
    mal_all = sum(r["malignant"] for r in results.values())
    ben_all = sum(r["benign"] for r in results.values())

    for split_name, r in results.items():
        print(f"\n[{split_name.upper()}]")
        print(f"  Total nodules   : {r['total']}")
        print(f"  Malignant (1)   : {r['malignant']} ({r['malignant_pct']:.1f}%)")
        print(f"  Benign    (0)   : {r['benign']} ({100-r['malignant_pct']:.1f}%)")
        print(f"  Patch shape     : {r['patch_shape']}")
        print(f"  Intensity range : {r['intensity_range']}")
        if r["missing_files"]:
            print(f"  MISSING FILES   : {len(r['missing_files'])} — {r['missing_files'][:5]}")
        if r["zero_patches"]:
            print(f"  ZERO PATCHES    : {len(r['zero_patches'])} files have max=0.0 (preprocessing issue)")

    print(f"\n[OVERALL]")
    print(f"  Total           : {total_all}")
    print(f"  Malignant       : {mal_all} ({100*mal_all/max(total_all,1):.1f}%)")
    print(f"  Benign          : {ben_all} ({100*ben_all/max(total_all,1):.1f}%)")
    print(f"\n[PAPER COMPARISON]")
    print(f"  Paper nodules   : 5,926 (across 1,330 LDCT scans + 312 institutional)")
    print(f"  Our nodules     : {total_all} (across ~250 patients, LIDC-IDRI only)")
    print(f"  Dataset ratio   : {100*total_all/5926:.1f}% of paper dataset size")
    print(f"\n  This is the primary reproducibility constraint.")

    # Save JSON summary
    out = {
        "splits": results,
        "overall": {"total": total_all, "malignant": mal_all, "benign": ben_all},
    }
    with open("dataset_audit.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: dataset_audit.json")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", default="splits.json")
    args = parser.parse_args()
    analyze(args.data_dir, args.split)
