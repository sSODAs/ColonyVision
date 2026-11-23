from __future__ import annotations

import numpy as np


METHODS = ("YOLO_baseline", "Step4_refined", "Final_iterative_resolved")


def density_band(n: int):
    if n <= 10:
        return "1-10"
    if n <= 50:
        return "11-50"
    if n <= 100:
        return "51-100"
    if n <= 200:
        return "101-200"
    return ">200"

def aggregate(rows):
    tp = sum(int(r["tp"]) for r in rows)
    fp = sum(int(r["fp"]) for r in rows)
    fn = sum(int(r["fn"]) for r in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    abs_err = np.asarray([float(r["count_abs_error"]) for r in rows], dtype=float)
    sq_err = np.asarray([float(r["count_error"]) ** 2 for r in rows], dtype=float)
    rel_err = np.asarray([float(r["relative_error_pct"]) for r in rows], dtype=float)

    return {
        "images": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": float(np.mean([float(r["f1"]) for r in rows])),
        "count_mae": float(np.mean(abs_err)),
        "count_rmse": float(np.sqrt(np.mean(sq_err))),
        "mean_relative_count_error_pct": float(np.mean(rel_err)),
    }

def make_eval_graphs(eval_dir: Path, aggregate_rows, density_rows, per_image):
    import matplotlib.pyplot as plt

    graph_dir = eval_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    method_labels = ["YOLO baseline", "Refined", "Final"]
    x = np.arange(len(method_labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    ax.set_facecolor("white")
    for i, key in enumerate(("precision", "recall", "f1")):
        vals = [100 * float(aggregate_rows[j][key]) for j in range(3)]
        ax.bar(x + (i - 1) * width, vals, width, label=key.capitalize())
    ax.set_xticks(x, method_labels)
    ax.set_ylim(70, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("Test50 | Detection performance", loc="left")
    ax.legend()
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(graph_dir / "01_precision_recall_f1.png", dpi=180, facecolor="white")
    plt.close(fig)

    final_rows = [r for r in per_image if r["method"] == "Final_iterative_resolved"]
    gt = np.asarray([float(r["gt_in_roi"]) for r in final_rows])
    pred = np.asarray([float(r["pred_count"]) for r in final_rows])
    mx = max(float(gt.max()), float(pred.max())) * 1.03

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(gt, pred, s=38)
    ax.plot([0, mx], [0, mx], linestyle="--")
    ax.set_xlim(0, mx)
    ax.set_ylim(0, mx)
    ax.set_xlabel("Ground Truth count")
    ax.set_ylabel("Predicted count")
    ax.set_title("Final model | Ground Truth vs Predicted", loc="left")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(graph_dir / "02_ground_truth_vs_prediction.png", dpi=180, facecolor="white")
    plt.close(fig)

    bands = ["1-10", "11-50", "51-100", "101-200", ">200"]
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    ax.set_facecolor("white")
    for method, label in zip(METHODS, method_labels):
        vals = []
        for band in bands:
            row = next(
                (r for r in density_rows if r["method"] == method and r["density"] == band),
                None,
            )
            vals.append(100 * float(row["f1"]) if row else np.nan)
        ax.plot(bands, vals, marker="o", linewidth=2, label=label)
    ax.set_ylim(75, 100)
    ax.set_xlabel("Ground Truth colonies in evaluation ROI")
    ax.set_ylabel("F1-score (%)")
    ax.set_title("F1-score by colony density", loc="left")
    ax.legend()
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(graph_dir / "03_f1_by_density.png", dpi=180, facecolor="white")
    plt.close(fig)

    ids = [r["eval_id"] for r in final_rows]
    err = np.asarray([float(r["count_abs_error"]) for r in final_rows])
    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor="white")
    ax.set_facecolor("white")
    ax.bar(ids, err)
    ax.set_xlabel("Test image")
    ax.set_ylabel("Absolute count error")
    ax.set_title("Final model | Absolute count error per plate", loc="left")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(graph_dir / "04_absolute_count_error.png", dpi=180, facecolor="white")
    plt.close(fig)
