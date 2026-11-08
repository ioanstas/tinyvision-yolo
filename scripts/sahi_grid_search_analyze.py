"""
SAHI Grid Search - Analyze Results
===================================
Reads Stage 1 grid search results and ranks combinations by multiple metrics.
Recommends the best configuration to carry forward to Stage 2.

Usage:
    python scripts/sahi_grid_search_analyze.py
    python scripts/sahi_grid_search_analyze.py --dir outputs/sahi_grid_search/stage1_p2_baseline_20260215_143000

If --dir is not specified, automatically finds the most recent stage1 folder.
"""

import argparse
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
GRID_SEARCH_DIR = PROJECT_ROOT / "outputs" / "sahi_grid_search"


def find_latest_stage1_dir():
    """Find the most recent stage1 results folder."""
    if not GRID_SEARCH_DIR.exists():
        return None
    stage1_dirs = sorted(
        [d for d in GRID_SEARCH_DIR.iterdir() if d.is_dir() and d.name.startswith("stage1_")],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return stage1_dirs[0] if stage1_dirs else None


def load_results(results_dir):
    """Load summary.json from a grid search run."""
    summary_file = results_dir / "summary.json"
    if not summary_file.exists():
        print(f"ERROR: summary.json not found in {results_dir}")
        return None
    with open(summary_file, 'r') as f:
        return json.load(f)


def analyze(results_dir):
    results_dir = Path(results_dir)
    summary = load_results(results_dir)
    if summary is None:
        return

    baseline = summary['baseline']
    results = summary['results']

    print("=" * 90)
    print("SAHI GRID SEARCH - STAGE 1 ANALYSIS")
    print("=" * 90)
    print(f"\nModel: {summary['model']}")
    print(f"Split: {summary['split']}")
    print(f"Results dir: {results_dir}")
    print(f"Total combinations: {len(results)}")

    # =========================================================================
    # Baseline
    # =========================================================================
    bm = baseline
    print(f"\n--- Baseline (Standard Inference) ---")
    print(f"  mAP50:        {bm['overall']['mAP50']:.4f}")
    print(f"  Weighted mAP: {bm['weighted_map50']:.4f}")
    for size in ["tiny", "small", "medium", "large"]:
        print(f"  {size:<8} AP50: {bm['size_metrics'][size]['ap']:.4f}")

    # =========================================================================
    # Ranking by different criteria
    # =========================================================================
    rankings = {
        'weighted_map50': ('Weighted mAP50 (overall best)', lambda r: r['weighted_map50']),
        'tiny_ap': ('Tiny AP50 (best for tiny objects)', lambda r: r['size_metrics']['tiny']['ap']),
        'mAP50': ('Overall mAP50', lambda r: r['overall']['mAP50']),
        'small_ap': ('Small AP50', lambda r: r['size_metrics']['small']['ap']),
        'tiny_small_avg': ('Avg(Tiny + Small) AP50', lambda r: (r['size_metrics']['tiny']['ap'] + r['size_metrics']['small']['ap']) / 2),
    }

    best_configs = {}
    for key, (label, sort_fn) in rankings.items():
        sorted_r = sorted(results, key=sort_fn, reverse=True)
        best = sorted_r[0]
        best_configs[key] = best

        print(f"\n--- Best by {label} ---")
        print(f"  Config: slice_size={best['slice_size']}, overlap={best['overlap']}")
        print(f"  mAP50:        {best['overall']['mAP50']:.4f}  ({best['improvements_vs_standard']['mAP50_diff']:+.4f})")
        print(f"  Weighted mAP: {best['weighted_map50']:.4f}  ({best['improvements_vs_standard']['weighted_map50_diff']:+.4f})")
        print(f"  Tiny AP50:    {best['size_metrics']['tiny']['ap']:.4f}  ({best['improvements_vs_standard']['tiny_ap_diff']:+.4f})")
        print(f"  Small AP50:   {best['size_metrics']['small']['ap']:.4f}  ({best['improvements_vs_standard']['small_ap_diff']:+.4f})")
        print(f"  Time:         {best['inference_time_s']}s")

    # =========================================================================
    # Full table sorted by weighted mAP
    # =========================================================================
    print(f"\n{'=' * 90}")
    print("FULL RESULTS TABLE (sorted by Weighted mAP50)")
    print(f"{'=' * 90}")

    header = f"{'Rank':<5} {'slice':>5} {'ovlp':>5} {'mAP50':>8} {'W-mAP':>8} {'Tiny':>8} {'Small':>8} {'Med':>8} {'Large':>8} {'Preds':>7} {'Time':>6}"
    print(header)
    print("-" * len(header))

    sorted_by_wmap = sorted(results, key=lambda r: r['weighted_map50'], reverse=True)
    for rank, r in enumerate(sorted_by_wmap, 1):
        sm = r['size_metrics']
        marker = " *" if rank == 1 else ""
        print(f"{'#'+str(rank):<5} {r['slice_size']:>5} {r['overlap']:>5.1f} "
              f"{r['overall']['mAP50']:>8.4f} {r['weighted_map50']:>8.4f} "
              f"{sm['tiny']['ap']:>8.4f} {sm['small']['ap']:>8.4f} "
              f"{sm['medium']['ap']:>8.4f} {sm['large']['ap']:>8.4f} "
              f"{r['total_predictions']:>7} {r['inference_time_s']:>5.0f}s{marker}")

    # =========================================================================
    # Heatmaps (text-based)
    # =========================================================================
    slice_sizes = summary['search_grid']['slice_sizes']
    overlap_ratios = summary['search_grid']['overlap_ratios']

    # Build lookup
    lookup = {}
    for r in results:
        lookup[(r['slice_size'], r['overlap'])] = r

    for metric_name, metric_fn, metric_label in [
        ('weighted_map50', lambda r: r['weighted_map50'], 'Weighted mAP50'),
        ('tiny_ap', lambda r: r['size_metrics']['tiny']['ap'], 'Tiny AP50'),
    ]:
        print(f"\n--- Heatmap: {metric_label} ---")
        header_label = "slice\\ovlp"
        print(f"{header_label:>12}", end="")
        for ov in overlap_ratios:
            print(f"{ov:>8.1f}", end="")
        print()
        print("-" * (12 + 8 * len(overlap_ratios)))

        for ss in slice_sizes:
            print(f"{ss:>12}", end="")
            for ov in overlap_ratios:
                r = lookup.get((ss, ov))
                if r:
                    val = metric_fn(r)
                    print(f"{val:>8.4f}", end="")
                else:
                    print(f"{'N/A':>8}", end="")
            print()

    # =========================================================================
    # Recommendation
    # =========================================================================
    print(f"\n{'=' * 90}")
    print("RECOMMENDATION FOR STAGE 2")
    print(f"{'=' * 90}")

    # Pick the best by weighted mAP as the primary recommendation
    best = best_configs['weighted_map50']
    best_tiny = best_configs['tiny_ap']

    print(f"\n  Primary pick (best weighted mAP50):")
    print(f"    slice_size = {best['slice_size']}")
    print(f"    overlap    = {best['overlap']}")

    if best['combo_name'] != best_tiny['combo_name']:
        print(f"\n  Alternative (best tiny AP50, if tiny objects are priority):")
        print(f"    slice_size = {best_tiny['slice_size']}")
        print(f"    overlap    = {best_tiny['overlap']}")

    print(f"\n  Stage 2 will search over these with the chosen slice_size + overlap:")
    print(f"    postprocess_type:            [GREEDYNMM, NMS]")
    print(f"    postprocess_match_metric:    [IOS, IOU]")
    print(f"    postprocess_match_threshold: [0.3, 0.5, 0.7]")
    print(f"    = 12 combinations")

    # Save analysis
    analysis = {
        'results_dir': str(results_dir),
        'baseline_weighted_map50': baseline['weighted_map50'],
        'best_by_weighted_map50': {
            'slice_size': best['slice_size'],
            'overlap': best['overlap'],
            'weighted_map50': best['weighted_map50'],
            'improvement': best['improvements_vs_standard']['weighted_map50_diff'],
        },
        'best_by_tiny_ap': {
            'slice_size': best_tiny['slice_size'],
            'overlap': best_tiny['overlap'],
            'tiny_ap50': best_tiny['size_metrics']['tiny']['ap'],
            'improvement': best_tiny['improvements_vs_standard']['tiny_ap_diff'],
        },
        'full_ranking': [
            {
                'rank': rank,
                'slice_size': r['slice_size'],
                'overlap': r['overlap'],
                'weighted_map50': r['weighted_map50'],
                'mAP50': r['overall']['mAP50'],
                'tiny_ap50': r['size_metrics']['tiny']['ap'],
                'small_ap50': r['size_metrics']['small']['ap'],
            }
            for rank, r in enumerate(sorted_by_wmap, 1)
        ],
    }

    analysis_file = results_dir / "analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Analysis saved to: {analysis_file}")


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Analyze SAHI grid search results")
    parser.add_argument("--dir", type=str, default=None,
                        help="Path to stage1 results directory. If omitted, uses most recent.")

    args = parser.parse_args()

    if args.dir:
        results_dir = Path(args.dir)
        if not results_dir.is_absolute():
            results_dir = PROJECT_ROOT / results_dir
    else:
        results_dir = find_latest_stage1_dir()
        if results_dir is None:
            print("ERROR: No stage1 results found in outputs/sahi_grid_search/")
            print("Run sahi_grid_search_stage1.py first.")
            return

    analyze(results_dir)


if __name__ == "__main__":
    main()
