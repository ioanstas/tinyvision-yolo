"""
Generate P2 vs Standard YOLOv8 architecture comparison diagram.
Saves to outputs/figures/p2_architecture_comparison.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_PATH = "outputs/figures/p2_architecture_comparison.png"


def draw_block(ax, x, y, w, h, text, color, fontsize=8, bold=False):
    """Draw a rounded rectangle with centered text."""
    box = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor="black", linewidth=1.2
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color="white" if color in ["#2c3e50", "#8e44ad", "#c0392b", "#2980b9"] else "black")


def draw_arrow(ax, x1, y1, x2, y2, color="black", style="-|>"):
    """Draw an arrow between two points."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=1.5))


def draw_architecture(ax, title, has_p2=False):
    """Draw one YOLOv8 architecture (standard or P2)."""
    bb_color = "#2c3e50"     # backbone - dark blue
    neck_color = "#2980b9"   # neck - blue
    p2_color = "#c0392b"     # P2 blocks - red
    detect_color = "#8e44ad" # detect - purple
    cat_color = "#f39c12"    # concat - orange

    ax.set_xlim(-1, 11)
    ax.set_ylim(-0.5, 10.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    bw, bh = 1.8, 0.6  # block width, height

    # ===================== BACKBONE (left column, bottom to top) ============
    bb_x = 1.5
    backbone_layers = [
        ("Conv\n3x3 s=2", 0.5, "P1/2\n320x320"),
        ("Conv\n3x3 s=2", 1.7, "P2/4\n160x160"),
        ("C2f + Conv\n3x3 s=2", 3.0, "P3/8\n80x80"),
        ("C2f + Conv\n3x3 s=2", 4.3, "P4/16\n40x40"),
        ("C2f + SPPF", 5.6, "P5/32\n20x20"),
    ]

    for text, y_pos, label in backbone_layers:
        draw_block(ax, bb_x, y_pos, bw, bh, text, bb_color, fontsize=6.5)
        ax.text(bb_x - 1.2, y_pos, label, ha="center", va="center",
                fontsize=5.5, color="#555555")

    # Arrows between backbone layers
    for i in range(len(backbone_layers) - 1):
        y1 = backbone_layers[i][1] + bh / 2
        y2 = backbone_layers[i + 1][1] - bh / 2
        draw_arrow(ax, bb_x, y1, bb_x, y2)

    # ===================== NECK - Top-down (FPN) ===========================
    neck_x = 4.5
    # P5 -> P4
    draw_block(ax, neck_x, 5.6, 1.4, 0.4, "Upsample", "#3498db", fontsize=6)
    draw_arrow(ax, bb_x + bw / 2, 5.6, neck_x - 0.7, 5.6)

    draw_block(ax, neck_x, 4.8, 1.4, 0.4, "Cat + C2f", cat_color, fontsize=6)
    draw_arrow(ax, neck_x, 5.6 - 0.2, neck_x, 4.8 + 0.2)
    # lateral from backbone P4
    draw_arrow(ax, bb_x + bw / 2, 4.3, neck_x - 0.7, 4.8, color="#888888")

    # P4 -> P3
    draw_block(ax, neck_x, 4.0, 1.4, 0.4, "Upsample", "#3498db", fontsize=6)
    draw_arrow(ax, neck_x, 4.8 - 0.2, neck_x, 4.0 + 0.2)

    draw_block(ax, neck_x, 3.2, 1.4, 0.4, "Cat + C2f", cat_color, fontsize=6)
    draw_arrow(ax, neck_x, 4.0 - 0.2, neck_x, 3.2 + 0.2)
    # lateral from backbone P3
    draw_arrow(ax, bb_x + bw / 2, 3.0, neck_x - 0.7, 3.2, color="#888888")

    if has_p2:
        # P3 -> P2 (NEW)
        draw_block(ax, neck_x, 2.4, 1.4, 0.4, "Upsample", p2_color, fontsize=6)
        draw_arrow(ax, neck_x, 3.2 - 0.2, neck_x, 2.4 + 0.2)

        draw_block(ax, neck_x, 1.6, 1.4, 0.4, "Cat + C2f", p2_color, fontsize=6)
        draw_arrow(ax, neck_x, 2.4 - 0.2, neck_x, 1.6 + 0.2)
        # lateral from backbone P2
        draw_arrow(ax, bb_x + bw / 2, 1.7, neck_x - 0.7, 1.6, color="#888888")

        ax.text(neck_x + 1.0, 2.0, "NEW\nP2 path", ha="center", va="center",
                fontsize=7, color=p2_color, fontweight="bold")

    # ===================== NECK - Bottom-up (PAN) ==========================
    pan_x = 7.0

    if has_p2:
        # P2 -> P3 (NEW)
        draw_block(ax, pan_x, 1.6, 1.4, 0.4, "Conv s=2", p2_color, fontsize=6)
        draw_arrow(ax, neck_x + 0.7, 1.6, pan_x - 0.7, 1.6)

        draw_block(ax, pan_x, 2.6, 1.4, 0.4, "Cat + C2f", p2_color, fontsize=6)
        draw_arrow(ax, pan_x, 1.6 + 0.2, pan_x, 2.6 - 0.2)
        # lateral from FPN P3
        draw_arrow(ax, neck_x + 0.7, 3.2, pan_x - 0.7, 2.6, color="#888888")

        # P3 -> P4
        draw_block(ax, pan_x, 3.5, 1.4, 0.4, "Conv s=2", neck_color, fontsize=6)
        draw_arrow(ax, pan_x, 2.6 + 0.2, pan_x, 3.5 - 0.2)

        draw_block(ax, pan_x, 4.4, 1.4, 0.4, "Cat + C2f", cat_color, fontsize=6)
        draw_arrow(ax, pan_x, 3.5 + 0.2, pan_x, 4.4 - 0.2)
        draw_arrow(ax, neck_x + 0.7, 4.8, pan_x - 0.7, 4.4, color="#888888")

        # P4 -> P5
        draw_block(ax, pan_x, 5.3, 1.4, 0.4, "Conv s=2", neck_color, fontsize=6)
        draw_arrow(ax, pan_x, 4.4 + 0.2, pan_x, 5.3 - 0.2)

        draw_block(ax, pan_x, 6.2, 1.4, 0.4, "Cat + C2f", cat_color, fontsize=6)
        draw_arrow(ax, pan_x, 5.3 + 0.2, pan_x, 6.2 - 0.2)
        draw_arrow(ax, neck_x + 0.7, 5.6, pan_x - 0.7, 6.2, color="#888888")

    else:
        # Standard: P3 -> P4
        draw_block(ax, pan_x, 3.2, 1.4, 0.4, "Conv s=2", neck_color, fontsize=6)
        draw_arrow(ax, neck_x + 0.7, 3.2, pan_x - 0.7, 3.2)

        draw_block(ax, pan_x, 4.1, 1.4, 0.4, "Cat + C2f", cat_color, fontsize=6)
        draw_arrow(ax, pan_x, 3.2 + 0.2, pan_x, 4.1 - 0.2)
        draw_arrow(ax, neck_x + 0.7, 4.8, pan_x - 0.7, 4.1, color="#888888")

        # P4 -> P5
        draw_block(ax, pan_x, 5.0, 1.4, 0.4, "Conv s=2", neck_color, fontsize=6)
        draw_arrow(ax, pan_x, 4.1 + 0.2, pan_x, 5.0 - 0.2)

        draw_block(ax, pan_x, 5.9, 1.4, 0.4, "Cat + C2f", cat_color, fontsize=6)
        draw_arrow(ax, pan_x, 5.0 + 0.2, pan_x, 5.9 - 0.2)
        draw_arrow(ax, neck_x + 0.7, 5.6, pan_x - 0.7, 5.9, color="#888888")

    # ===================== DETECT HEAD =====================================
    det_x = 9.5

    if has_p2:
        heads = [
            ("P2\n160x160", 1.6, p2_color),
            ("P3\n80x80", 2.6, detect_color),
            ("P4\n40x40", 4.4, detect_color),
            ("P5\n20x20", 6.2, detect_color),
        ]
        # Arrows to Detect (P2 comes from FPN Cat+C2f, rest from PAN)
        draw_arrow(ax, neck_x + 0.7, 1.6, det_x - 0.6, 1.6, color=p2_color)  # P2 from FPN
        draw_arrow(ax, pan_x + 0.7, 2.6, det_x - 0.6, 2.6)   # P3
        draw_arrow(ax, pan_x + 0.7, 4.4, det_x - 0.6, 4.4)   # P4
        draw_arrow(ax, pan_x + 0.7, 6.2, det_x - 0.6, 6.2)   # P5
    else:
        heads = [
            ("P3\n80x80", 3.2, detect_color),
            ("P4\n40x40", 4.1, detect_color),
            ("P5\n20x20", 5.9, detect_color),
        ]
        draw_arrow(ax, pan_x + 0.7, 3.2, det_x - 0.6, 3.2)
        draw_arrow(ax, pan_x + 0.7, 4.1, det_x - 0.6, 4.1)
        draw_arrow(ax, pan_x + 0.7, 5.9, det_x - 0.6, 5.9)

    for text, y_pos, color in heads:
        draw_block(ax, det_x, y_pos, 1.2, 0.55, text, color, fontsize=6.5, bold=True)

    # Detect label
    mid_y = sum(h[1] for h in heads) / len(heads)
    ax.text(det_x, mid_y + 1.5, "Detect\nHead", ha="center", va="center",
            fontsize=9, fontweight="bold", color=detect_color)

    # ===================== LABELS ==========================================
    ax.text(bb_x, 6.6, "Backbone", ha="center", fontsize=9, fontweight="bold",
            color=bb_color)
    ax.text(neck_x, 6.6, "Neck\n(FPN)", ha="center", fontsize=8,
            fontweight="bold", color="#2980b9")
    ax.text(pan_x, 7.2 if has_p2 else 6.8, "Neck\n(PAN)", ha="center",
            fontsize=8, fontweight="bold", color="#2980b9")


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle("YOLOv8n Architecture: Standard vs P2 Detection Head",
                 fontsize=15, fontweight="bold", y=0.98)

    draw_architecture(ax1, "Standard YOLOv8n (3 heads)", has_p2=False)
    draw_architecture(ax2, "YOLOv8n-P2 (4 heads — ours)", has_p2=True)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#2c3e50", label="Backbone"),
        mpatches.Patch(facecolor="#2980b9", label="Neck (standard)"),
        mpatches.Patch(facecolor="#c0392b", label="P2 addition (new)"),
        mpatches.Patch(facecolor="#f39c12", label="Concat + C2f"),
        mpatches.Patch(facecolor="#8e44ad", label="Detect head"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5,
               fontsize=10, frameon=True, fancybox=True)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    import os
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
