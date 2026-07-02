"""Generate the two IEEE-style diagrams for the report:
   1. System architecture (data -> models -> selection -> calibration -> outputs)
   2. Methodology / pipeline flowchart (3 phases)
Saved as PNGs into analysis_output/visualizations/architecture/.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path("analysis_output/visualizations/architecture")
OUT.mkdir(parents=True, exist_ok=True)

C = {  # palette
    "blue":  ("#DCE9F7", "#2E5C92"), "green": ("#DCEFDD", "#2E7D32"),
    "amber": ("#FCEFD6", "#B8860B"), "red":   ("#FAD9D6", "#C0392B"),
    "purple":("#EAE0F5", "#6A3D9A"), "grey":  ("#ECEFF1", "#455A64"),
}

def box(ax, x, y, w, h, text, color="blue", fs=8, bold=False):
    fill, edge = C[color]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                linewidth=1.4, edgecolor=edge, facecolor=fill))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            weight="bold" if bold else "normal", color="#12233b", wrap=True)

def arrow(ax, x1, y1, x2, y2, color="#455A64"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, linewidth=1.5, color=color))

# ── FIGURE 1: SYSTEM ARCHITECTURE (landscape, full-width) ────────────────────
fig, ax = plt.subplots(figsize=(13.6, 6.0))
ax.set_xlim(0, 13.6); ax.set_ylim(0, 9); ax.axis("off")
ax.text(6.8, 8.55, "Responsible-ML System Architecture",
        ha="center", fontsize=12, weight="bold", color="#12233b")

# main horizontal flow (top row, y ~ 6.1)
mh = 1.35
box(ax, 0.15, 5.9, 2.15, mh, "Patient Inputs\n9 admission-time\nfeatures", "blue", 7.2, True)
box(ax, 2.7, 5.9, 2.15, mh, "Preprocessing\nimpute + scale\n(train-fit only)", "grey", 7.2)
box(ax, 5.25, 5.9, 2.55, mh, "Model Benchmarking\nLR · RF · GB\nXGBoost · Stacking", "amber", 7.2, True)
box(ax, 8.2, 5.9, 2.2, mh, "DeLong Selection\np > 0.05\nLR non-inferior", "green", 7.2)
box(ax, 10.85, 5.9, 2.55, mh, "Isotonic Calibration\nECE 0.195 -> 0.002\nBrier 0.135 -> 0.078", "green", 7.2)
for x1, x2 in [(2.30,2.70),(4.85,5.25),(7.80,8.20),(10.40,10.85)]:
    arrow(ax, x1, 6.57, x2, 6.57)

# deployed model (centred, second row)
box(ax, 4.6, 3.9, 4.4, 1.0, "Deployed Model:  Calibrated Logistic Regression", "purple", 8.5, True)
arrow(ax, 12.1, 5.9, 6.8, 4.9)   # calibration -> deployed

# four output boxes (third row) + prototype
outs = [
    (0.3,  "Calibrated\nRisk Score (%)"),
    (3.55, "Threshold Flag\n0.26 / 0.10"),
    (6.8,  "Explainability\nSHAP · OR · Nomogram"),
    (10.05,"Fairness Audit\n8 subgroups + mitigation"),
]
for ox, t in outs:
    box(ax, ox, 1.85, 3.05, 1.0, t, "blue", 7.0)
    arrow(ax, 6.8, 3.9, ox + 1.52, 2.9)
box(ax, 4.85, 0.25, 3.9, 0.95, "Streamlit Clinical Prototype\n(+ Model Card & Datasheet)", "grey", 7.5, True)
for ox, _ in outs:
    arrow(ax, ox + 1.52, 1.85, 6.8, 1.2)
fig.tight_layout()
fig.savefig(OUT / "fig_system_architecture.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ── FIGURE 2: METHODOLOGY PIPELINE (3 phases) ────────────────────────────────
fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.set_xlim(0, 15); ax.set_ylim(0, 10); ax.axis("off")
ax.text(7.5, 9.6, "13-Phase Methodology Pipeline", ha="center", fontsize=11,
        weight="bold", color="#12233b")

# phase panels
ax.add_patch(FancyBboxPatch((0.2, 0.4), 4.5, 8.6, boxstyle="round,pad=0.05,rounding_size=0.1",
                            linewidth=1.6, edgecolor="#2E5C92", facecolor="#F5F9FE"))
ax.add_patch(FancyBboxPatch((5.25, 0.4), 4.5, 8.6, boxstyle="round,pad=0.05,rounding_size=0.1",
                            linewidth=1.6, edgecolor="#B8860B", facecolor="#FFFBF3"))
ax.add_patch(FancyBboxPatch((10.3, 0.4), 4.5, 8.6, boxstyle="round,pad=0.05,rounding_size=0.1",
                            linewidth=1.6, edgecolor="#2E7D32", facecolor="#F5FBF5"))
ax.text(2.45, 8.55, "1  Data Preparation", ha="center", fontsize=8.5, weight="bold", color="#2E5C92")
ax.text(7.5, 8.55, "2  Training & Selection", ha="center", fontsize=8.5, weight="bold", color="#B8860B")
ax.text(12.55, 8.55, "3  Evaluation", ha="center", fontsize=8.5, weight="bold", color="#2E7D32")

def step(ax, cx, cy, t, color):
    box(ax, cx-2.0, cy-0.45, 4.0, 0.9, t, color, 6.6)

p1 = ["Load 566,602 records","Filter confirmed: 220,218",
      "Sentinel decode + median impute","Standardise (train-fit)",
      "Stratified 80/20 split","Cost-sensitive weighting (1:7)"]
for i, t in enumerate(p1):
    step(ax, 2.45, 7.7-i*1.18, t, "blue")
    if i: arrow(ax, 2.45, 7.7-(i-1)*1.18-0.45, 2.45, 7.7-i*1.18+0.45)

p2 = ["5 models: GridSearchCV (5-fold)","DeLong AUC comparison",
      "Isotonic calibration","Threshold optimisation (F1 / Youden)","Bootstrap 95% CIs"]
for i, t in enumerate(p2):
    step(ax, 7.5, 7.5-i*1.4, t, "amber")
    if i: arrow(ax, 7.5, 7.5-(i-1)*1.4-0.45, 7.5, 7.5-i*1.4+0.45)

p3 = ["AUC + DeLong tests","Calibration (Brier / ECE)",
      "8-subgroup fairness audit","SHAP · OR · nomogram",
      "Decision curve + temporal val.","Model Card + Datasheet"]
for i, t in enumerate(p3):
    step(ax, 12.55, 7.7-i*1.18, t, "green")
    if i: arrow(ax, 12.55, 7.7-(i-1)*1.18-0.45, 12.55, 7.7-i*1.18+0.45)

arrow(ax, 4.7, 4.7, 5.25, 4.7); arrow(ax, 9.75, 4.7, 10.3, 4.7)
fig.tight_layout()
fig.savefig(OUT / "fig_pipeline.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", OUT / "fig_system_architecture.png")
print("Saved:", OUT / "fig_pipeline.png")
