"""Publication-style diagram of the implemented full-width GeometryGS architecture.

CPU only. Produces editable SVG, vector PDF and 300-dpi PNG in docs/.
The RGB thumbnail illustrates the scene (GT, not a model output); Gaussian glyphs are schematic.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Ellipse
from matplotlib.path import Path as MplPath
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
C = dict(ink="#273342", line="#536071", muted="#637082",
         blue="#E7EFF8", blue_edge="#7B9DBB", green="#E8F3EE", green_edge="#6E9B87",
         amber="#FFF1D7", amber_edge="#B68D46", purple="#F1EBF7", purple_edge="#967BB2",
         anchor="#DF9B41", free="#8B79AE")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 16, "text.color": C["ink"],
    "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "savefig.facecolor": "white",
})
fig, ax = plt.subplots(figsize=(19, 8.7))
ax.set(xlim=(0, 21.5), ylim=(0, 9.9), aspect="equal")
ax.axis("off")
fig.subplots_adjust(0.01, 0.015, 0.99, 0.99)


def text(x, y, label, size=16, weight="normal", color=None, ha="center", **kw):
    return ax.text(x, y, label, fontsize=size, fontweight=weight, color=color or C["ink"],
                   ha=ha, va="center", linespacing=1.35, zorder=8, **kw)


def box(x, y, w, h, fill="white", edge=None, dashed=False, radius=.09, lw=1.2):
    patch = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0.02,rounding_size={radius}",
                          facecolor=fill, edgecolor=edge or C["line"], linewidth=lw,
                          linestyle=(0, (4, 3)) if dashed else "-", zorder=2)
    ax.add_patch(patch)
    return patch


def arrow(points, color=None, dashed=False, lw=1.35, head=13):
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points)-1))
    patch = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=head,
                            linewidth=lw, color=color or C["line"],
                            linestyle=(0, (4, 3)) if dashed else "-", zorder=4,
                            capstyle="round", joinstyle="round")
    ax.add_patch(patch)


def tokens(x, y, cols=6, rows=3, color=None, size=.16, gap=.09):
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((x+c*(size+gap), y+r*(size+gap)), size, size,
                                   facecolor=color or C["blue_edge"], edgecolor="white", lw=.5, zorder=5))


# Figure title and compact legend.
text(.35, 9.45, "GeometryGS", size=29, weight="bold", ha="left")
text(4.15, 9.45, "SfM-anchored spatial Gaussian generation", size=21, ha="left")
text(.36, 8.98, "TokenGS-scale backbone  ·  4,096 scene latents  ·  4,096 GS tokens  ·  1,024-d features",
     size=14, ha="left", color=C["muted"])
for x, label, fill, edge in [(13.35, "Pretrained", C["blue"], C["blue_edge"]),
                             (15.65, "New", C["green"], C["green_edge"]),
                             (17.2, "SfM", C["amber"], C["amber_edge"]),
                             (18.8, "Optional", C["purple"], C["purple_edge"])]:
    box(x, 8.85, .25, .22, fill, edge, radius=.03, lw=.9)
    text(x+.38, 8.96, label, 13, ha="left")

# Input views and camera icon.
text(1.35, 8.45, "Input views", 18, "bold")
thumbnail_path = ROOT / "EXP/demo/scene_7b6477cb95/spatial_smoke/render_strip.png"
image = np.asarray(Image.open(thumbnail_path).convert("RGB"))[:256, :448]
for dx, dy in ((.25, .25), (.12, .12), (0, 0)):
    ax.imshow(image, extent=(.35+dx, 2.15+dx, 6.97+dy, 8.00+dy), zorder=5+(2 if dx == 0 else 0))
    ax.add_patch(Rectangle((.35+dx, 6.97+dy), 1.8, 1.03, fill=False, edgecolor="white", lw=1.2, zorder=7))
box(.72, 6.25, .57, .36, "#F4F6F8", C["line"], radius=.04)
ax.add_patch(Circle((1.00, 6.43), .10, facecolor="white", edgecolor=C["line"], lw=1.1, zorder=5))
arrow([(1.35, 6.42), (1.72, 6.62)], head=6, lw=.9)
arrow([(1.35, 6.42), (1.72, 6.24)], head=6, lw=.9)
text(1.55, 5.95, "RGB + known cameras\n"+r"$\{I_v,P_v\}$", 12.5)

# Pretrained image -> scene hierarchy.
box(2.82, 5.54, 5.85, 3.20, "#F8FAFD", C["blue_edge"], dashed=True, radius=.12, lw=1.0)
text(5.75, 8.52, "PRETRAINED IMAGE–SCENE PATH", 13, "bold", C["blue_edge"])
box(3.02, 5.85, 2.45, 2.35, C["blue"], C["blue_edge"])
text(4.245, 7.77, "Multi-view\nimage encoder", 16.5, "bold")
text(4.245, 6.97, "RGB + ray embeddings", 12)
for i in range(4):
    box(3.51+i*.36, 6.48+i*.055, .30, .30, "#B4CCE2", C["blue_edge"], radius=.025, lw=.7)
text(4.245, 6.15, "12 layers · multi-scale", 12.5)
box(5.94, 5.85, 2.45, 2.35, C["blue"], C["blue_edge"])
text(7.165, 7.77, "Scene latent\naggregator", 16.5, "bold")
tokens(6.50, 6.62)
text(7.165, 6.20, "12 layers · 4,096 latents", 12)
arrow([(2.45, 7.14), (3.00, 7.14)])
arrow([(5.49, 7.14), (5.92, 7.14)])

# Scene-level SfM residual fusion.
box(9.16, 5.85, 2.40, 2.35, C["green"], C["green_edge"])
text(10.36, 7.77, "SfM scene\nfusion", 18, "bold")
text(10.36, 6.98, "Cross-attn + FFN", 13)
text(10.36, 6.55, "2 residual blocks", 14)
text(10.36, 6.15, "Identity initialization", 13, color=C["muted"])
arrow([(8.41, 7.14), (9.14, 7.14)])
arrow([(11.58, 7.20), (13.58, 7.20)])
text(12.58, 7.57, r"Scene memory $\widetilde{Z}$", 13.5)

# SfM branch.
box(2.84, 3.02, 2.36, 1.60, C["amber"], C["amber_edge"])
text(4.02, 4.22, "Known-pose SfM", 14.5, "bold")
text(4.02, 3.75, "Match · triangulate", 14)
text(4.02, 3.32, "Filter + FPS", 14)
arrow([(1.30, 5.65), (1.30, 3.82), (2.82, 3.82)], C["amber_edge"])
text(1.10, 4.66, "Same\ninput views", 12, color=C["amber_edge"], ha="right")
box(5.77, 3.02, 2.63, 1.60, C["amber"], C["amber_edge"])
text(7.085, 4.23, "SfM memory", 17, "bold")
text(7.085, 3.76, r"$S=\{\mathrm{PE}(a_j),f_j,q_j\}$", 16)
text(7.085, 3.32, "3D · features · quality", 12)
arrow([(5.22, 3.82), (5.75, 3.82)], C["amber_edge"])
arrow([(4.25, 5.82), (4.25, 5.11), (7.09, 5.11), (7.09, 4.64)], C["blue_edge"])
text(6.75, 5.39, "Track-sampled image features", 12.5, color=C["blue_edge"])
arrow([(8.42, 4.20), (8.80, 4.20), (8.80, 6.30), (9.14, 6.30)], C["amber_edge"])

# GS token initialization: explicit fixed/measured and free branches.
box(9.16, 2.95, 3.35, 2.08, "white", C["green_edge"])
text(10.835, 4.68, "GS token initialization", 15.5, "bold")
text(10.835, 4.26, r"Query $\widetilde{Z}$; predict free centers", 12.5)
arrow([(10.36, 5.83), (10.36, 5.05)])
arrow([(8.42, 3.56), (9.14, 3.56)], C["amber_edge"])
for x, color in ((9.95, C["anchor"]), (11.72, C["free"])):
    for dx, dy in ((-.22, -.02), (0, .13), (.22, -.02)):
        ax.add_patch(Circle((x+dx, 3.70+dy), .095, facecolor=color, edgecolor="white", lw=.6, zorder=5))
text(9.96, 3.31, r"$K$ fixed", 12.5, "bold", C["amber_edge"])
text(11.72, 3.31, r"$4096-K$ free", 13, "bold", C["free"])
text(10.83, 2.71, r"$T_i=(h_i,c_i,r_i,t_i)$    ·    $K\leq1024$", 13)

# Spatial decoder block.
box(13.60, 5.85, 3.18, 2.35, C["green"], C["green_edge"])
text(15.19, 7.91, "Spatial GS decoder", 16.5, "bold")
text(15.19, 7.52, "4 layers · 1024-d", 13.5)
for y, label in [(6.94, r"Cross-attention to $\widetilde{Z}$"),
                 (6.48, "Self-attn + relative 3D bias"),
                 (6.02, "FFN + free-center update")]:
    box(13.79, y-.14, 2.80, .34, "#FFFFFF", "#AAC7B8", radius=.035, lw=.75)
    text(15.19, y+.02, label, 12.5)
arrow([(12.53, 4.00), (15.19, 4.00), (15.19, 4.91)])
text(13.77, 4.30, "Spatial GS states", 13)
box(13.79, 4.93, 2.80, .47, "#FFFFFF", C["green_edge"], radius=.035, lw=.85)
text(15.19, 5.165, "3D PE · radius · type", 13, color=C["green_edge"])
arrow([(15.19, 5.42), (15.19, 5.83)])

# Locally bounded Gaussian generation.
box(17.38, 5.85, 3.25, 2.35, C["green"], C["green_edge"])
text(19.005, 7.91, "Local Gaussian head", 16.5, "bold")
text(19.005, 7.51, "Child MLP · bounded scales", 12.5)
text(19.005, 7.11, r"$\mu_{ik}=c_i+r_i\,\mathrm{Ball}(\delta_{ik})$", 15)
rng = np.random.default_rng(7)
for cx, color in [(18.16, C["anchor"]), (19.80, C["free"])]:
    ax.add_patch(Ellipse((cx, 6.49), .95, .58, facecolor="none", edgecolor=color,
                         linestyle=(0, (3, 3)), lw=.9, zorder=5))
    for _ in range(9):
        x, y = rng.normal(0, [.20, .09])
        ax.add_patch(Ellipse((cx+x, 6.49+y), .27, .09, angle=rng.uniform(-30, 30),
                            facecolor=color, edgecolor=color, alpha=.38, lw=.6, zorder=6))
text(19.005, 6.00, "64 Gaussians / token", 13)
arrow([(16.80, 7.20), (17.36, 7.20)])
box(17.38, 3.10, 3.25, 1.38, "#F5F7F9", "#9BA7B5")
text(19.005, 4.12, "Gaussian rasterizer", 17, "bold")
text(19.005, 3.61, r"RGB $\hat I$ + depth $\hat D$", 14)
arrow([(19.005, 5.83), (19.005, 4.50)])
text(19.65, 5.13, r"$\mathcal{G}$", 18)

# Explicit training-only lane; no privileged information feeds the inference graph.
box(.38, .45, 20.25, 1.80, "#FCFCFD", "#ADB4BF", dashed=True, radius=.10, lw=1)
text(.64, 2.02, "TRAINING ONLY", 12.5, "bold", C["muted"], ha="left")
box(.65, .85, 3.65, .89, "#F0F2F5", "#ADB4BF")
text(2.475, 1.48, "RGB + laser depth / normals", 13.5, "bold")
text(2.475, 1.10, "Privileged supervision", 13, color=C["muted"])
box(5.02, .85, 5.35, .89, "#F0F2F5", "#ADB4BF")
text(7.695, 1.48, "Rendering losses", 15, "bold")
text(7.695, 1.10, "RGB + SSIM + depth + visibility", 13)
arrow([(4.32, 1.30), (5.00, 1.30)], dashed=True)
arrow([(19.005, 3.08), (19.005, 2.40), (7.70, 2.40), (7.70, 1.76)], dashed=True)
box(12.15, .85, 8.12, .89, C["purple"], C["purple_edge"], dashed=True)
text(16.21, 1.48, "Optional soft chain  ·  off by default", 15, "bold", C["purple_edge"])
text(16.21, 1.10, r"Same-surface links:  $\rho\!\left(n_{ij}^{\top}(p_i-p_j)\right)$  ·  preserve tangential spacing", 12.5)
arrow([(4.32, 1.13), (4.66, 1.13), (4.66, .65), (11.72, .65), (11.72, 1.30), (12.13, 1.30)],
      C["purple_edge"], dashed=True, lw=1.1)
arrow([(20.65, 6.45), (21.02, 6.45), (21.02, 1.30), (20.29, 1.30)],
      C["purple_edge"], dashed=True, lw=1.15)
text(10.8, .12, "Solid arrows: inference data flow     Dashed arrows: training supervision     Gaussian glyphs: schematic",
     12, color=C["muted"])

destination = ROOT / "docs" / "geometrygs_architecture"
for extension in ("svg", "pdf", "png"):
    fig.savefig(destination.with_suffix("."+extension), dpi=300)
    print(destination.with_suffix("."+extension))
plt.close(fig)
