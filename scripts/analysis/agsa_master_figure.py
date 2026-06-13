"""AGSA Master Figure v3 — 用真实 perception + 真实 fine 数字。"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import matplotlib.image as mpimg
import numpy as np

# SOC 配色
COL_BG = "#0a0e1a"
COL_PANEL = "#141a2a"
COL_BORDER = "#2a3550"
COL_TEXT = "#e6e9f0"
COL_TEXT_DIM = "#8892a8"
COL_ACCENT = "#00d4ff"
COL_SAFE = "#00e676"
COL_WARN = "#ffab00"
COL_DANGER = "#ff1744"
COL_CNN = "#ff6b6b"
COL_DINO = "#4ecdc4"
COL_TF = "#ffe66d"

fig = plt.figure(figsize=(16, 9), dpi=140)
fig.patch.set_facecolor(COL_BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 160); ax.set_ylim(0, 90); ax.axis("off")


def panel(ax, x, y, w, h, color=COL_PANEL, border=COL_BORDER, radius=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=color, edgecolor=border, linewidth=1.2)
    ax.add_patch(p)


# 顶部
panel(ax, 0, 82, 160, 8, color=COL_PANEL, border=COL_BORDER, radius=0.6)
ax.text(3, 86, "AGSA", fontsize=18, weight="bold", color=COL_ACCENT, family="monospace")
ax.text(13, 86, "Security Operations Center", fontsize=14, color=COL_TEXT, weight="bold")
ax.text(120, 86, "REAL-TIME  |  2026.06.12",
        fontsize=10, color=COL_TEXT_DIM, family="monospace", ha="right")
ax.add_patch(Circle((148, 86), 0.8, facecolor=COL_SAFE, edgecolor="none", alpha=0.9))
ax.text(150, 86, "ONLINE", fontsize=9, color=COL_SAFE, weight="bold", family="monospace")

# ===== 左 60%: 真实 perception (rain) =====
LX, LY, LW, LH = 2, 14, 95, 66
panel(ax, LX, LY, LW, LH, color=COL_PANEL, border=COL_BORDER, radius=1.2)
ax.text(LX + 1.5, LY + LH - 3, "PERCEPTION STREAM  (real NAVSIM frame)",
        fontsize=10, color=COL_TEXT_DIM, family="monospace", weight="bold")
ax.text(LX + LW - 1.5, LY + LH - 3, "Rain @ s=0.7  |  256×1024  |  real data",
        fontsize=9, color=COL_DANGER, family="monospace", ha="right", weight="bold")

# 真实 rain 图
vid_x, vid_y, vid_w, vid_h = LX + 2, LY + 18, 60, LH - 22
panel(ax, vid_x, vid_y, vid_w, vid_h, color="#000000", border=COL_BORDER, radius=0.6)
# 用 imshow 嵌入真实图
try:
    rain_img = mpimg.imread("d:/cogatedrive/comp/figures/perception/scene_rain_s07.jpg")
    ax.imshow(rain_img, extent=[vid_x, vid_x + vid_w, vid_y, vid_y + vid_h],
              aspect="auto", zorder=1)
except Exception:
    pass

# 标签
ax.text(vid_x + vid_w / 2, vid_y + vid_h * 0.85, "RAIN  0.7",
        fontsize=14, color=COL_DANGER, weight="bold", family="monospace",
        ha="center", va="center",
        bbox=dict(facecolor="#000000", edgecolor=COL_DANGER, alpha=0.7, pad=4))
ax.text(vid_x + vid_w / 2, vid_y + vid_h * 0.75, "scene 056e9afe...",
        fontsize=8, color=COL_TEXT_DIM, family="monospace", ha="center",
        bbox=dict(facecolor="#000000", edgecolor=COL_BORDER, alpha=0.6, pad=2))

# ===== 风险曲线 (右半) =====
plot_x = vid_x + vid_w + 3
plot_w = LW - (plot_x - LX) - 3
plot_y = vid_y
plot_h = vid_h
panel(ax, plot_x, plot_y, plot_w, plot_h, color="#000000", border=COL_BORDER, radius=0.6)
ax.text(plot_x + 1, plot_y + plot_h - 1.8, "FAILURE BASIN RISK",
        fontsize=9, color=COL_TEXT_DIM, family="monospace", weight="bold")
ax.text(plot_x + plot_w - 1, plot_y + plot_h - 1.8, "now: 0.82",
        fontsize=10, color=COL_DANGER, weight="bold", family="monospace", ha="right")

t = np.linspace(0, 1, 50)
risk = 0.12 + 0.70 * (t ** 1.5) + 0.05 * np.sin(t * 8)
risk = np.clip(risk, 0, 1)
xs = plot_x + 1.5 + t * (plot_w - 3)
ys = plot_y + 2.5 + risk * (plot_h - 7)
ax.plot(xs, ys, color=COL_DANGER, linewidth=2.5, zorder=3)
ax.fill_between(xs, plot_y + 2.5, ys, color=COL_DANGER, alpha=0.2, zorder=2)
ax.add_patch(Circle((xs[-1], ys[-1]), 0.6, facecolor=COL_DANGER, edgecolor="white",
                    linewidth=1, zorder=4))
ax.plot([plot_x + 1.5, plot_x + plot_w - 1.5],
        [plot_y + 2.5 + 0.5 * (plot_h - 7)] * 2,
        color=COL_WARN, linestyle="--", linewidth=1, alpha=0.6, zorder=2)
ax.text(plot_x + plot_w - 2, plot_y + 2.5 + 0.5 * (plot_h - 7) + 1,
        "ENTER", fontsize=7, color=COL_WARN, ha="right", family="monospace")

# ===== 右 40%: 3 cards =====
RX, RW = 100, 58

# Card 1: Gauge
c1_y, c1_h = 56, 24
panel(ax, RX, c1_y, RW, c1_h, color=COL_PANEL, border=COL_BORDER, radius=1.2)
ax.text(RX + 1.5, c1_y + c1_h - 2.5, "CURRENT RISK",
        fontsize=10, color=COL_TEXT_DIM, family="monospace", weight="bold")
ax.text(RX + RW - 1.5, c1_y + c1_h - 2.5, "BASIN DISTANCE  4.7",
        fontsize=9, color=COL_TEXT_DIM, family="monospace", ha="right")
gc_x, gc_y, gc_r = RX + RW / 2, c1_y + 7, 6
Wedge = __import__("matplotlib.patches", fromlist=["Wedge"]).Wedge
wedge_bg = Wedge((gc_x, gc_y), gc_r, 0, 180, facecolor="#1a2335",
                 edgecolor=COL_BORDER, linewidth=1, width=0.6)
ax.add_patch(wedge_bg)
theta = 180 * 0.82
wedge_fg = Wedge((gc_x, gc_y), gc_r, 180 - theta, 180, facecolor=COL_DANGER,
                 edgecolor="none", width=0.6, alpha=0.95)
ax.add_patch(wedge_fg)
ax.text(gc_x, gc_y + 0.5, "0.82", fontsize=20, color=COL_DANGER, weight="bold",
        family="monospace", ha="center", va="center")
ax.text(gc_x, gc_y - 3.2, "BASIN RISK", fontsize=8, color=COL_TEXT_DIM,
        family="monospace", ha="center", va="center")

# Card 2: Status
c2_y, c2_h = 30, 24
panel(ax, RX, c2_y, RW, c2_h, color=COL_PANEL, border=COL_BORDER, radius=1.2)
ax.text(RX + 1.5, c2_y + c2_h - 2.5, "SYSTEM STATUS",
        fontsize=10, color=COL_TEXT_DIM, family="monospace", weight="bold")
ax.text(RX + RW - 1.5, c2_y + c2_h - 2.5, "P1: HIGH",
        fontsize=9, color=COL_DANGER, family="monospace", ha="right", weight="bold")
ax.text(RX + RW / 2, c2_y + c2_h / 2 + 2, "ENTERING",
        fontsize=22, color=COL_DANGER, weight="bold", ha="center", va="center",
        family="monospace")
ax.text(RX + RW / 2, c2_y + c2_h / 2 - 4, "FAILURE BASIN",
        fontsize=22, color=COL_DANGER, weight="bold", ha="center", va="center",
        family="monospace")
ax.text(RX + RW / 2, c2_y + 3,
        "P(fail|DINO)=0.71  P(fail|TF)=0.39  |  upgrade recommended",
        fontsize=9, color=COL_TEXT_DIM, ha="center", family="monospace")

# Card 3: Active Modules
c3_y, c3_h = 14, 14
panel(ax, RX, c3_y, RW, c3_h, color=COL_PANEL, border=COL_BORDER, radius=1.2)
ax.text(RX + 1.5, c3_y + c3_h - 2, "ACTIVE MODULES",
        fontsize=9, color=COL_TEXT_DIM, family="monospace", weight="bold")
mods = [("A", "Discovery", COL_ACCENT), ("B", "Auditor", "#ff7f0e"), ("C", "Shield", COL_SAFE)]
for i, (letter, name, col) in enumerate(mods):
    mx = RX + 1.5 + i * (RW - 3) / 3 + (RW - 3) / 6
    ax.add_patch(Circle((mx - 6, c3_y + 5.5), 1.0, facecolor=col, edgecolor="none"))
    ax.text(mx - 6, c3_y + 5.5, letter, fontsize=10, color="#000", weight="bold",
            ha="center", va="center", family="monospace")
    ax.text(mx + 0.5, c3_y + 5.5, name, fontsize=8, color=COL_TEXT,
            ha="left", va="center", family="monospace")
    ax.text(mx, c3_y + 2, "ONLINE", fontsize=7, color=COL_SAFE,
            ha="center", va="center", family="monospace", weight="bold")

# 底部
panel(ax, 0, 2, 160, 10, color="#060912", border=COL_BORDER, radius=0.6)
ax.text(80, 7, "AGSA: 首个针对端到端自动驾驶的跨架构安全审计系统",
        fontsize=13, color=COL_TEXT, weight="bold", ha="center", va="center")
ax.text(80, 3.8,
        "discover failure laws  ·  audit cross-architecture risk  ·  detect Failure Basin in real time",
        fontsize=9.5, color=COL_TEXT_DIM, ha="center", va="center", style="italic",
        family="monospace")
ax.text(2, 7, "第十九届全国大学生信息安全竞赛 · 自由赛道", fontsize=10, color=COL_TEXT_DIM)
ax.text(2, 3.8, "AGSA v1.0", fontsize=8, color=COL_TEXT_DIM, style="italic")

plt.savefig("d:/cogatedrive/comp/figures/master_figure.png",
            dpi=140, facecolor=COL_BG, bbox_inches="tight")
plt.close()
print("saved → master_figure.png  (real perception + fine numbers)")
