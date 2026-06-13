"""AGSA 30秒 SOC 风格 MP4 视频 — 答辩 Demo 视频。

叙事 (3 段):
  0-10s: SAFE     晴朗场景, 风险 ~0.12
  10-18s: WARNING  小雨, 风险上升 0.12 -> 0.45
  18-26s: ENTERING 中到大雨, 风险 0.45 -> 0.75
  26-30s: IN BASIN 暴雨, 风险 0.75 -> 0.85, 状态变红

布局 (16:9, 1920x1080):
  顶  header  AGSA + status dot
  左  camera feed placeholder + rain 动画
  右  Risk gauge + Status
  下  Risk curve (rising)
  底  footer + 4 key numbers

配音 (你自己录): "AGSA detects planner failure in real time, before collision."

输出: comp/agsa_demo_30s.mp4 (H.264, 30s, 30 fps)
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Wedge
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.gridspec as gridspec
import imageio_ffmpeg


# 颜色 — SOC
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


FPS = 30
TOTAL_SEC = 30
N_FRAMES = FPS * TOTAL_SEC  # 900 frames


def get_stage(t: float) -> tuple[str, str, str, float, float]:
    """t in [0,1]: 返回 (status_text, status_color, scene_desc, risk_target, current_risk_factor)."""
    if t < 0.33:  # 0-10s SAFE
        s = (t - 0) / 0.33
        return "SAFE", COL_SAFE, "CLEAR WEATHER", 0.12, s
    elif t < 0.60:  # 10-18s WARNING
        s = (t - 0.33) / 0.27
        return "APPROACHING", COL_WARN, "LIGHT RAIN", 0.45, s
    elif t < 0.87:  # 18-26s ENTERING
        s = (t - 0.60) / 0.27
        return "ENTERING BASIN", COL_DANGER, "HEAVY RAIN", 0.75, s
    else:  # 26-30s IN BASIN
        s = (t - 0.87) / 0.13
        return "IN FAILURE BASIN", COL_DANGER, "STORM", 0.85, s


def render_frame(fig, ax_panels, t: float, frame_idx: int):
    """Render 1 frame at time t in [0,1]."""
    # 清所有 axes
    for ax in ax_panels:
        ax.clear()

    # 主坐标系
    ax_main = ax_panels["main"]
    ax_main.set_xlim(0, 160); ax_main.set_ylim(0, 90); ax_main.axis("off")
    ax_main.set_facecolor(COL_BG)

    # 顶部 header
    header = FancyBboxPatch((0, 82), 160, 8, boxstyle="round,pad=0,rounding_size=0.6",
                            facecolor=COL_PANEL, edgecolor=COL_BORDER, linewidth=1.2)
    ax_main.add_patch(header)
    ax_main.text(3, 86, "AGSA", fontsize=18, weight="bold", color=COL_ACCENT, family="monospace")
    ax_main.text(13, 86, "/  Security Operations Center", fontsize=14, color=COL_TEXT,
                 weight="bold", family="monospace")

    status_text, status_color, scene_desc, target_risk, s = get_stage(t)
    # 状态 dot
    ax_main.add_patch(Circle((148, 86), 0.8, facecolor=status_color, edgecolor="none", alpha=0.9))
    ax_main.text(150, 86, status_text, fontsize=9, color=status_color, weight="bold",
                 family="monospace")

    # ===== 左 60%: 视频 feed + 雨动画 =====
    LX, LY, LW, LH = 2, 14, 95, 66
    panel = FancyBboxPatch((LX, LY), LW, LH, boxstyle="round,pad=0,rounding_size=1.2",
                           facecolor=COL_PANEL, edgecolor=COL_BORDER, linewidth=1.2)
    ax_main.add_patch(panel)
    ax_main.text(LX + 1.5, LY + LH - 3, "PERCEPTION STREAM",
                 fontsize=10, color=COL_TEXT_DIM, family="monospace", weight="bold")
    ax_main.text(LX + LW - 1.5, LY + LH - 3, f"t={t * TOTAL_SEC:.1f}s  |  {scene_desc}",
                 fontsize=9, color=COL_TEXT_DIM, family="monospace", ha="right")

    # 视频框
    vid_x, vid_y, vid_w, vid_h = LX + 2, LY + 18, 60, LH - 22
    vbox = FancyBboxPatch((vid_x, vid_y), vid_w, vid_h,
                          boxstyle="round,pad=0,rounding_size=0.6",
                          facecolor="#000000", edgecolor=COL_BORDER, linewidth=1)
    ax_main.add_patch(vbox)

    # 天空渐变
    from matplotlib.colors import LinearSegmentedColormap
    sky_cmap = LinearSegmentedColormap.from_list("sky", ["#3a4d6a", "#1a2335"])
    gradient = np.linspace(0, 1, 64).reshape(-1, 1)
    ax_main.imshow(sky_cmap(gradient), extent=[vid_x, vid_x + vid_w, vid_y + vid_h * 0.5, vid_y + vid_h],
                   aspect="auto", zorder=1)

    # 道路
    import matplotlib.patches as mp
    road = mp.Polygon([
        (vid_x + vid_w * 0.15, vid_y + 0.5),
        (vid_x + vid_w * 0.85, vid_y + 0.5),
        (vid_x + vid_w * 0.55, vid_y + vid_h * 0.5),
        (vid_x + vid_w * 0.45, vid_y + vid_h * 0.5),
    ], closed=True, facecolor="#1a1a1a", edgecolor="#3a3a3a", linewidth=1, zorder=2)
    ax_main.add_patch(road)
    for i in range(8):
        y_pos = vid_y + 1.5 + i * 3
        if y_pos < vid_y + vid_h * 0.5:
            ax_main.plot([vid_x + vid_w * 0.5, vid_x + vid_w * 0.5],
                         [y_pos, y_pos + 1.5], color="#aaaa44", linewidth=2, zorder=3)

    # 雨强度: 从 0 (safe) 到 1 (storm)
    rain_intensity = min(1.0, max(0, (t - 0.2) * 2.0))  # 0.5s 后开始下雨
    n_drops = int(rain_intensity * 80)
    rng = np.random.RandomState(int(t * 10))
    for _ in range(n_drops):
        rx = rng.uniform(vid_x, vid_x + vid_w)
        ry = rng.uniform(vid_y, vid_y + vid_h * 0.5)
        drop_len = 2 + rain_intensity * 4
        ax_main.plot([rx, rx - 1.5], [ry, ry + drop_len], color="#aaccee",
                     linewidth=0.7, alpha=0.5, zorder=4)

    # 场景标签
    ax_main.text(vid_x + vid_w / 2, vid_y + vid_h * 0.78, scene_desc,
                 fontsize=14, color=status_color, weight="bold", family="monospace",
                 ha="center", va="center",
                 bbox=dict(facecolor="#000000", edgecolor=status_color, alpha=0.6, pad=4))

    # ===== 风险曲线 (右半) =====
    plot_x = vid_x + vid_w + 3
    plot_w = LW - (plot_x - LX) - 3
    plot_y = vid_y
    plot_h = vid_h
    pbox = FancyBboxPatch((plot_x, plot_y), plot_w, plot_h,
                          boxstyle="round,pad=0,rounding_size=0.6",
                          facecolor="#000000", edgecolor=COL_BORDER, linewidth=1)
    ax_main.add_patch(pbox)
    ax_main.text(plot_x + 1, plot_y + plot_h - 1.8, "FAILURE BASIN RISK",
                 fontsize=9, color=COL_TEXT_DIM, family="monospace", weight="bold")
    current_risk = min(0.95, 0.12 + (target_risk - 0.12) * s + 0.02 * np.sin(t * 30))
    ax_main.text(plot_x + plot_w - 1, plot_y + plot_h - 1.8, f"now: {current_risk:.2f}",
                 fontsize=10, color=status_color, weight="bold", family="monospace", ha="right")

    # 曲线 (到目前为止的 risk 曲线)
    t_hist = np.linspace(0, max(t, 0.01), 50)
    # 风险随时间 t 变化 (简化: 线性插值 + 微扰)
    risk_hist = []
    for th in t_hist:
        if th < 0.33:
            r = 0.12 + 0.02 * np.sin(th * 30)
        elif th < 0.60:
            r = 0.12 + (0.45 - 0.12) * ((th - 0.33) / 0.27) + 0.02 * np.sin(th * 30)
        elif th < 0.87:
            r = 0.45 + (0.75 - 0.45) * ((th - 0.60) / 0.27) + 0.02 * np.sin(th * 30)
        else:
            r = 0.75 + (0.85 - 0.75) * ((th - 0.87) / 0.13) + 0.02 * np.sin(th * 30)
        risk_hist.append(min(0.95, r))
    risk_hist = np.array(risk_hist)
    xs_hist = plot_x + 1.5 + t_hist * (plot_w - 3)
    ys_hist = plot_y + 2.5 + risk_hist * (plot_h - 7)
    ax_main.plot(xs_hist, ys_hist, color=status_color, linewidth=2.5, zorder=3)
    ax_main.fill_between(xs_hist, plot_y + 2.5, ys_hist, color=status_color, alpha=0.2, zorder=2)
    # 当前点
    if len(xs_hist) > 0:
        ax_main.add_patch(Circle((xs_hist[-1], ys_hist[-1]), 0.7,
                                 facecolor=status_color, edgecolor="white",
                                 linewidth=1, zorder=4))

    # 阈值线
    ax_main.plot([plot_x + 1.5, plot_x + plot_w - 1.5],
                 [plot_y + 2.5 + 0.5 * (plot_h - 7)] * 2,
                 color=COL_WARN, linestyle="--", linewidth=1, alpha=0.6, zorder=2)

    # ===== 右 40%: 3 dashboard cards =====
    RX, RW = 100, 58

    # Card 1: Gauge
    c1_y, c1_h = 56, 24
    c1 = FancyBboxPatch((RX, c1_y), RW, c1_h, boxstyle="round,pad=0,rounding_size=1.2",
                        facecolor=COL_PANEL, edgecolor=COL_BORDER, linewidth=1.2)
    ax_main.add_patch(c1)
    ax_main.text(RX + 1.5, c1_y + c1_h - 2.5, "CURRENT RISK",
                 fontsize=10, color=COL_TEXT_DIM, family="monospace", weight="bold")
    ax_main.text(RX + RW - 1.5, c1_y + c1_h - 2.5, "BASIN DISTANCE  4.7",
                 fontsize=9, color=COL_TEXT_DIM, family="monospace", ha="right")
    # gauge
    gc_x, gc_y, gc_r = RX + RW / 2, c1_y + 7, 6
    wedge_bg = Wedge((gc_x, gc_y), gc_r, 0, 180, facecolor="#1a2335",
                     edgecolor=COL_BORDER, linewidth=1, width=0.6)
    ax_main.add_patch(wedge_bg)
    theta = 180 * current_risk
    wedge_fg = Wedge((gc_x, gc_y), gc_r, 180 - theta, 180, facecolor=status_color,
                     edgecolor="none", width=0.6, alpha=0.95)
    ax_main.add_patch(wedge_fg)
    ax_main.text(gc_x, gc_y + 0.5, f"{current_risk:.2f}", fontsize=20, color=status_color,
                 weight="bold", family="monospace", ha="center", va="center")
    ax_main.text(gc_x, gc_y - 3.2, "BASIN RISK", fontsize=8, color=COL_TEXT_DIM,
                 family="monospace", ha="center", va="center")

    # Card 2: Status
    c2_y, c2_h = 30, 24
    c2 = FancyBboxPatch((RX, c2_y), RW, c2_h, boxstyle="round,pad=0,rounding_size=1.2",
                        facecolor=COL_PANEL, edgecolor=COL_BORDER, linewidth=1.2)
    ax_main.add_patch(c2)
    ax_main.text(RX + 1.5, c2_y + c2_h - 2.5, "SYSTEM STATUS",
                 fontsize=10, color=COL_TEXT_DIM, family="monospace", weight="bold")
    ax_main.text(RX + RW - 1.5, c2_y + c2_h - 2.5,
                 f"P1: {status_text}" if status_color == COL_DANGER else "P3: INFO",
                 fontsize=9, color=status_color, family="monospace", ha="right", weight="bold")
    # 大字
    parts = status_text.split()
    if len(parts) >= 2:
        ax_main.text(RX + RW / 2, c2_y + c2_h / 2 + 2, parts[0],
                     fontsize=20, color=status_color, weight="bold", ha="center", va="center",
                     family="monospace")
        ax_main.text(RX + RW / 2, c2_y + c2_h / 2 - 4, " ".join(parts[1:]),
                     fontsize=20, color=status_color, weight="bold", ha="center", va="center",
                     family="monospace")
    else:
        ax_main.text(RX + RW / 2, c2_y + c2_h / 2 - 1, status_text,
                     fontsize=20, color=status_color, weight="bold", ha="center", va="center",
                     family="monospace")

    # Card 3: Modules
    c3_y, c3_h = 14, 14
    c3 = FancyBboxPatch((RX, c3_y), RW, c3_h, boxstyle="round,pad=0,rounding_size=1.2",
                        facecolor=COL_PANEL, edgecolor=COL_BORDER, linewidth=1.2)
    ax_main.add_patch(c3)
    ax_main.text(RX + 1.5, c3_y + c3_h - 2, "ACTIVE MODULES",
                 fontsize=9, color=COL_TEXT_DIM, family="monospace", weight="bold")
    mods = [
        ("A", "Discovery", COL_ACCENT),
        ("B", "Auditor",  "#ff7f0e"),
        ("C", "Shield",   COL_SAFE),
    ]
    for i, (letter, name, col) in enumerate(mods):
        mx = RX + 1.5 + i * (RW - 3) / 3 + (RW - 3) / 6
        ax_main.add_patch(Circle((mx - 6, c3_y + 5.5), 1.0, facecolor=col, edgecolor="none"))
        ax_main.text(mx - 6, c3_y + 5.5, letter, fontsize=10, color="#000", weight="bold",
                     ha="center", va="center", family="monospace")
        ax_main.text(mx + 0.5, c3_y + 5.5, name, fontsize=8, color=COL_TEXT,
                     ha="left", va="center", family="monospace")
        ax_main.text(mx, c3_y + 2, "ONLINE", fontsize=7, color=COL_SAFE,
                     ha="center", va="center", family="monospace", weight="bold")

    # ===== 底部 footer =====
    footer = FancyBboxPatch((0, 2), 160, 10, boxstyle="round,pad=0,rounding_size=0.6",
                            facecolor="#060912", edgecolor=COL_BORDER, linewidth=1.2)
    ax_main.add_patch(footer)
    ax_main.text(80, 7, "AGSA: 首个针对端到端自动驾驶的跨架构安全审计系统",
                 fontsize=13, color=COL_TEXT, weight="bold", ha="center", va="center", family="monospace")
    ax_main.text(80, 3.8,
                 "discover failure laws  ·  audit cross-architecture risk  ·  detect Failure Basin in real time",
                 fontsize=9.5, color=COL_TEXT_DIM, ha="center", va="center", style="italic",
                 family="monospace")

    # 4 关键数字 (右下小)
    nums = ["88,560", "89.2%", "0.876-0.939", "0.4ms"]
    for i, n in enumerate(nums):
        ax_main.text(151 - i * 1.5, 9, n, fontsize=7, color=COL_ACCENT,
                     family="monospace", weight="bold", ha="right")


def main():
    print(f"=== Generating 30s AGSA SOC MP4 ({N_FRAMES} frames @ {FPS} fps) ===")
    out_path = "d:/cogatedrive/comp/agsa_demo_30s.mp4"

    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=COL_BG)
    ax = fig.add_axes([0, 0, 1, 1])

    print("  rendering frames ...")
    # 1) 用 imageio-ffmpeg 写 mp4
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    import subprocess
    # 先渲染所有 frame 为 PNG 临时文件
    tmp_dir = "d:/cogatedrive/comp/_tmp_frames"
    import os
    os.makedirs(tmp_dir, exist_ok=True)

    for fi in range(N_FRAMES):
        t = fi / (N_FRAMES - 1)
        render_frame(fig, {"main": ax}, t, fi)
        out_png = f"{tmp_dir}/f{fi:04d}.png"
        fig.savefig(out_png, dpi=120, facecolor=COL_BG)
        if fi % 30 == 0:
            print(f"    {fi}/{N_FRAMES}", flush=True)

    plt.close(fig)

    print(f"  frames rendered, encoding MP4 via ffmpeg ...")
    # 2) ffmpeg 合成 mp4
    cmd = [
        ffmpeg_exe, "-y",
        "-framerate", str(FPS),
        "-i", f"{tmp_dir}/f%04d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",  # 视觉质量
        "-preset", "veryfast",
        out_path,
    ]
    print(f"  cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] ffmpeg failed: {result.stderr[:500]}")
        return
    print(f"  ✓ MP4 saved → {out_path}")

    # 3) 清理临时
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  cleaned tmp dir {tmp_dir}")


if __name__ == "__main__":
    main()
