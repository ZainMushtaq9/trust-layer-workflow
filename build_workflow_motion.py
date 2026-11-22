"""Animated motion version of the trust-layer future-work workflow graph.

Reuses the layout, boxes, arrows, text and colours of build_workflow_graph.py
(the static diagram) and animates it as a 40-second 1080p video:

  * sequential reveal, top to bottom (title -> trust score row -> M1 -> M2
    -> M3 -> extensions -> validation strip)
  * arrows draw progressively after their boxes appear
  * the TRUST SCORE box counts 0.00 -> 0.89 while it appears
  * a glowing pulse travels the main vertical flow path once the diagram
    is complete, then the M2 feedback-loop arc pulses twice
  * final 3+ seconds hold the complete static diagram

Run:
    ~/workspace/venv_nb/bin/python build_workflow_motion.py
        [--seconds 40] [--fps 24] [--output future_work_workflow_motion.mp4]
        [--preview]        # save 3 PNG check-frames instead of rendering video
        [--test-frames N]  # render only the first N frames (quick smoke test)

Requires: matplotlib + ffmpeg (libx264). Rendered with FFMpegWriter.
Output: future_work_workflow_motion.mp4  (1920x1080, 24 fps)
"""
import argparse

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import ConnectionStyle, FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------- colours
# (identical to build_workflow_graph.py)
NAVY = "#1F3864"
TEAL = "#2E75B6"
LIGHT_BLUE = "#D9E2F3"
LIGHT_TEAL = "#DDEBF7"
GOLD = "#ED7D31"
LIGHT_GOLD = "#FCE4D6"
GREEN = "#548235"
LIGHT_GREEN = "#E2EFDA"
GREY = "#595959"

FPS = 24
DURATION = 40.0  # seconds

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)  # 1920x1080
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_position([0, 0, 1, 1])  # diagram fills the whole frame

# Every animated element registers an updater(t_seconds) here.
updaters = []


def smooth(p):
    """Smoothstep easing, clamped to [0, 1]."""
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


# ---------------------------------------------------------------- elements
def make_box(x, y, w, h, text, face, edge, fontsize, t0, dur,
             bold_first=False, slide=1.5, textcolor="#1a1a1a"):
    """Rounded box + centred text; fades in and slides up during [t0, t0+dur].

    Text placement matches build_workflow_graph.py exactly so the final
    frame is identical to the static diagram.
    """
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                           facecolor=face, edgecolor=edge, linewidth=1.6)
    patch.set_alpha(0)
    ax.add_patch(patch)
    texts = []
    if text:
        lines = text.split("\n")
        if bold_first and len(lines) > 1:
            texts.append(ax.text(x + w / 2, y + h / 2 + (len(lines) - 1) * 1.1,
                                 lines[0], ha="center", va="center",
                                 fontsize=fontsize + 0.5, fontweight="bold",
                                 color=textcolor))
            texts.append(ax.text(x + w / 2, y + h / 2 - 2.2, "\n".join(lines[1:]),
                                 ha="center", va="center", fontsize=fontsize - 0.5,
                                 color=textcolor))
        else:
            texts.append(ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                                 fontsize=fontsize, color=textcolor,
                                 fontweight="bold" if bold_first else "normal"))
    for t in texts:
        t.set_alpha(0)
    y0_patch = y
    y0_texts = [t.get_position()[1] for t in texts]

    def update(t):
        p = smooth((t - t0) / dur)
        patch.set_alpha(p)
        off = slide * (1 - p)
        patch.set_y(y0_patch - off)
        for txt, y0 in zip(texts, y0_texts):
            txt.set_alpha(p)
            txt.set_position((txt.get_position()[0], y0 - off))

    updaters.append(update)
    return patch, texts


def make_text(x, y, s, t0, dur, slide=1.0, **kw):
    """Standalone text; fades in and slides during [t0, t0+dur]."""
    t = ax.text(x, y, s, **kw)
    t.set_alpha(0)

    def update(tt):
        p = smooth((tt - t0) / dur)
        t.set_alpha(p)
        t.set_position((x, y - slide * (1 - p)))

    updaters.append(update)
    return t


def make_arrow(x1, y1, x2, y2, color, lw, t0, dur):
    """Straight arrow that draws progressively, arrowhead riding the tip.

    While drawing, a line + marker head are shown; once complete they are
    swapped for the real FancyArrowPatch so the final frame matches the
    static diagram exactly.
    """
    final = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                            linewidth=lw, color=color, shrinkA=0, shrinkB=0)
    final.set_alpha(0)
    ax.add_patch(final)
    line, = ax.plot([], [], color=color, lw=lw, solid_capstyle="round")
    if abs(x2 - x1) > abs(y2 - y1):
        marker = ">" if x2 > x1 else "<"
    else:
        marker = "v" if y2 < y1 else "^"
    head, = ax.plot([], [], marker=marker, color=color, markersize=11,
                    linestyle="None")

    def update(t):
        p = (t - t0) / dur
        if p <= 0:
            line.set_data([], [])
            head.set_data([], [])
            final.set_alpha(0)
        elif p >= 1:
            line.set_data([], [])
            head.set_data([], [])
            final.set_alpha(1)
        else:
            tip_x = x1 + (x2 - x1) * p
            tip_y = y1 + (y2 - y1) * p
            line.set_data([x1, tip_x], [y1, tip_y])
            head.set_data([tip_x], [tip_y])
            final.set_alpha(0)

    updaters.append(update)
    return final


# --- the M2 feedback-loop arc: quadratic Bezier from matplotlib's Arc3 ---
_ARC_PATH = ConnectionStyle.Arc3(rad=-0.05)((84.5, 47.5), (12.5, 47.5))
_ARC_P0, _ARC_P1, _ARC_P2 = _ARC_PATH.vertices
_ARC_N = 300
_ARC_TS = np.linspace(0, 1, _ARC_N)
_ARC_PTS = np.array([(1 - t) ** 2 * _ARC_P0 + 2 * (1 - t) * t * _ARC_P1
                     + t ** 2 * _ARC_P2 for t in _ARC_TS])
_ARC_TAN = np.array([2 * (1 - t) * (_ARC_P1 - _ARC_P0) + 2 * t * (_ARC_P2 - _ARC_P1)
                     for t in _ARC_TS])


def make_arc(t0, dur):
    """The M2 loop-back arc, drawn progressively along its Bezier curve."""
    final = FancyArrowPatch((84.5, 47.5), (12.5, 47.5), arrowstyle="-|>",
                            mutation_scale=18, linewidth=1.8, color=TEAL,
                            shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=-0.05")
    final.set_alpha(0)
    ax.add_patch(final)
    line, = ax.plot([], [], color=TEAL, lw=1.8, solid_capstyle="round")
    head, = ax.plot([], [], marker=(3, 0, 0), color=TEAL, markersize=11,
                    linestyle="None")

    def update(t):
        p = (t - t0) / dur
        if p <= 0:
            line.set_data([], [])
            head.set_data([], [])
            final.set_alpha(0)
        elif p >= 1:
            line.set_data([], [])
            head.set_data([], [])
            final.set_alpha(1)
        else:
            n = max(2, int(p * _ARC_N))
            line.set_data(_ARC_PTS[:n, 0], _ARC_PTS[:n, 1])
            tip = _ARC_PTS[n - 1]
            tan = _ARC_TAN[n - 1]
            # marker (3,0,0) points up; rotate to the tangent direction
            ang = np.degrees(np.arctan2(tan[1], tan[0])) - 90
            head.set_marker((3, 0, ang))
            head.set_data([tip[0]], [tip[1]])
            final.set_alpha(0)

    updaters.append(update)
    return final


def make_bar(t0, dur):
    """The extensions distribution bar, growing outward from the centre."""
    line, = ax.plot([], [], color=NAVY, lw=2.0, solid_capstyle="round")

    def update(t):
        p = (t - t0) / dur
        if p <= 0:
            line.set_data([], [])
        else:
            d = 32 * min(p, 1.0)
            line.set_data([50 - d, 50 + d], [27.2, 27.2])

    updaters.append(update)
    return line


# ---------------------------------------------------------------- pulses
class Pulse:
    """A glowing dot (head + halo + fading trail) travelling along a path.

    points: (N, 2) dense samples along the path; motion is by arc length,
    so the speed is constant.
    """

    def __init__(self, points, color=GOLD, head_size=15, glow_size=44, n_trail=8):
        self.pts = np.asarray(points, dtype=float)
        seg = np.diff(self.pts, axis=0)
        self.cum = np.concatenate([[0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
        self.total = self.cum[-1]
        self.glow, = ax.plot([], [], marker="o", color=color, markersize=glow_size,
                             alpha=0.22, linestyle="None", markeredgecolor="none")
        self.head, = ax.plot([], [], marker="o", color=color, markersize=head_size,
                             linestyle="None", markeredgecolor="white",
                             markeredgewidth=1.4)
        self.trail = []
        for k in range(n_trail):
            a = 0.55 * (1 - (k + 1) / (n_trail + 1))
            s = max(head_size * (1 - 0.75 * (k + 1) / (n_trail + 1)), 2)
            ln, = ax.plot([], [], marker="o", color=color, markersize=s,
                          alpha=a, linestyle="None")
            self.trail.append(ln)

    def _pos(self, s):
        s = min(max(s, 0.0), self.total)
        i = int(np.searchsorted(self.cum, s))
        if i <= 0:
            return self.pts[0]
        if i >= len(self.pts):
            return self.pts[-1]
        s0, s1 = self.cum[i - 1], self.cum[i]
        f = 0.0 if s1 == s0 else (s - s0) / (s1 - s0)
        return self.pts[i - 1] + f * (self.pts[i] - self.pts[i - 1])

    def set_s(self, s):
        """Place the head at arc-length s; None hides the whole pulse."""
        if s is None:
            self.glow.set_data([], [])
            self.head.set_data([], [])
            for ln in self.trail:
                ln.set_data([], [])
            return
        p = self._pos(s)
        self.glow.set_data([p[0]], [p[1]])
        self.head.set_data([p[0]], [p[1]])
        for k, ln in enumerate(self.trail):
            tp = self._pos(s - 1.6 * (k + 1))
            ln.set_data([tp[0]], [tp[1]])


def densify_polyline(waypoints, step=0.2):
    """Sample a polyline densely for arc-length motion."""
    pts = []
    for (x1, y1), (x2, y2) in zip(waypoints[:-1], waypoints[1:]):
        n = max(2, int(np.hypot(x2 - x1, y2 - y1) / step))
        xs = np.linspace(x1, x2, n, endpoint=False)
        ys = np.linspace(y1, y2, n, endpoint=False)
        pts.extend(zip(xs, ys))
    pts.append(waypoints[-1])
    return np.array(pts, dtype=float)


# ---------------------------------------------------------------- timeline
# Times in seconds. Reveal runs top to bottom; pulses run after the full
# diagram is visible; the last 4 s hold the complete static diagram.

# --- title (0.0-1.5) ---
make_text(50, 97.5, "Trust Layer \u2014 Complete Future Work Workflow",
          0.0, 1.5, slide=-1.5, ha="center", va="center",
          fontsize=19, fontweight="bold", color=NAVY)
make_text(50, 94.8,
          "TeaPle \u2192 trust-weighted link prediction \u2192 feedback loop \u2192 Digitization Footprint \u2192 extensions",
          0.0, 1.5, slide=-1.0, ha="center", va="center",
          fontsize=11.5, color=GREY)

# --- trust-score row (1.5-11.3) ---
make_text(2, 90.5, "TRUST SCORE  (per triple, 0\u20131)", 1.5, 0.5,
          ha="left", va="center", fontsize=11, fontweight="bold", color=NAVY)
make_box(2, 80.5, 17, 8, "TeaPle\n6,698 triples", LIGHT_BLUE, NAVY, 11,
         2.0, 0.8, bold_first=True)
make_arrow(19.3, 84.5, 21.2, 84.5, NAVY, 2.0, 2.8, 0.4)
make_box(21.5, 80.5, 11.5, 8, "Provenance\nsource reliability", LIGHT_TEAL, TEAL,
         9.5, 3.2, 0.8, bold_first=True)
make_arrow(33.3, 84.5, 34.2, 84.5, NAVY, 2.0, 4.0, 0.4)
make_box(34.5, 80.5, 11.5, 8, "Consistency\nno contradictions", LIGHT_TEAL, TEAL,
         9.5, 4.4, 0.8, bold_first=True)
make_arrow(46.3, 84.5, 47.2, 84.5, NAVY, 2.0, 5.2, 0.4)
make_box(47.5, 80.5, 11.5, 8, "Corroboration\nsources agreeing", LIGHT_TEAL, TEAL,
         9.5, 5.6, 0.8, bold_first=True)
make_arrow(59.3, 84.5, 60.2, 84.5, NAVY, 2.0, 6.4, 0.4)
make_box(60.5, 80.5, 11.5, 8, "Extraction conf.\nmodel probability", LIGHT_TEAL,
         TEAL, 9.5, 6.8, 0.8, bold_first=True)
make_arrow(72.3, 84.5, 74.7, 84.5, NAVY, 2.0, 7.6, 0.4)

# --- TRUST SCORE box + counting score (8.0-10.8) ---
# Box and title fade/slide in 8.0-8.8; the counter runs 8.8-10.3 in place
# of the subtitle, then cross-fades to the real subtitle by 10.8.
_ts_patch = FancyBboxPatch((75, 80.5), 23, 8,
                           boxstyle="round,pad=0.6,rounding_size=1.2",
                           facecolor=LIGHT_GOLD, edgecolor=GOLD, linewidth=1.6)
_ts_patch.set_alpha(0)
ax.add_patch(_ts_patch)
_ts_title = ax.text(86.5, 85.6, "TRUST SCORE 0\u20131", ha="center", va="center",
                    fontsize=11.5, fontweight="bold", color="#1a1a1a")
_ts_title.set_alpha(0)
_ts_sub = ax.text(86.5, 82.3, "weighted sum of 4 components", ha="center",
                  va="center", fontsize=10.5, color="#1a1a1a")
_ts_sub.set_alpha(0)
_ts_count = ax.text(86.5, 82.3, "0.00", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=NAVY)
_ts_count.set_alpha(0)


def _trust_score_update(t):
    p = smooth((t - 8.0) / 0.8)
    _ts_patch.set_alpha(p)
    off = 1.5 * (1 - p)
    _ts_patch.set_y(80.5 - off)
    _ts_title.set_alpha(p)
    _ts_title.set_position((86.5, 85.6 - off))
    if t < 8.8:
        _ts_count.set_alpha(0)
        _ts_sub.set_alpha(0)
    elif t < 10.3:
        # count 0.00 -> 0.89, easing out so it settles on the final value
        cp = (t - 8.8) / 1.5
        val = 0.89 * (1 - (1 - cp) ** 2)
        _ts_count.set_text(f"{val:.2f}")
        _ts_count.set_alpha(1)
        _ts_sub.set_alpha(0)
    elif t < 10.8:
        # cross-fade counter -> subtitle
        cp = smooth((t - 10.3) / 0.5)
        _ts_count.set_alpha(1 - cp)
        _ts_sub.set_alpha(cp)
    else:
        _ts_count.set_alpha(0)
        _ts_sub.set_alpha(1)


updaters.append(_trust_score_update)

make_text(50, 78.3,
          "weights: start equal (0.25 each)  \u2192  AHP expert ranking  \u2192  logistic regression on expert-audited sample",
          10.8, 0.5, ha="center", va="center", fontsize=9.5, color=GREY,
          style="italic")

# --- M1 (11.3-14.9) ---
make_arrow(86.5, 80.2, 86.5, 72.8, NAVY, 2.0, 11.3, 0.5)
make_text(2, 74.5, "MILESTONE 1 \u2014 FIRST MILESTONE (decided in the meeting)",
          11.8, 0.5, ha="left", va="center", fontsize=11, fontweight="bold",
          color=NAVY)
make_box(2, 63, 96, 9.5, "", LIGHT_BLUE, NAVY, 10.5, 11.8, 0.5)
make_text(50, 70.6, "M1 \u2014 Trust-scored TeaPle  +  Trust-weighted Link Prediction",
          12.3, 0.4, ha="center", va="center", fontsize=12.5,
          fontweight="bold", color=NAVY)
for _i, (_txt, _sx) in enumerate([("Trust-weighted\nloss", 8),
                                  ("Trust-weighted\nGNN messages", 30),
                                  ("Trust-adjusted\nprediction", 52)]):
    make_box(_sx, 64.2, 18, 5.2, _txt, "white", TEAL, 9.5, 12.7 + 0.4 * _i, 0.5)
make_box(74, 64.2, 18, 5.2, "Benchmark vs\nGNN-FTuckER", LIGHT_GOLD, GOLD, 9.5,
         13.9, 0.5)
make_text(50, 61.3,
          "final score = model score \u00d7 trust score  \u00b7  auto-add only if final \u2265 0.9  \u00b7  test set: 665 triples  \u00b7  target: beat MRR 64.6% / H@1 58.5% / H@10 76.3%",
          14.4, 0.5, ha="center", va="center", fontsize=9.5, color=GREY,
          style="italic")

# --- M2 (14.9-20.7) ---
make_arrow(97, 62.8, 97, 56.3, NAVY, 2.0, 14.9, 0.5)
make_text(2, 57.5, "MILESTONE 2 \u2014 FEEDBACK LOOP", 15.4, 0.5,
          ha="left", va="center", fontsize=11, fontweight="bold", color=NAVY)
make_box(2, 46.5, 96, 9.5, "", LIGHT_TEAL, TEAL, 10.5, 15.4, 0.5)
make_text(50, 54.2, "M2 \u2014 Contradiction Feedback Loop  (KG checks NER, NER improves)",
          15.9, 0.4, ha="center", va="center", fontsize=12.5,
          fontweight="bold", color=NAVY)
_loop = [("NER\nextracts", 5), ("Check vs\nknowledge graph", 23),
         ("Flag\ncontradictions", 41), ("Correct\nthe data", 59),
         ("Retrain\nNER model", 77)]
for _i, (_txt, _lx) in enumerate(_loop):
    make_box(_lx, 47.8, 15, 5.2, _txt, "white", TEAL, 9.5, 16.4 + 0.6 * _i, 0.4)
    if _i < 4:
        make_arrow(_lx + 15.3, 50.4, _lx + 17.7, 50.4, TEAL, 1.8,
                   16.8 + 0.6 * _i, 0.2)
make_arc(19.2, 1.0)
make_text(48.5, 43.4, "loop repeats \u2014 errors found by the graph go back to extraction",
          20.2, 0.5, ha="center", va="center", fontsize=9.5, color=TEAL,
          style="italic")

# --- M3 (20.7-23.7) ---
make_arrow(97, 46.2, 97, 39.3, NAVY, 2.0, 20.7, 0.5)
make_text(2, 41.2, "MILESTONE 3 \u2014 TRUST IN THE DIGITIZATION FOOTPRINT",
          21.2, 0.5, ha="left", va="center", fontsize=11, fontweight="bold",
          color=NAVY)
make_box(2, 29.5, 96, 9.5, "", LIGHT_GREEN, GREEN, 10.5, 21.2, 0.5)
make_text(50, 37.2, "M3 \u2014 Trust as the 4th DF Dimension", 21.7, 0.4,
          ha="center", va="center", fontsize=12.5, fontweight="bold", color=NAVY)
make_box(5, 30.8, 26, 5.2,
         "Per-DPU trust indicators\nsensor \u00b7 completeness \u00b7 transmission \u00b7 coverage",
         "white", GREEN, 9.5, 22.1, 0.4)
make_arrow(31.3, 33.4, 36.7, 33.4, GREEN, 1.8, 22.5, 0.2)
make_box(37, 30.8, 26, 5.2, "DF index: Performance \u00b7 Energy \u00b7 Value \u00b7 TRUST",
         "white", GREEN, 9.5, 22.7, 0.4)
make_arrow(63.3, 33.4, 68.7, 33.4, GREEN, 1.8, 23.1, 0.2)
make_box(69, 30.8, 26, 5.2, "Real farm validation\nstubble pipeline / Grafana system",
         LIGHT_GOLD, GOLD, 9.5, 23.3, 0.4)

# --- extensions (23.7-26.4) ---
make_arrow(50, 29.2, 50, 27.2, NAVY, 2.0, 23.7, 0.3)
make_bar(24.0, 0.4)
make_arrow(18, 27.2, 18, 23.8, NAVY, 2.0, 24.4, 0.5)
make_arrow(50, 27.2, 50, 23.8, NAVY, 2.0, 24.4, 0.5)
make_arrow(82, 27.2, 82, 23.8, NAVY, 2.0, 24.4, 0.5)
make_text(2, 25.5, "EXTENSIONS", 24.9, 0.3,
          ha="left", va="center", fontsize=11, fontweight="bold", color=NAVY)
make_box(2, 15.5, 30, 8, "Trust-gated KG-RAG\n(pepper graph)", LIGHT_BLUE, NAVY,
         11, 25.2, 0.4, bold_first=True)
make_box(35, 15.5, 30, 8, "TeaPle v2\nLLM-assisted expansion", LIGHT_BLUE, NAVY,
         11, 25.6, 0.4, bold_first=True)
make_box(68, 15.5, 30, 8, "3 domains\ntea \u2192 pepper \u2192 disease KGs", LIGHT_BLUE,
         NAVY, 11, 26.0, 0.4, bold_first=True)

# --- validation strip + sources (26.4-27.6) ---
make_box(2, 4.5, 96, 8, "", "#F2F2F2", GREY, 10.5, 26.4, 0.8)
make_text(50, 11.2, "VALIDATION THROUGHOUT", 26.4, 0.8,
          ha="center", va="center", fontsize=11, fontweight="bold", color=NAVY)
make_text(50, 7.3,
          "Expert audit of ~300 triples  \u00b7  Test set (665 triples)  \u00b7  NER F1 before vs after the loop  \u00b7  Milestone updates every 2\u20133 weeks",
          26.4, 0.8, ha="center", va="center", fontsize=10, color=GREY)
make_text(50, 1.8,
          "Sources: Huang group papers (Tea KG 2023, GNN-FTuckER 2025, Tea NER 2026, KG-RAG 2026, DF Part 1/2, Yu 2026)  \u00b7  Yu 2026 is simulation-only",
          27.2, 0.4, ha="center", va="center", fontsize=8.5, color=GREY,
          style="italic")

# ---------------------------------------------------------------- pulses
# Main vertical flow path: down the trust->M1 arrow, across the gap to the
# right margin, down past M1/M2/M3, then along to the distribution point and
# down into the TeaPle v2 box.
_MAIN_WAYPOINTS = [(86.5, 80.2), (86.5, 76.0), (97.0, 76.0), (97.0, 62.8),
                   (97.0, 56.3), (97.0, 46.2), (97.0, 39.3), (97.0, 29.0),
                   (50.0, 29.0), (50.0, 23.8)]
main_pulse = Pulse(densify_polyline(_MAIN_WAYPOINTS))
loop_pulse = Pulse(_ARC_PTS, color=TEAL, n_trail=6)

# Pulse windows (seconds): main 28-32, loop twice 32-34 and 34-36.
PULSE_WINDOWS = [(main_pulse, 28.0, 32.0),
                 (loop_pulse, 32.0, 34.0),
                 (loop_pulse, 34.0, 36.0)]


def update(frame):
    t = frame / FPS
    for u in updaters:
        u(t)
    for pulse, t0, t1 in PULSE_WINDOWS:
        if t0 <= t < t1:
            pulse.set_s(pulse.total * (t - t0) / (t1 - t0))
        elif t < t0 or t >= t1:
            # hide only if no other active window owns this pulse now
            if not any(p is pulse and a <= t < b
                       for p, a, b in PULSE_WINDOWS):
                pulse.set_s(None)
    return []


def main():
    global FPS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=DURATION)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--output", default="future_work_workflow_motion.mp4")
    ap.add_argument("--preview", action="store_true",
                    help="save 3 PNG check-frames instead of rendering video")
    ap.add_argument("--test-frames", type=int, default=0,
                    help="render only the first N frames (smoke test)")
    args = ap.parse_args()

    FPS = args.fps

    if args.preview:
        for tsec, name in [(5.0, "preview_early"), (20.0, "preview_mid"),
                           (39.0, "preview_end")]:
            update(int(tsec * FPS))
            fig.canvas.draw()
            fig.savefig(f"{name}.png", facecolor="white")
            print(f"saved {name}.png (t={tsec}s)")
        return

    nframes = args.test_frames or int(args.seconds * FPS)
    anim = FuncAnimation(fig, update, frames=nframes, interval=1000 / FPS,
                         blit=False)
    writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=6000,
                          extra_args=["-pix_fmt", "yuv420p",
                                      "-movflags", "+faststart"])
    print(f"rendering {nframes} frames ({nframes / FPS:.1f}s) -> {args.output}")
    anim.save(args.output, writer=writer)
    print("done")


if __name__ == "__main__":
    main()
