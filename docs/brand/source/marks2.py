"""Round 2 marks: A, C, F redrawn (two each) and four new masters."""
import numpy as np
from skimage.measure import find_contours
from scipy.sparse import diags, kron, identity
from scipy.sparse.linalg import spsolve
import brush as B
from brush import stroke as S, dot, catmull, seed
from marks import line, arc, wave_pts, petal

pi = np.pi


def poisson(srcs, n=161):
    h = 1.0 / (n + 1)
    e = np.ones(n)
    T = diags([e, -2 * e, e], [-1, 0, 1], shape=(n, n))
    A = (kron(identity(n), T) + kron(T, identity(n))) / h ** 2
    b = np.zeros((n, n))
    for (x, y, q) in srcs:
        b[int(y / h) - 1, int(x / h) - 1] = -q / h ** 2
    u = spsolve(A.tocsc(), b.ravel()).reshape(n, n)
    U = np.zeros((n + 2, n + 2))
    U[1:-1, 1:-1] = u
    return U


def contour_strokes(U, levels, x0, x1, gaps, starts, ws, centre=None, pick="longest"):
    m = U.shape[0] - 1
    out = []
    for lv, g, a0, w in zip(levels, gaps, starts, ws):
        cs = find_contours(U, lv)
        cs = sorted(cs, key=len, reverse=True)
        sel = cs if pick == "all" else cs[:1]
        for c in sel:
            if len(c) < 20:
                continue
            pts = np.c_[x0 + c[:, 1] / m * (x1 - x0), x0 + c[:, 0] / m * (x1 - x0)]
            cen = pts.mean(axis=0)
            ang = np.arctan2(pts[:, 1] - cen[1], pts[:, 0] - cen[0])
            k0 = np.argmin(np.abs(((ang - a0 + pi) % (2 * pi)) - pi))
            pts = np.r_[pts[k0:], pts[:k0]]
            pts = pts[: int(len(pts) * (1 - g))]
            out += [("ink", d) for d in S(pts, w, end="harai", head=0.05, tail=0.35, jitter=0.09)]
    return out


def hearth(x0, x1, w=3.0):
    """The ro-buchi in one breath: start top-left, clockwise, sweep out over the start."""
    def side(p, q, n=40):
        return line(p, q, n)
    r = 2.6
    segs = [side((x0 + r, x0), (x1 - r, x0 - 0.2)), arc((x1 - r, x0 + r), r, -pi / 2, 0, 12),
            side((x1, x0 + r), (x1 + 0.2, x1 - r)), arc((x1 - r, x1 - r), r, 0, pi / 2, 12),
            side((x1 - r, x1), (x0 + r, x1 + 0.2)), arc((x0 + r, x1 - r), r, pi / 2, pi, 12),
            side((x0, x1 - r), (x0 - 0.2, x0 + r)), arc((x0 + r, x0 + r), r, pi, 1.5 * pi, 12),
            side((x0 + r, x0), (x0 + 9, x0 - 0.5))]
    c = np.vstack(segs)
    return [("ink", d) for d in S(c, w, end="harai", head=0.03, tail=0.06, press=1.06, jitter=0.05, kasure=4, kasure_from=0.92)]


# ---------------------------------------------------------------- A1 · Haigata, redrawn
def mark_A1():
    seed(111)
    x0, x1 = 8.0, 48.0
    U = poisson([(0.40, 0.42, 1.0)])
    um = U.max()
    out = contour_strokes(U, um * np.array([0.40, 0.22, 0.105, 0.035]), x0, x1,
                          [0.12, 0.10, 0.14, 0.07], [2.4, 0.4, 3.9, 2.0], [1.35, 1.2, 1.05, 0.9])
    out += hearth(x0, x1)
    ex, ey = x0 + 0.40 * 40, x0 + 0.42 * 40
    out += [("accent", d) for d in dot((ex, ey), 2.9, ang=-0.5, jitter=0.1)]
    return dict(layers=out, refine=((16, 16), (16, 16)))


# ---------------------------------------------------------------- A2 · Futatsu-bi, two embers
def mark_A2():
    seed(122)
    x0, x1 = 8.0, 48.0
    s1, s2 = (0.34, 0.36, 1.0), (0.66, 0.62, 0.75)
    U = poisson([s1, s2])
    um = U.max()
    out = []
    # small rings around each ember, the waist that holds both, and the square one at the rim
    out += [("m2", d) for _, d in contour_strokes(U, [um * 0.29], x0, x1, [0.16], [2.2], [1.2], pick="all")]
    out += [("m4", d) for _, d in contour_strokes(U, [um * 0.20], x0, x1, [0.10], [0.6], [1.15])]
    out += contour_strokes(U, [um * 0.11, um * 0.04], x0, x1, [0.12, 0.08], [3.8, 2.2], [1.05, 0.9])
    out += hearth(x0, x1)
    for (x, y, q) in (s1, s2):
        out += [("accent", d) for d in dot((x0 + x * 40, x0 + y * 40), 2.5 * (0.8 + 0.2 * q), ang=-0.4)]
    return dict(layers=out, refine=((16, 16), (16, 16)))


# ---------------------------------------------------------------- C1 · Kasumi, redrawn
def mark_C1():
    seed(133)
    out = []
    X0, X1 = 9.0, 55.0
    P = X1 - X0
    out += [("accent", d) for d in S(wave_pts(X0, X1, 17, 1, 4.2, P), 6.2, end="harai", head=0.06, tail=0.18, press=1.08, kasure=5, kasure_from=0.82, jitter=0.06)]
    out += [("ink", d) for d in S(wave_pts(X0, X1, 31, 2, 3.0, P), 4.0, end="harai", head=0.06, tail=0.18, kasure=4, kasure_from=0.84, jitter=0.06)]
    out += [("ink", d) for d in S(wave_pts(X0, X1, 43, 4, 1.7, P), 2.4, end="harai", head=0.06, tail=0.2, jitter=0.06)]
    # the screen's two edges are one edge: brushed ticks, matched dots where each wave crosses
    for x in (6.0, 58.0):
        out += [("ink", d) for d in S(line((x, 9.0), (x + 0.1, 48.5)), 0.9, end="harai", jitter=0.08, press=1.0)]
        for y, r in ((17, 1.15), (31, 0.95), (43, 0.8)):
            out += [("ink", d) for d in dot((x, y), r)]
    return dict(layers=out, refine=((0, 0), (8, 8)))


# ---------------------------------------------------------------- C2 · Kasumi bunkai, one cloud read as three
def mark_C2():
    seed(144)
    out = []
    X0, X1 = 7.0, 57.0
    P = X1 - X0
    x = np.linspace(X0, X1, 300)
    ph = 2 * pi * (x - X0) / P
    y = 14 + 4.2 * np.sin(ph) + 2.4 * np.sin(2 * ph + 0.6) + 1.3 * np.sin(4 * ph + 1.4)
    out += [("ink", d) for d in S(np.c_[x, y], 3.6, end="harai", head=0.05, tail=0.15, press=1.1, kasure=4, kasure_from=0.84)]
    # the prism: a short pale cut where the cloud is read apart
    out += [("cut", d) for d in S(line((8.0, 25.8), (56.0, 25.5)), 0.7, end="harai", head=0.02, tail=0.6, jitter=0.1)]
    out += [("accent", d) for d in S(wave_pts(X0, X1, 33.5, 1, 4.2, P), 3.6, end="harai", head=0.06, tail=0.18, kasure=3, kasure_from=0.84)]
    out += [("m2", d) for d in S(wave_pts(X0, X0 + P * 0.9, 42.5, 2, 2.4, P, phase=0.6), 2.4, end="harai", head=0.06, tail=0.2)]
    out += [("m4", d) for d in S(wave_pts(X0, X0 + P * 0.82, 50.0, 4, 1.3, P, phase=1.4), 1.5, end="harai", head=0.06, tail=0.25)]
    return dict(layers=out, refine=((0, 0), (0, 0)))


# ---------------------------------------------------------------- F1 · Chō, redrawn as an agehachō crest
def wing(c, tip, bw, back=0.0, tail=None, n=16):
    """A closed wing outline from the body c out to tip, round on the outer edge."""
    c, tip = np.asarray(c, float), np.asarray(tip, float)
    ax = tip - c
    L = np.hypot(*ax)
    u = ax / L
    v = np.array([-u[1], u[0]])
    t = np.linspace(0, 1, n)
    lead = c + np.outer(t * L, u) + np.outer(bw * 0.35 * np.sin(pi * t) ** 1.2, v)
    trail = c + np.outer(t[::-1] * L * 0.98, u) - np.outer(bw * np.sin(pi * t[::-1] ** 0.75) ** 0.9, v)
    pts = np.r_[lead, trail[1:]]
    if tail is not None:
        k = len(lead) + n // 3
        pts = np.r_[pts[:k], [np.asarray(tail)], pts[k:]]
    return catmull(pts, per=6)


def mark_F1():
    seed(155)
    out = []
    c = np.array([32.0, 29.0])
    spec = [((9.5, 12.0), 8.8, "ink", 2.6), ((54.5, 12.0), -8.8, "ink", 2.6),
            ((13.0, 47.0), -6.4, "accent", 2.5), ((51.0, 47.0), 6.4, "ink", 2.3)]
    for nd, bw, role, w in spec:
        nd = np.asarray(nd)
        p = petal(c + (nd - c) * 0.05, nd + (c - nd) * 0.07, bw, n=18, asym=0.35)
        out += [(role, d) for d in S(p, w, end="harai", head=0.05, tail=0.16, press=1.2, jitter=0.07)]
        out += [(role, d) for d in dot(nd, 2.2)]
    out += [("ink", d) for d in S(line((32.0, 17.5), (32.2, 47.0)), 3.6, end="harai", head=0.15, tail=0.45, press=1.25)]
    for sgn in (-1, 1):
        a = catmull([(32 + 0.5 * sgn, 18.0), (32 + 3.5 * sgn, 11.0), (32 + 8.0 * sgn, 5.8)], per=12)
        out += [("ink", d) for d in S(a, 0.9, end="tome", press=1.0)]
        out += [("ink", d) for d in dot((32 + 8.3 * sgn, 5.5), 1.15)]
    return dict(layers=out, refine=((0, 32), (0, 40)))


# ---------------------------------------------------------------- F2 · Chō kōsa, the butterfly as its graph
def mark_F2():
    seed(166)
    out = []
    L, R, T, Bt = 11.0, 53.0, 14.0, 44.0
    out += [("ink", d) for d in S(line((L, T), (R, T + 0.3), bow=-1.2), 3.4, end="tome", press=1.1)]
    out += [("ink", d) for d in S(line((L, Bt), (R, Bt - 0.2), bow=1.2), 3.4, end="tome", press=1.1)]
    out += [("ink", d) for d in S(line((L, T), (R, Bt), bow=0.8), 3.0, end="harai", head=0.08, tail=0.3, kasure=3, kasure_from=0.75)]
    out += [("accent", d) for d in S(line((L, Bt), (R, T), bow=0.8), 3.6, end="harai", head=0.08, tail=0.3, press=1.15)]
    for p in [(L, T), (L, Bt), (R, T), (R, Bt)]:
        out += [("ink", d) for d in dot(p, 3.0)]
    # the ω mark carried by the crossing input
    out += [("accent", d) for d in S(arc((L, Bt), 6.2, -0.35 * pi, 0.15 * pi), 1.0, end="harai")]
    return dict(layers=out, refine=((0, 32), (0, 40)))


# ---------------------------------------------------------------- G · Hanabi, the fireworks master
def mark_G():
    seed(177)
    out = []
    c = np.array([30.0, 26.0])
    n = 16
    for k in range(n):
        a = 2 * pi * k / n + 0.12
        r0, r1 = 4.2, 19.5 + 1.6 * np.sin(3 * a)
        p = c + r0 * np.array([np.cos(a), np.sin(a)])
        q = c + r1 * np.array([np.cos(a), np.sin(a)])
        droop = np.array([0, 1.0]) * (r1 / 20) ** 2 * 1.6
        pts = np.r_[[p], [(p + q) / 2 + droop * 0.4], [q + droop]]
        role = "accent" if k % 4 == 0 else "ink"
        out += [(role, d) for d in S(catmull(pts, per=10), 1.5, end="harai", head=0.2, tail=0.55, press=1.2, start="point", jitter=0.08)]
        out += [(role, d) for d in dot(q + droop * 1.15 + (q - c) / r1 * 1.6, 0.95 if role == "ink" else 1.15)]
    out += [("accent", d) for d in dot(c, 2.2)]
    # the shell's climb
    out += [("ink", d) for d in S(catmull([(28.5, 62.0), (29.4, 50.0), (29.9, 40.0), (30.0, 33.5)], per=12), 1.0, end="harai", head=0.05, tail=0.5, start="point", kasure=2, kasure_from=0.6)]
    return dict(layers=out, refine=((16, 16), (24, 16)))


# ---------------------------------------------------------------- H · Bonshō, the bell founder
def mark_H():
    seed(188)
    out = []
    cx = 28.0
    top, mouth = 13.0, 47.0
    # crown and the dragon loop it hangs by
    crown = catmull([(18.6, 15.5), (21.0, 12.4), (28.0, 11.4), (35.0, 12.4), (37.4, 15.5)], per=10)
    out += [("ink", d) for d in S(crown, 2.4, end="tome", press=1.05)]
    out += [("ink", d) for d in S(catmull([(25.0, 11.8), (25.2, 7.6), (28.0, 5.4), (30.8, 7.6), (31.0, 11.8)], per=10), 1.7, end="tome")]
    for sgn in (-1, 1):
        side = catmull([(cx + sgn * 9.4, 15.0), (cx + sgn * 9.9, 26.0), (cx + sgn * 10.6, 38.0), (cx + sgn * 12.6, 46.5)], per=12)
        out += [("ink", d) for d in S(side, 2.4, end="tome", press=1.08)]
    out += [("ink", d) for d in S(line((cx - 13.8, mouth), (cx + 13.8, mouth - 0.2), bow=0.3), 3.2, end="tome", press=1.1, kasure=3, kasure_from=0.8)]
    # kesadasuki: the bands that divide the bronze
    out += [("ink", d) for d in S(line((cx - 9.6, 24.0), (cx + 9.6, 24.0)), 1.1, end="tome", press=1.0)]
    out += [("ink", d) for d in S(line((cx - 10.4, 36.5), (cx + 10.4, 36.4)), 1.1, end="tome", press=1.0)]
    out += [("ink", d) for d in S(line((cx, 15.5), (cx, 24.0)), 0.9, end="tome", press=1.0)]
    # nyū: the studs, a small lattice in each upper panel
    for ox in (cx - 7.6, cx + 2.6):
        for i in range(3):
            for j in range(3):
                out += [("ink", d) for d in dot((ox + 2.5 * i, 16.6 + 2.5 * j), 0.62, jitter=0.2)]
    # tsukiza, the lotus where the beam strikes
    out += [("accent", d) for d in dot((cx + 4.6, 41.6), 2.3)]
    # the sound, once
    out += [("accent", d) for d in S(arc((cx, 38.0), 17.5, pi * 0.62, pi * 0.38, n=60), 1.1, end="harai", head=0.1, tail=0.4)]
    return dict(layers=out, refine=((16, 32), (24, 32)))


# ---------------------------------------------------------------- I · Noborigama, the climbing kiln
def chamber(c, r, n=60):
    a = np.linspace(pi + 0.08, 2 * pi - 0.08, n)
    return np.c_[c[0] + r * np.cos(a), c[1] + r * 1.05 * np.sin(a)]


def mark_I():
    seed(199)
    out = []
    slope = catmull([(4.0, 56.0), (22.0, 46.5), (38.0, 35.0), (52.0, 25.0), (58.0, 21.0)], per=12)
    out += [("ink", d) for d in S(slope, 1.6, end="harai", head=0.05, tail=0.3, kasure=3, kasure_from=0.8)]
    ch = [((17.0, 49.0), 9.0, 2.8), ((32.0, 39.0), 6.8, 2.4), ((44.0, 30.5), 4.9, 2.0)]
    for (c, r, w) in ch:
        out += [("ink", d) for d in S(chamber(c, r), w, end="tome", head=0.06, tail=0.1, press=1.12)]
    # the firebox at the foot
    fl = catmull([(16.6, 48.6), (13.6, 45.6), (15.2, 41.6), (16.6, 37.6), (19.2, 42.2), (19.6, 45.4), (16.6, 48.6)], per=10)
    out += [("accent", d) for d in S(fl, 1.4, end="harai", press=1.3, head=0.05, tail=0.25)]
    out += [("accent", d) for d in dot((16.8, 45.0), 1.4, squash=1.2)]
    # chimney and its breath
    out += [("ink", d) for d in S(line((53.6, 25.0), (53.4, 9.0)), 2.6, end="tome", press=1.05)]
    out += [("ink", d) for d in S(catmull([(53.2, 7.4), (51.0, 5.4), (47.4, 4.8), (43.0, 3.0)], per=10), 0.9, end="harai", tail=0.6)]
    # ember between chambers: heat handed up the hill
    for (x, y) in ((25.0, 45.0), (38.7, 35.6)):
        out += [("accent", d) for d in dot((x, y), 0.95)]
    return dict(layers=out, refine=((0, 32), (8, 40)))


# ---------------------------------------------------------------- J · Kagaribi, the bonfire basket
def mark_J():
    seed(211)
    out = []
    cx = 28.0
    ty, by = 27.0, 37.0
    out += [("ink", d) for d in S(arc((cx, ty), 12.0, 0, 2 * pi * 0.97, ry=3.0), 1.9, end="harai", head=0.04, tail=0.15)]
    out += [("ink", d) for d in S(arc((cx, by), 7.0, 0.3, 0.3 + 2 * pi * 0.96, ry=1.8), 1.6, end="harai", head=0.04, tail=0.15)]
    out += [("ink", d) for d in S(arc((cx, (ty + by) / 2 + 0.6), 9.6, 0.0, pi, ry=2.4), 1.0, end="tome", press=1.0)]
    for t in np.linspace(-0.82, 0.82, 5):
        p = (cx + 12.0 * t, ty + 3.0 * np.sqrt(1 - t * t))
        q = (cx + 7.0 * t, by + 1.8 * np.sqrt(1 - t * t))
        out += [("ink", d) for d in S(line(p, q), 1.0, end="tome", press=1.0)]
    for (p, q, w) in (((cx - 5.5, by + 1.0), (cx - 13.0, 57.5), 2.2), ((cx + 5.5, by + 1.0), (cx + 13.0, 57.5), 2.2), ((cx + 0.3, by + 2.0), (cx + 0.6, 59.0), 1.8)):
        out += [("ink", d) for d in S(line(p, q), w, end="harai", head=0.05, tail=0.35)]
    # the fire: three tongues
    for (x, hgt, w, lean) in ((cx - 5.5, 12.0, 2.2, -2.0), (cx + 0.5, 20.0, 3.0, 1.6), (cx + 6.0, 10.5, 2.0, 2.4)):
        pts = catmull([(x, ty - 0.5), (x + lean * 0.3, ty - hgt * 0.35), (x - lean * 0.4, ty - hgt * 0.7), (x + lean, ty - hgt)], per=12)
        out += [("accent", d) for d in S(pts, w, end="harai", head=0.08, tail=0.6, press=1.25)]
    for (x, y, r) in ((40.5, 9.5, 0.9), (17.0, 11.0, 0.75), (36.0, 4.0, 0.6)):
        out += [("accent", d) for d in dot((x, y), r)]
    return dict(layers=out, refine=((16, 0), (24, 8)))


MARKS2 = dict(A1=mark_A1, A2=mark_A2, C1=mark_C1, C2=mark_C2, F1=mark_F1, F2=mark_F2, G=mark_G, H=mark_H, I=mark_I, J=mark_J)
