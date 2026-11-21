"""Six round-1 marks for ELISHA and their icon packs."""
import numpy as np
from scipy.sparse import diags, kron, identity
from scipy.sparse.linalg import spsolve
from skimage.measure import find_contours
import brush as B
from brush import stroke, dot, ring, catmull, f2, seed

pi = np.pi


def S(*a, **k):
    return stroke(*a, **k)


def I(*a, **k):  # icon-scale stroke (2-decimal coordinates)
    k.setdefault("fmt", f2)
    k.setdefault("step", 0.15)
    return stroke(*a, **k)


def Id(c, r, **k):
    k.setdefault("fmt", f2)
    return dot(c, r, **k)


def arc(c, r, a0, a1, n=80, rx=None, ry=None):
    a = np.linspace(a0, a1, n)
    return np.c_[c[0] + (rx or r) * np.cos(a), c[1] + (ry or r) * np.sin(a)]


def line(p, q, n=20, bow=0.0):
    p, q = np.asarray(p, float), np.asarray(q, float)
    t = np.linspace(0, 1, n)
    c = p + np.outer(t, q - p)
    d = q - p
    nn = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-9)
    return c + np.outer(bow * np.sin(pi * t), nn)


# ---------------------------------------------------------------- A · Haigata
def poisson_square(src=(0.40, 0.42), n=161):
    h = 1.0 / (n + 1)
    e = np.ones(n)
    T = diags([e, -2 * e, e], [-1, 0, 1], shape=(n, n))
    A = (kron(identity(n), T) + kron(T, identity(n))) / h ** 2
    b = np.zeros((n, n))
    i, j = int(src[1] / h) - 1, int(src[0] / h) - 1
    b[i, j] = -1.0 / h ** 2
    u = spsolve(A.tocsc(), b.ravel()).reshape(n, n)
    U = np.zeros((n + 2, n + 2))
    U[1:-1, 1:-1] = u
    return U


def mark_A():
    seed(11)
    x0, x1 = 8.0, 48.0
    U = poisson_square()
    m = U.shape[0] - 1
    umax = U.max()
    out = []
    levels = umax * np.array([0.42, 0.24, 0.12, 0.045])
    gaps = [0.10, 0.16, 0.22, 0.30]
    starts = [2.2, 0.9, 4.0, 2.6]
    ws = [1.25, 1.15, 1.05, 0.95]
    for lv, g, a0, w in zip(levels, gaps, starts, ws):
        cs = find_contours(U, lv)
        c = max(cs, key=len)
        pts = np.c_[x0 + c[:, 1] / m * (x1 - x0), x0 + c[:, 0] / m * (x1 - x0)]
        # rotate start point and leave a raked gap
        cen = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - cen[1], pts[:, 0] - cen[0])
        k0 = np.argmin(np.abs(((ang - a0 + pi) % (2 * pi)) - pi))
        pts = np.r_[pts[k0:], pts[:k0]]
        pts = pts[: int(len(pts) * (1 - g))]
        out += [("ink", d) for d in S(pts, w, end="harai", head=0.06, tail=0.35, jitter=0.1)]
    # the hearth frame (ro-buchi): four strokes, slight overrun at the corners
    fw = 3.0
    out += [("ink", d) for d in S(line((x0 - 1.2, x0), (x1 + 0.6, x0 - 0.2)), fw, end="tome", jitter=0.05)]
    out += [("ink", d) for d in S(line((x1, x0 - 0.8), (x1 + 0.2, x1 + 1.0)), fw, end="tome", jitter=0.05)]
    out += [("ink", d) for d in S(line((x1 + 0.8, x1), (x0 - 0.4, x1 + 0.3)), fw, end="tome", jitter=0.05)]
    out += [("ink", d) for d in S(line((x0, x1 + 0.8), (x0 - 0.2, x0 - 1.4)), fw, end="harai", kasure=3, jitter=0.05)]
    ex = x0 + 0.40 * (x1 - x0)
    ey = x0 + 0.42 * (x1 - x0)
    out += [("accent", d) for d in dot((ex, ey), 2.6, ang=-0.4)]
    return dict(layers=out, refine=((16, 16), (16, 16)))


# ---------------------------------------------------------------- B · Tsuchioki
def blade_frame(base, tip, sori, s):
    base, tip = np.asarray(base, float), np.asarray(tip, float)
    ax = tip - base
    L = np.hypot(*ax)
    u = ax / L
    v = np.array([u[1], -u[0]])  # left of the blade axis (spine side, convex)
    return base, u, v, L


def mark_B():
    seed(23)
    base, tip = (12.0, 50.0), (54.0, 11.0)
    b, u, v, L = blade_frame(base, tip, 0, 0)
    s = np.linspace(0, 1, 160)
    sori = 3.2 * np.sin(pi * (0.15 + 0.85 * s)) * 0.9
    h = 4.3 * (1 - 0.3 * s)
    P = lambda ss, vv: b + np.outer(ss * L, u) + np.outer(vv, v)
    spine = P(s, sori + h)
    edge_s = s[s < 0.9]
    edge = P(edge_s, (sori - h)[s < 0.9])
    # kissaki: the edge sweeps up to meet the spine at the tip
    ks = np.linspace(0.9, 1.0, 30)
    kh = np.interp(ks, s, h)
    ksor = np.interp(ks, s, sori)
    tt = (ks - 0.9) / 0.1
    kiss = P(ks, ksor - kh + 2 * kh * np.sin(tt * pi / 2) ** 1.6)
    edge = np.r_[edge, kiss]
    out = []
    out += [("ink", d) for d in S(spine, 2.6, end="harai", head=0.05, tail=0.25, press=1.1)]
    out += [("ink", d) for d in S(edge, 1.3, end="harai", head=0.05, tail=0.15, press=1.05)]
    # tang (nakago): a short heavier stroke behind the habaki
    ts = np.linspace(-0.22, -0.02, 20)
    tsor = 3.2 * np.sin(pi * (0.15 + 0.85 * 0)) * 0.9
    out += [("ink", d) for d in S(P(ts, np.full_like(ts, tsor) + np.linspace(-0.6, 0.0, len(ts))), 3.6, end="tome", start="round", press=1.0, jitter=0.05, wprof=lambda t: 0.75 + 0.25 * t)]
    # habaki collar
    out += [("ink", d) for d in S(np.r_[P(np.array([0.0]), np.array([tsor + h[0] + 1.2])), P(np.array([0.0]), np.array([tsor - h[0] - 1.2]))], 1.6, end="tome")]
    # the temper line (yakiba): where the clay let the heat go
    ts = np.linspace(0.06, 0.86, 220)
    hs = np.interp(ts, s, h)
    so = np.interp(ts, s, sori)
    wave = -hs + 2.1 + 0.75 * np.sin(2 * pi * 6.5 * ts) + 0.25 * np.sin(2 * pi * 13 * ts + 1)
    out += [("accent", d) for d in S(P(ts, so + wave), 1.25, end="harai", head=0.05, tail=0.3, jitter=0.12)]
    return dict(layers=out, refine=((32, 0), (40, 8)))


# ---------------------------------------------------------------- C · Kasumi
def wave_pts(x0, x1, y, k, amp, period, phase=0.0, n=200):
    x = np.linspace(x0, x1, n)
    return np.c_[x, y + amp * np.sin(2 * pi * k * (x - x0) / period + phase)]


def mark_C():
    seed(37)
    out = []
    X0, X1 = 7.0, 57.0
    P = X1 - X0
    out += [("accent", d) for d in S(wave_pts(X0, X1, 19, 1, 5.0, P), 4.6, end="harai", head=0.08, tail=0.22, press=1.12, kasure=4, kasure_from=0.78)]
    out += [("ink", d) for d in S(wave_pts(X0, X1, 33, 2, 3.4, P), 3.3, end="harai", head=0.08, tail=0.22, kasure=3, kasure_from=0.8)]
    out += [("ink", d) for d in S(wave_pts(X0, X0 + 3 * P / 4, 45, 4, 2.0, P), 2.2, end="harai", head=0.08, tail=0.25)]
    # the two edges of the screen are the same edge
    out += [("ink", d) for d in S(line((5.0, 10.5), (5.2, 50.5)), 1.0, end="harai", jitter=0.08)]
    out += [("ink", d) for d in S(line((59.0, 10.0), (58.8, 47.5)), 1.0, end="harai", jitter=0.08)]
    return dict(layers=out, refine=((0, 0), (0, 8)))


# ---------------------------------------------------------------- D · Gojū-no-tō
def roof(cx, y, hw, lift=2.4, curl=1.8, n=40):
    xs = np.array([-1.0, -0.82, -0.45, 0.0, 0.45, 0.82, 1.0]) * hw
    ys = np.array([-curl, 0.4, -0.2, -lift, -0.2, 0.4, -curl])
    return catmull(np.c_[cx + xs, y + ys], per=12)


def mark_D():
    seed(41)
    cx = 29.0
    out = []
    ys = [49.0, 40.0, 31.8, 24.6, 18.2]
    hws = [20.0, 16.6, 13.4, 10.4, 7.8]
    ws = [3.0, 2.7, 2.4, 2.1, 1.9]
    # shinbashira: the central pillar through every storey
    out += [("ink", d) for d in S(line((cx, 54.5), (cx + 0.1, 8.5)), 1.1, end="tome", jitter=0.05, press=1.0)]
    for y, hw, w in zip(ys, hws, ws):
        out += [("ink", d) for d in S(roof(cx, y, hw), w, end="harai", head=0.1, tail=0.18, press=1.15)]
        # the storey wall under the roof
        if y < 49:
            pass
    # storey walls: short verticals at 0.55 hw between roofs
    for (ya, hwa), yb in zip(zip(ys[1:], hws[1:]), ys[:-1]):
        for sgn in (-1, 1):
            xw = cx + sgn * hwa * 0.62
            out += [("ink", d) for d in S(line((xw, ya + 1.2), (xw, yb - 1.6)), 0.9, end="tome", press=1.0)]
    # platform
    out += [("ink", d) for d in S(line((cx - 15, 55.2), (cx + 15, 55.0)), 2.2, end="tome", kasure=3, kasure_from=0.7)]
    for sgn in (-1, 1):
        xw = cx + sgn * 12.4
        out += [("ink", d) for d in S(line((xw, 50.0), (xw, 54.0)), 0.9, end="tome", press=1.0)]
    # kurin: the nine rings of the spire, drawn as five ticks
    for i, yy in enumerate(np.linspace(15.2, 9.4, 5)):
        hw = 2.4 - 0.25 * i
        out += [("ink", d) for d in S(line((cx - hw, yy), (cx + hw, yy - 0.1)), 0.75, end="tome", press=1.0)]
    # suien: the flame finial, named for water so the temple will not burn
    fl = catmull([(cx + 0.2, 8.6), (cx - 1.8, 6.4), (cx - 0.6, 3.9), (cx + 0.2, 1.6), (cx + 1.5, 4.0), (cx + 1.6, 6.3), (cx + 0.2, 8.6)], per=10)
    out += [("accent", d) for d in S(fl, 1.3, end="harai", head=0.05, tail=0.25, press=1.3)]
    out += [("accent", d) for d in dot((cx + 0.1, 6.2), 1.2, squash=1.3)]
    return dict(layers=out, refine=((16, 0), (24, 0)))


# ---------------------------------------------------------------- E · Uzumibi
def mound(x0, x1, base, hgt, xc, d, n=200):
    x = np.linspace(x0, x1, n)
    g = d / np.sqrt((x - xc) ** 2 + d ** 2)
    g = (g - g.min()) / (g.max() - g.min())
    return np.c_[x, base - hgt * g]


def mark_E():
    seed(53)
    out = []
    out += [("ink", d) for d in S(mound(5.0, 52.0, 45.5, 19.0, 28.0, 12.0), 3.0, end="harai", head=0.06, tail=0.3, press=1.2, kasure=4, kasure_from=0.75)]
    out += [("ink", d) for d in S(line((3.0, 47.6), (50.5, 47.2), bow=-0.4), 1.0, end="harai", head=0.05, tail=0.5)]
    out += [("accent", d) for d in dot((28.2, 39.6), 2.5, ang=-0.3)]
    # kagerō: the warmth above the ash, felt rather than seen
    sh = catmull([(28.5, 20.0), (27.2, 16.4), (29.2, 12.6), (27.8, 8.6), (29.0, 5.0)], per=14)
    out += [("accent", d) for d in S(sh, 1.05, end="harai", head=0.1, tail=0.45, jitter=0.12)]
    return dict(layers=out, refine=((16, 32), (24, 32)))


# ---------------------------------------------------------------- F · Chō
def petal(c, tip, bw, n=14, asym=0.0):
    c, tip = np.asarray(c, float), np.asarray(tip, float)
    ax = tip - c
    L = np.hypot(*ax)
    u = ax / L
    v = np.array([-u[1], u[0]])
    t = np.linspace(0, 1, n)
    shape = np.sin(pi * t) ** 0.8 * (1 + asym * t)
    left = c + np.outer(t * L, u) + np.outer(shape * bw, v)
    right = c + np.outer(t[::-1] * L, u) - np.outer(shape[::-1] * bw, v)
    return catmull(np.r_[left, right[1:]], per=6)


def mark_F():
    seed(67)
    out = []
    c = np.array([32.0, 30.0])
    nodes = [(11.0, 15.0), (11.0, 45.0), (53.0, 15.0), (53.0, 45.0)]
    bws = [7.2, 5.2, 7.2, 5.2]
    for k, (nd, bw) in enumerate(zip(nodes, bws)):
        sgn = 1 if k in (0, 3) else -1
        p = petal(c + (np.asarray(nd) - c) * 0.04, np.asarray(nd) + (c - np.asarray(nd)) * 0.08, bw * sgn, asym=0.25)
        role = "accent" if k == 1 else "ink"
        out += [(role, d) for d in S(p, 1.9 if k != 1 else 2.0, end="harai", head=0.06, tail=0.2, press=1.2, jitter=0.08)]
        out += [(role, d) for d in dot(nd, 1.7)]
    # body
    out += [("ink", d) for d in S(line((32.0, 19.0), (32.3, 45.0)), 2.8, end="harai", head=0.12, tail=0.4, press=1.2)]
    # antennae
    out += [("ink", d) for d in S(catmull([(31.4, 19.5), (28.0, 12.0), (24.0, 7.5)], per=12), 0.8, end="harai")]
    out += [("ink", d) for d in S(catmull([(32.6, 19.5), (36.0, 12.0), (40.0, 7.5)], per=12), 0.8, end="harai")]
    return dict(layers=out, refine=((0, 32), (0, 40)))


MARKS = dict(A=mark_A, B=mark_B, C=mark_C, D=mark_D, E=mark_E, F=mark_F)
