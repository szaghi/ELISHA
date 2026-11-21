"""Tiny sumi brush engine: centrelines -> filled SVG outlines."""
import numpy as np

RNG = np.random.default_rng(7)


def seed(s):
    global RNG
    RNG = np.random.default_rng(s)


def smooth_noise(n, scale=1.0, k=6):
    """Low-frequency noise of length n in [-1,1]."""
    m = max(3, n // k)
    knots = RNG.uniform(-1, 1, m)
    x = np.linspace(0, m - 1, n)
    return np.interp(x, np.arange(m), knots) * scale


def resample(pts, n=None, step=0.35):
    pts = np.asarray(pts, float)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    L = d[-1]
    if n is None:
        n = max(8, int(L / step))
    s = np.linspace(0, L, n)
    return np.c_[np.interp(s, d, pts[:, 0]), np.interp(s, d, pts[:, 1])], L


def catmull(ctrl, per=24, closed=False):
    P = np.asarray(ctrl, float)
    if closed:
        P = np.r_[P[-1:], P, P[:2]]
    else:
        P = np.r_[P[:1], P, P[-1:]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, per, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    return np.array(out)


def f(v):
    s = f"{v:.1f}"
    return "0.0" if s == "-0.0" else s


def f2(v):
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def poly_d(pts, fmt=f):
    pts = np.asarray(pts)
    return "M" + fmt(pts[0, 0]) + " " + fmt(pts[0, 1]) + "L" + " ".join(fmt(x) + " " + fmt(y) for x, y in pts[1:]) + "Z"


def profile(t, head=0.12, tail=0.3, end="harai", press=1.15, start="round"):
    """Width multiplier along the stroke."""
    w = np.ones_like(t)
    # entry: irikomi, a quick press
    a = np.clip(t / head, 0, 1)
    if start == "round":
        w *= 0.55 + 0.45 * np.sin(a * np.pi / 2)
    else:
        w *= 0.15 + 0.85 * np.sin(a * np.pi / 2)
    # body swell
    w *= 1 + (press - 1) * np.sin(np.pi * np.clip((t - head) / max(1e-6, 1 - head - tail), 0, 1))
    b = np.clip((1 - t) / tail, 0, 1)
    if end == "harai":  # sweep out to a point
        w *= np.power(b, 0.8) * 0.97 + 0.03
    elif end == "tome":  # stop: blunt
        w *= 0.7 + 0.3 * b
    return w


def stroke(centre, w0, end="harai", start="round", head=0.12, tail=0.3, press=1.15,
           jitter=0.07, kasure=0, kasure_from=0.62, step=0.3, fmt=f, wprof=None):
    """Return list of polygon d-strings for one brush stroke."""
    P, L = resample(centre, step=step)
    n = len(P)
    t = np.linspace(0, 1, n)
    T = np.gradient(P, axis=0)
    T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9
    N = np.c_[-T[:, 1], T[:, 0]]
    w = w0 * (wprof(t) if wprof is not None else profile(t, head, tail, end, press, start))
    w = w * (1 + smooth_noise(n, jitter, k=10))
    jl = 1 + smooth_noise(n, jitter * 0.9, k=5)
    jr = 1 + smooth_noise(n, jitter * 0.9, k=5)
    out = []
    if kasure <= 0:
        L_ = P + N * (w * jl / 2)[:, None]
        R_ = P - N * (w * jr / 2)[:, None]
        poly = np.r_[L_, cap(P[-1], T[-1], N[-1], w[-1] / 2, end), R_[::-1], cap(P[0], -T[0], -N[0], w[0] / 2, start)]
        out.append(poly_d(poly, fmt))
        return out
    # body up to kasure_from, then strands that break up
    cut = int(n * kasure_from)
    body = slice(0, cut + 2)
    L_ = P[body] + N[body] * (w[body] * jl[body] / 2)[:, None]
    R_ = P[body] - N[body] * (w[body] * jr[body] / 2)[:, None]
    poly = np.r_[L_, R_[::-1], cap(P[0], -T[0], -N[0], w[0] / 2, start)]
    out.append(poly_d(poly, fmt))
    k = kasure
    for j in range(k):
        off = (j + 0.5) / k - 0.5
        sw = 1.0 / k * (0.95 + RNG.uniform(-0.15, 0.1))
        stop = int(cut + (n - cut) * RNG.uniform(0.45, 1.0))
        stop = max(stop, cut + 4)
        s = slice(cut - 1, stop)
        ww = w[s]
        c = P[s] + N[s] * (ww * off * 0.92)[:, None]
        tt = np.linspace(0, 1, len(c))
        sw_ = ww * sw * (1 - 0.85 * tt ** 2)
        Ls = c + N[s] * (sw_ / 2)[:, None]
        Rs = c - N[s] * (sw_ / 2)[:, None]
        out.append(poly_d(np.r_[Ls, Rs[::-1]], fmt))
    return out


def cap(p, t, n, r, kind):
    if kind == "harai" or r < 0.05:
        return np.empty((0, 2))
    a = np.linspace(0, np.pi, 7)[1:-1]
    return np.array([p + n * r * np.cos(x) + t * r * np.sin(x) * (0.8 if kind == "tome" else 1) for x in a])


def dot(c, r, squash=0.85, ang=0.0, jitter=0.12, nseg=28, fmt=f):
    """A brushed dot (ten): pressed blob."""
    a = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
    rr = r * (1 + smooth_noise(nseg, jitter, k=4))
    rr *= 1 + 0.18 * np.cos(a - 0.6)  # heavier on one side
    x = rr * np.cos(a)
    y = rr * np.sin(a) * squash
    ca, sa = np.cos(ang), np.sin(ang)
    pts = np.c_[c[0] + x * ca - y * sa, c[1] + x * sa + y * ca]
    return [poly_d(pts, fmt)]


def ring(c, r, w, gap=0.0, a0=0.0, end="harai", kasure=0, rx=None, ry=None, fmt=f, step=0.3, jitter=0.07, press=1.1, head=0.08, tail=0.25):
    rx = rx or r
    ry = ry or r
    a = np.linspace(a0, a0 + 2 * np.pi * (1 - gap), 200)
    pts = np.c_[c[0] + rx * np.cos(a), c[1] + ry * np.sin(a)]
    return stroke(pts, w, end=end, kasure=kasure, fmt=fmt, step=step, jitter=jitter, press=press, head=head, tail=tail)


def dash_line(p, q, w, fmt=f, shrink=0.55, jitter=0.25):
    """One sumi dash for a cell edge: nodes stay bare."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    L = np.hypot(*d)
    u = d / L
    p2 = p + u * shrink * RNG.uniform(0.6, 1.4)
    q2 = q - u * shrink * RNG.uniform(0.6, 1.4)
    n = max(5, int(L / 1.2))
    t = np.linspace(0, 1, n)
    c = p2 + np.outer(t, q2 - p2)
    nn = np.array([-u[1], u[0]])
    wob = smooth_noise(n, w * 0.25, k=3)
    c = c + np.outer(wob, nn)
    ww = w * (0.55 + 0.45 * np.sin(np.pi * np.clip(t * 1.1, 0, 1))) * (1 + smooth_noise(n, jitter, k=3))
    Lp = c + np.outer(ww / 2, nn)
    Rp = c - np.outer(ww / 2, nn)
    return poly_d(np.r_[Lp, Rp[::-1]], fmt)


def amr_ground(r1=(0, 0), r2=(8, 8), size=64.0, widths=(0.36, 0.33, 0.3)):
    """Three-level brushed AMR ground. r1: origin of the 32x32 level-1 block,
    r2: origin of the 16x16 level-2 block (inside r1). Returns 3 d-strings."""
    L0, L1, L2 = [], [], []
    for x in (16, 32, 48):
        for y0 in range(0, 64, 16):
            L0.append(dash_line((x, y0), (x, y0 + 16), widths[0]))
            L0.append(dash_line((y0, x), (y0 + 16, x), widths[0]))
    ox, oy = r1
    for k in (8, 24):
        for s in range(0, 32, 8):
            L1.append(dash_line((ox + k, oy + s), (ox + k, oy + s + 8), widths[1]))
            L1.append(dash_line((ox + s, oy + k), (ox + s + 8, oy + k), widths[1]))
    ox, oy = r2
    for k in (4, 12):
        for s in range(0, 16, 4):
            L2.append(dash_line((ox + k, oy + s), (ox + k, oy + s + 4), widths[2]))
            L2.append(dash_line((ox + s, oy + k), (ox + s + 4, oy + k), widths[2]))
    return [" ".join(L0), " ".join(L1), " ".join(L2)]


def seal_E():
    outer = [(52.6, 53.6), (62.0, 53.5), (62.1, 62.7), (52.5, 63.1)]
    E = [(54.6, 55.4), (60.2, 55.5), (60.2, 56.7), (55.9, 56.7), (55.9, 57.7), (59.1, 57.7),
         (59.2, 58.9), (55.9, 58.9), (55.9, 59.9), (60.2, 59.9), (60.2, 61.1), (54.6, 61.2)]
    return poly_d(outer) + " " + poly_d(E)


def seal_E_icon(x=3.9, y=3.9, s=16.2):
    """Seal glyph for the licence icon (24 box)."""
    sc = s / 9.5
    def tr(pts):
        return [((px - 52.5) * sc + x, (py - 53.5) * sc + y) for px, py in pts]
    outer = tr([(52.6, 53.6), (62.0, 53.5), (62.1, 62.7), (52.5, 63.1)])
    E = tr([(54.6, 55.4), (60.2, 55.5), (60.2, 56.7), (55.9, 56.7), (55.9, 57.7), (59.1, 57.7),
            (59.2, 58.9), (55.9, 58.9), (55.9, 59.9), (60.2, 59.9), (60.2, 61.1), (54.6, 61.2)])
    return poly_d(outer, f2), poly_d(E, f2)
