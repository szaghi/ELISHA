"""Nine ELISHA features, drawn six ways (one primitive per artist), 24x24."""
import numpy as np
import brush as B
from brush import seed, catmull
from marks import I, Id, arc, line, wave_pts, roof, mound, petal

pi = np.pi
FEATURES = ["FFT spectral solve", "AMR multigrid", "Periodic faces", "Dirichlet & Neumann", "Isolated domains",
            "Flexible CG", "MPI distributed", "GPU · cuFFT + OpenACC", "Round-off residual"]


def R(role, ds):
    return [(role, d) for d in ds]


def licence():
    o, e = B.seal_E_icon()
    return [("seal", o), ("paper", e)]


# ----------------------------------------------------------- A · rings, frame, ember
def frameA(x0, y0, x1, y1, w=1.5, role="ink"):
    out = []
    out += R(role, I(line((x0 - 0.4, y0), (x1 + 0.2, y0 - 0.1)), w, end="tome", jitter=0.04))
    out += R(role, I(line((x1, y0 - 0.3), (x1 + 0.1, y1 + 0.3)), w, end="tome", jitter=0.04))
    out += R(role, I(line((x1 + 0.3, y1), (x0 - 0.2, y1 + 0.1)), w, end="tome", jitter=0.04))
    out += R(role, I(line((x0, y1 + 0.3), (x0 - 0.1, y0 - 0.5)), w, end="harai", jitter=0.04))
    return out


def ringA(c, r, w=0.95, gap=0.18, a0=0.6, role="ink"):
    return R(role, I(arc(c, r, a0, a0 + 2 * pi * (1 - gap)), w, end="harai", head=0.08, tail=0.35))


def icons_A():
    seed(101)
    ic = []
    # 1 FFT: the rake counts the rings
    ic.append(R("accent", Id((12, 12), 1.9)) + ringA((12, 12), 4.6, 1.1) + ringA((12, 12), 7.4, 1.0, a0=2.0) + ringA((12, 12), 9.7, 0.9, gap=0.35, a0=3.6))
    # 2 multigrid: three hearths inside each other
    ic.append(frameA(2.5, 2.5, 21.5, 21.5, 1.5) + frameA(5.5, 9.5, 14.5, 18.5, 1.2) + frameA(7.2, 13.6, 11.2, 17.0, 1.1, role="accent"))
    # 3 periodic: the ring leaving one wall comes back through the other
    o = frameA(3, 3, 21, 21, 1.4)
    o += R("ink", I(arc((19.0, 12), 5.0, 0.5 * pi + 0.25, 1.5 * pi - 0.25), 1.0, end="harai"))
    o += R("ink", I(arc((1.0, 12), 5.0, -0.5 * pi + 0.65, 0.5 * pi - 0.65), 1.0, end="harai"))
    o += R("accent", Id((19.6, 12), 1.6))
    ic.append(o)
    # 4 Dirichlet & Neumann: one wall fixed hot, one wall that lets the flux through
    o = R("accent", I(line((4, 3), (4.2, 21)), 2.6, end="tome"))
    o += R("ink", I(line((3.6, 3.2), (20.5, 3.0)), 1.3, end="tome")) + R("ink", I(line((20.8, 21), (3.6, 21.1)), 1.3, end="tome"))
    for y in (4.5, 9.5, 14.5):
        o += R("ink", I(line((20.3, y), (20.4, y + 3.0)), 1.1, end="harai"))
    o += R("ink", I(arc((4, 12), 5.5, -1.2, 1.2), 0.9, end="harai")) + R("ink", I(arc((4, 12), 10.0, -0.85, 0.85), 0.85, end="harai"))
    ic.append(o)
    # 5 isolated: an ember in open ash, no hearth
    ic.append(R("accent", Id((12, 12), 1.8)) + ringA((12, 12), 4.4, 1.0, gap=0.08) + ringA((12, 12), 7.4, 0.8, gap=0.3, a0=2.2) + ringA((12, 12), 10.3, 0.6, gap=0.55, a0=4.0))
    # 6 flexible CG: the rake spirals in to the ember
    a = np.linspace(0, 4.2 * pi, 260)
    r = 10.0 * np.exp(-0.17 * a)
    sp = np.c_[12 + r * np.cos(a + 0.4), 12 + r * np.sin(a + 0.4)]
    ic.append(R("ink", I(sp, 1.1, end="harai", head=0.04, tail=0.25)) + R("accent", Id((12, 12), 1.5)))
    # 7 MPI: four hearths, one ember each
    o = []
    for k, (x, y) in enumerate([(2.5, 2.5), (13, 2.5), (2.5, 13), (13, 13)]):
        o += frameA(x, y, x + 8.5, y + 8.5, 1.1)
        o += R("accent" if k == 0 else "ink", Id((x + 4.25, y + 4.25), 1.3 if k == 0 else 1.0))
    ic.append(o)
    # 8 GPU: one brazier, many coals
    o = frameA(2.5, 2.5, 21.5, 21.5, 1.4)
    for i in range(3):
        for j in range(3):
            o += R("accent" if (i, j) == (1, 1) else "ink", Id((7 + 5 * i, 7 + 5 * j), 1.25))
    ic.append(o)
    # 9 round-off residual: raked flat, one grain left
    o = frameA(2.5, 2.5, 21.5, 21.5, 1.4)
    o += R("ink", I(line((6, 14.5), (18.5, 14.4)), 1.0, end="harai")) + R("ink", I(line((6, 17.8), (18.5, 17.7)), 0.9, end="harai"))
    o += R("accent", Id((12.2, 9.4), 0.85))
    ic.append(o)
    ic.append(licence())
    return ic


# ----------------------------------------------------------- B · blades
def bladeB(p, q, w=1.6, edge=True, role="ink", temper=False, sori=0.9):
    p, q = np.asarray(p, float), np.asarray(q, float)
    out = R(role, I(line(p, q, 30, bow=sori), w, end="harai", head=0.05, tail=0.3))
    if temper:
        d = q - p
        n = np.array([d[1], -d[0]]) / np.hypot(*d)
        t = np.linspace(0.1, 0.8, 90)
        c = p + np.outer(t, d) + np.outer(0.9 * np.sin(pi * t) * sori - 1.4 + 0.45 * np.sin(2 * pi * 4 * t), n) * -1
        out += R("accent", I(c, 0.75, end="harai"))
    return out


def icons_B():
    seed(202)
    ic = []
    # 1 FFT: the folded billet, layers doubling at every fold
    o = []
    for k, (y, n) in enumerate([(5.0, 1), (11.5, 2), (18.0, 4)]):
        o += R("ink", I(line((3, y), (21, y)), 1.4, end="tome"))
        for j in range(1, n):
            x = 3 + 18 * j / n
            o += R("accent" if k == 2 else "ink", I(line((x, y + 0.9), (x, y + 4.0)), 0.8, end="harai"))
    ic.append(o)
    # 2 multigrid: katana, wakizashi, tantō
    ic.append(bladeB((3, 21), (21, 3)) + bladeB((3, 13), (12.5, 3.6), 1.4) + bladeB((11, 21), (17.5, 14.6), 1.3, role="accent", sori=0.5))
    # 3 periodic: the tsuba, a ring with the blade's slot
    o = R("ink", I(arc((12, 12), 8.6, 0.3, 0.3 + 2 * pi * 0.95), 2.0, end="harai", head=0.05, tail=0.2))
    o += R("accent", I(line((12, 8.0), (12.1, 16.0)), 2.2, end="tome", press=1.0))
    o += R("ink", Id((7.6, 12), 0.9)) + R("ink", Id((16.4, 12), 0.9))
    ic.append(o)
    # 4 Dirichlet & Neumann: clay thick on the spine, thin at the edge
    o = R("accent", I(line((2.5, 7.0), (21.5, 6.6), bow=0.4), 3.2, end="tome", press=1.0))
    o += R("ink", I(line((2.5, 9.6), (21.5, 9.2), bow=0.4), 1.2, end="harai"))
    o += R("ink", I(line((2.5, 18.5), (21.5, 13.0), bow=0.6), 1.5, end="harai"))
    for x in (6, 11, 16):
        o += R("ink", I(line((x, 18.6 - (x - 2.5) * 0.29 + 1.2), (x + 0.6, 21.5)), 0.7, end="harai"))
    ic.append(o)
    # 5 isolated: one blade, sparks flying free
    o = bladeB((3.5, 20.5), (16, 8), 1.6)
    for (x, y, r) in [(19, 5, 1.0), (21.2, 9.3, 0.7), (16.8, 3.0, 0.6), (21.4, 2.6, 0.45)]:
        o += R("accent", Id((x, y), r))
    ic.append(o)
    # 6 flexible CG: the polisher's strokes, shorter each pass
    o = []
    for k, (L, y) in enumerate([(18, 5), (12, 10), (7, 15), (3, 19.5)]):
        o += R("ink" if k < 3 else "accent", I(line((21 - L, y), (21, y - 0.3)), 1.5 - 0.15 * k, end="harai"))
    ic.append(o)
    # 7 MPI: four strikers' hammers around one anvil
    o = R("accent", I(line((8.5, 12.2), (15.5, 12.0)), 3.4, end="tome", press=1.0))
    for (p, q) in [((3, 3), (7.5, 7.5)), ((21, 3), (16.5, 7.5)), ((3, 21), (7.5, 16.5)), ((21, 21), (16.5, 16.5))]:
        o += R("ink", I(line(p, q), 1.6, end="tome"))
    ic.append(o)
    # 8 GPU: the katana-kake, a rack of many blades
    o = []
    for k, y in enumerate([4.0, 8.4, 12.8, 17.2, 21.0]):
        o += R("accent" if k == 2 else "ink", I(line((3, y + 0.6), (21, y - 0.4), bow=0.5), 1.25, end="harai"))
    ic.append(o)
    # 9 round-off: the polished mirror, one last speck
    ic.append(bladeB((3, 20), (21, 4), 2.2) + R("ink", I(line((6, 20.6), (21.3, 7.0)), 0.6, end="harai")) + R("accent", Id((11.5, 9.0), 0.8)))
    ic.append(licence())
    return ic


# ----------------------------------------------------------- C · mist waves
def W(x0, x1, y, k, amp, per, w, role="ink", phase=0.0, end="harai"):
    return R(role, I(wave_pts(x0, x1, y, k, amp, per, phase), w, end=end, head=0.08, tail=0.25))


def icons_C():
    seed(303)
    ic = []
    ic.append(W(2.5, 21.5, 5.5, 1, 2.2, 19, 2.0, "accent") + W(2.5, 21.5, 12.5, 2, 1.6, 19, 1.5) + W(2.5, 21.5, 19, 4, 1.1, 19, 1.1))
    ic.append(W(2.5, 21.5, 5.5, 1, 2.0, 19, 1.8) + W(2.5, 12, 12.5, 1, 1.6, 9.5, 1.4) + W(2.5, 7.25, 19, 1, 1.2, 4.75, 1.1, "accent"))
    o = R("ink", I(line((3, 3), (3.1, 21)), 0.9, end="harai")) + R("ink", I(line((21, 3), (20.9, 21)), 0.9, end="harai"))
    o += W(0.0, 24.0, 12, 1, 4.0, 18, 1.6, "accent", phase=-pi * 2 * 3 / 18)
    ic.append(o)
    o = R("accent", I(line((3, 3), (3.1, 21)), 2.4, end="tome"))
    x = np.linspace(3, 21, 120)
    o += R("ink", I(np.c_[x, 12 - 6 * np.sin(pi / 2 * (x - 3) / 18)], 1.4, end="harai"))
    for y in (5, 10, 15):
        o += R("ink", I(line((21, y), (21, y + 3)), 0.9, end="harai"))
    ic.append(o)
    x = np.linspace(2, 22, 200)
    ic.append(R("ink", I(np.c_[x, 12 + 6 * np.exp(-((x - 12) / 4.2) ** 2) * np.cos(2 * pi * (x - 12) / 6)], 1.4, end="harai", head=0.15, tail=0.3)) + R("accent", Id((12, 18.4), 1.1)))
    x = np.linspace(2.5, 21.5, 200)
    ic.append(R("ink", I(np.c_[x, 12 + 7 * np.exp(-0.22 * (x - 2.5)) * np.sin(2 * pi * (x - 2.5) / 6.5)], 1.4, end="harai")) + R("accent", Id((21, 12), 1.1)))
    o = []
    for k, (a, b) in enumerate([(2, 6.5), (7.5, 11.5), (12.5, 16.5), (17.5, 22)]):
        o += W(a, b, 12, 0.5, 3.5, b - a, 1.6, "accent" if k == 1 else "ink", phase=0)
    o += W(2, 22, 20, 2, 0.8, 20, 0.8)
    ic.append(o)
    o = []
    for k, y in enumerate(np.linspace(4, 20, 6)):
        o += W(3, 21, y, 3, 0.8, 18, 0.85, "accent" if k == 2 else "ink")
    ic.append(o)
    ic.append(R("ink", I(line((2.5, 13), (21.5, 13)), 1.4, end="harai")) + W(9, 15, 9.0, 1, 0.5, 6, 0.7, "accent"))
    ic.append(licence())
    return ic


# ----------------------------------------------------------- D · roofs
def Rf(cx, y, hw, w, role="ink", lift=1.2, curl=0.9):
    return R(role, I(roof(cx, y, hw, lift=lift, curl=curl), w, end="harai", head=0.1, tail=0.2))


def pillar(x, y0, y1, w=0.8, role="ink"):
    return R(role, I(line((x, y0), (x, y1)), w, end="tome", press=1.0))


def icons_D():
    seed(404)
    ic = []
    o = pillar(12, 22, 9.5, 0.9)
    for k, (y, hw) in enumerate([(20, 4.2), (17.0, 3.4), (14.0, 2.6), (11.0, 1.8)]):
        o += R("ink", I(line((12 - hw, y), (12 + hw, y - 0.1)), 0.9, end="tome", press=1.0))
    fl = catmull([(12.1, 8.6), (10.6, 6.2), (11.6, 3.6), (12.2, 1.6), (13.4, 4.0), (13.4, 6.3), (12.1, 8.6)], per=8)
    o += R("accent", I(fl, 1.0, end="harai", press=1.3))
    ic.append(o)
    ic.append(Rf(12, 20.5, 10, 1.8) + Rf(12, 13.5, 6.5, 1.5) + Rf(12, 7.5, 3.6, 1.3, "accent") + pillar(12, 22, 6.2, 0.7))
    o = Rf(12, 6.5, 9.5, 1.6) + R("accent", I(arc((12, 14.6), 5.6, -1.3, -1.3 + 2 * pi * 0.94), 1.4, end="harai"))
    ic.append(o)
    o = Rf(12, 5.5, 10, 1.6) + pillar(5.5, 6.5, 21.5, 2.3, "accent")
    for y in (7.5, 12, 16.5):
        o += pillar(18.5, y, y + 3, 0.9)
    o += R("ink", I(line((3, 21.6), (21, 21.5)), 1.2, end="tome"))
    ic.append(o)
    o = Rf(12, 16, 8.0, 1.6) + pillar(12, 15, 6.5, 0.8) + R("accent", Id((12, 5.0), 1.3, squash=1.3))
    for x in (5, 9, 15, 19):
        o += R("ink", Id((x, 20.6), 0.45))
    ic.append(o)
    pts = np.array([(3, 21), (19, 18), (5, 14.5), (17, 11.5), (8, 8.6), (14, 6.5), (11.5, 5.0)])
    ic.append(R("ink", I(catmull(pts, per=10), 1.1, end="harai")) + R("accent", Id((11.6, 3.6), 1.2, squash=1.3)))
    o = []
    for k, (x, y) in enumerate([(6.5, 8.5), (17.5, 8.5), (6.5, 19.5), (17.5, 19.5)]):
        o += Rf(x, y - 1.5, 4.6, 1.3, "accent" if k == 0 else "ink", lift=0.8, curl=0.6) + pillar(x - 2.6, y - 0.8, y + 2.5, 0.7) + pillar(x + 2.6, y - 0.8, y + 2.5, 0.7)
    ic.append(o)
    o = Rf(12, 7, 10.5, 1.8)
    for k, x in enumerate(np.linspace(4, 20, 7)):
        o += pillar(x, 8.0, 19.5, 0.75, "accent" if k == 3 else "ink")
    o += R("ink", I(line((2.5, 20.8), (21.5, 20.7)), 1.3, end="tome"))
    ic.append(o)
    ic.append(R("ink", I(line((2.5, 17), (21.5, 16.9)), 2.0, end="tome", kasure=3, kasure_from=0.7)) + R("accent", Id((12, 11.5), 0.9, squash=1.3)))
    ic.append(licence())
    return ic


# ----------------------------------------------------------- E · mounds and embers
def Mo(x0, x1, base, h, xc, d, w, role="ink"):
    return R(role, I(mound(x0, x1, base, h, xc, d), w, end="harai", head=0.08, tail=0.3))


def icons_E():
    seed(505)
    ic = []
    ic.append(Mo(2, 22, 9.5, 6.5, 12, 4.0, 1.5) + Mo(2, 22, 15, 3.5, 12, 2.5, 1.1) + Mo(7, 17, 20.5, 2.2, 12, 1.2, 0.9) + R("accent", Id((12, 7.0), 1.0)))
    ic.append(Mo(1.5, 22.5, 21, 15, 12, 5.5, 1.5) + Mo(5, 19, 21, 9, 12, 3.5, 1.2) + Mo(8.5, 15.5, 21, 4.0, 12, 1.8, 1.0) + R("accent", Id((12, 19.4), 1.0)))
    o = R("ink", I(np.c_[np.linspace(1, 23, 300), 16 - 6 * (1 / np.cosh((((np.linspace(1, 23, 300) - 4.67) % 7.33) - 3.67) / 1.4))], 1.3, end="harai"))
    o += R("accent", Id((12.0, 14.4), 1.0)) + R("ink", I(line((1, 20.5), (23, 20.4)), 0.8, end="harai"))
    ic.append(o)
    o = R("accent", I(line((3, 3), (3.1, 21)), 2.4, end="tome"))
    x = np.linspace(3, 21, 100)
    o += R("ink", I(np.c_[x, 20 - 12 * np.sin(pi / 2 * (x - 3) / 18)], 1.4, end="harai"))
    o += R("ink", I(line((3, 21.4), (21, 21.3)), 0.9, end="harai"))
    ic.append(o)
    ic.append(Mo(1, 23, 17, 9, 12, 2.6, 1.3) + R("accent", Id((12, 14.8), 1.1)))
    ic.append(Mo(2, 22, 19, 13, 12, 3.4, 0.8) + Mo(2, 22, 19, 10, 12, 4.4, 0.9) + Mo(2, 22, 19, 8, 12, 5.4, 1.4) + R("accent", Id((12, 16.5), 1.0)))
    o = []
    for k, (x, y) in enumerate([(6.5, 10), (17.5, 10), (6.5, 21), (17.5, 21)]):
        o += Mo(x - 5, x + 5, y, 5.5, x, 1.8, 1.1)
        o += R("accent" if k == 1 else "ink", Id((x, y - 1.7), 0.8))
    ic.append(o)
    o = R("ink", I(catmull([(1.5, 19), (5, 11), (12, 8.5), (19, 11), (22.5, 19)], per=12), 1.5, end="harai"))
    for k, x in enumerate(np.linspace(6, 18, 5)):
        o += R("accent" if k == 2 else "ink", Id((x, 16.3), 0.85))
    o += R("ink", I(line((1.5, 20), (22.5, 19.9)), 0.8, end="harai"))
    ic.append(o)
    ic.append(R("ink", I(line((2, 16), (22, 15.9)), 1.6, end="harai", kasure=3, kasure_from=0.7)) + R("accent", Id((12, 18.4), 0.75)))
    ic.append(licence())
    return ic


# ----------------------------------------------------------- F · wings and nodes
def Pe(c, tip, bw, w=1.0, role="ink"):
    return R(role, I(petal(c, tip, bw, asym=0.2), w, end="harai", head=0.06, tail=0.2))


def icons_F():
    seed(606)
    ic = []
    o = R("ink", I(line((4, 5), (20, 5)), 1.1, end="tome")) + R("ink", I(line((4, 19), (20, 19)), 1.1, end="tome"))
    o += R("ink", I(line((4, 5), (20, 19)), 1.1, end="tome")) + R("accent", I(line((4, 19), (20, 5)), 1.3, end="tome"))
    for p in [(4, 5), (4, 19), (20, 5), (20, 19)]:
        o += R("ink", Id(p, 1.4))
    ic.append(o)
    o = []
    for k, (s, x) in enumerate([(1.0, 12), (0.6, 12), (0.3, 12)]):
        hw, hh = 9.5 * s, 8.0 * s
        o += Pe((12, 12), (12 - hw, 12 - hh), 3.2 * s, 0.9, "accent" if k == 2 else "ink") + Pe((12, 12), (12 + hw, 12 - hh), -3.2 * s, 0.9, "accent" if k == 2 else "ink")
        o += Pe((12, 12), (12 - hw * 0.8, 12 + hh * 0.8), -2.4 * s, 0.85, "accent" if k == 2 else "ink") + Pe((12, 12), (12 + hw * 0.8, 12 + hh * 0.8), 2.4 * s, 0.85, "accent" if k == 2 else "ink")
    ic.append(o)
    o = R("ink", I(arc((12, 12), 8.5, 0.2, 0.2 + 2 * pi * 0.93), 0.9, end="harai"))
    for k in range(4):
        a = k * pi / 2 - pi / 2
        p = (12 + 8.5 * np.cos(a), 12 + 8.5 * np.sin(a))
        o += R("accent" if k == 1 else "ink", Id(p, 1.5 if k == 1 else 1.2))
    o += Pe((12, 12), (12, 4.6), 2.0, 0.8) + Pe((12, 12), (19.4, 12), 2.0, 0.8, "accent")
    ic.append(o)
    o = R("accent", I(line((3, 3), (3.1, 21)), 2.4, end="tome"))
    o += Pe((3.6, 12), (19, 5.5), 3.2, 1.1) + Pe((3.6, 12), (18, 18.5), -2.8, 1.1)
    ic.append(o)
    o = Pe((12, 13), (5.5, 7.0), 2.6, 1.0) + Pe((12, 13), (18.5, 7.0), -2.6, 1.0) + Pe((12, 13), (7.0, 18.5), -2.0, 0.9) + Pe((12, 13), (17.0, 18.5), 2.0, 0.9)
    o += R("accent", I(line((12, 9), (12.1, 18)), 1.6, end="harai"))
    ic.append(o)
    a = np.linspace(0, 3.6 * pi, 220)
    r = 10 * np.exp(-0.2 * a)
    ic.append(R("ink", I(np.c_[12 + r * np.cos(a + 2.4), 12 + r * np.sin(a + 2.4)], 0.9, end="harai")) + Pe((12, 12), (9.8, 9.6), 1.1, 0.8, "accent") + Pe((12, 12), (14.2, 9.6), -1.1, 0.8, "accent"))
    o = []
    for k, (x, y) in enumerate([(6.5, 6.5), (17.5, 6.5), (6.5, 17.5), (17.5, 17.5)]):
        role = "accent" if k == 3 else "ink"
        o += Pe((x, y), (x - 4, y - 3.5), 1.4, 0.8, role) + Pe((x, y), (x + 4, y - 3.5), -1.4, 0.8, role) + Pe((x, y), (x - 3.2, y + 3.2), -1.1, 0.75, role) + Pe((x, y), (x + 3.2, y + 3.2), 1.1, 0.75, role)
    ic.append(o)
    o = []
    for k, (x, y) in enumerate([(5, 5), (12, 4.5), (19, 5.5), (5.5, 12), (12, 12), (18.5, 12), (5, 19), (12, 19.5), (19, 19)]):
        role = "accent" if k == 4 else "ink"
        o += R(role, I(line((x - 2.2, y - 1.8), (x + 2.2, y + 1.8)), 0.9, end="harai")) + R(role, I(line((x - 2.2, y + 1.8), (x + 2.2, y - 1.8)), 0.9, end="harai"))
    ic.append(o)
    ic.append(Pe((12, 20), (12.5, 4), 3.0, 1.2) + R("ink", I(line((5, 20.5), (19, 20.4)), 1.0, end="harai")) + R("accent", Id((17.5, 8.0), 0.8)))
    ic.append(licence())
    return ic


ICONS = dict(A=icons_A, B=icons_B, C=icons_C, D=icons_D, E=icons_E, F=icons_F)
LABELS = FEATURES + ["GPL v3"]
