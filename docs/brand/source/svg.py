import re
import numpy as np
import brush as B

OPAC = (0.186, 0.384, 0.62)


def bbox(ds):
    nums = []
    for d in ds:
        nums += [float(x) for x in re.findall(r"-?\d+\.?\d*", d)]
    a = np.array(nums).reshape(-1, 2)
    return a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()


def merge(layers):
    """Group consecutive same-role paths into one path each, keep order of first use."""
    groups = []
    for role, d in layers:
        if groups and groups[-1][0] == role:
            groups[-1][1].append(d)
        else:
            groups.append((role, [d]))
    return [(r, " ".join(ds)) for r, ds in groups]


FALL = dict(cut="ink", m2="ink", m4="ink", sum="ink")


def colour(role, pal, night):
    pre = "n_" if night else ""
    if pre + role in pal:
        return pal[pre + role]
    base = {"ink": "ink", "accent": "accent", "mesh": "mesh", "paper": "ground"}.get(role)
    if base:
        return pal[pre + base]
    if role == "seal":
        return pal.get(pre + "seal", pal[pre + "accent"])
    return colour(FALL[role], pal, night)


def mark_svg(m, pal, night=False, size=216, label="ELISHA mark", ground=True, seal=True):
    ink = pal["n_ink"] if night else pal["ink"]
    acc = pal["n_accent"] if night else pal["accent"]
    mesh = pal["n_mesh"] if night else pal["mesh"]
    sealc = pal.get("n_seal" if night else "seal", acc)
    col = {r: colour(r, pal, night) for r in ("ink", "accent", "mesh", "cut", "m2", "m4")}
    parts = []
    if ground:
        for d, o in zip(m["ground"], OPAC):
            parts.append(f'<path fill="{mesh}" opacity="{o}" d="{d}"/>')
    for role, d in merge(m["layers"]):
        parts.append(f'<path fill="{col[role]}" d="{d}"/>')
    if seal:
        parts.append(f'<path fill="{sealc}" fill-rule="evenodd" d="{B.seal_E()}"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="{size}" height="{size}" '
            f'role="img" aria-label="{label}">' + "".join(parts) + "</svg>")


def fav_svg(m, pal, night=False, size=54, label="favicon"):
    ink = pal["n_ink"] if night else pal["ink"]
    acc = pal["n_accent"] if night else pal["accent"]
    col = {r: colour(r, pal, night) for r in ("ink", "accent", "cut", "m2", "m4")}
    x0, y0, x1, y1 = m["fav_box"]
    parts = [f'<path fill="{col[r]}" d="{d}"/>' for r, d in merge(m["fav_layers"])]
    vb = f"{x0:.2f} {y0:.2f} {x1 - x0:.2f} {y1 - y0:.2f}"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="{size}" height="{size}" '
            f'role="img" aria-label="{label}">' + "".join(parts) + "</svg>")


def icon_svg(layers, pal, night=False, size=32, label="", ground=None):
    ink = pal["n_ink"] if night else pal["ink"]
    acc = pal["n_accent"] if night else pal["accent"]
    mesh = pal["n_mesh"] if night else pal["mesh"]
    paper = pal["n_ground"] if night else pal["ground"]
    sealc = pal.get("n_seal" if night else "seal", acc)
    col = dict(ink=ink, accent=acc, mesh=mesh, paper=paper, seal=sealc)
    parts = [f'<path fill="{col[r]}" d="{d}"/>' for r, d in merge(layers)]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="{size}" height="{size}" '
            f'role="img" aria-label="{label}">' + "".join(parts) + "</svg>")


def prepare(m):
    m["ground"] = B.amr_ground(*m["refine"])
    m["fav_layers"] = [(r, d) for r, d in m["layers"] if r in ("ink", "accent", "cut", "m2", "m4")]
    x0, y0, x1, y1 = bbox([d for _, d in m["fav_layers"]])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(x1 - x0, y1 - y0) * 1.04
    m["fav_box"] = (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)
    return m
