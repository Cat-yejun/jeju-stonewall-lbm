"""Irregular dry-stone wall generator (Jeju-style, one stone thick).

Coordinates (lattice cells): x = wind direction, y = along the wall (periodic), z = height.
Cell k=0 is the ground; the ground surface is at height zeta = 0 and cell k has its centre at zeta = k - 0.5.

Stones are rounded superellipsoids. Course by course, each stone is dropped onto the stones below until
it touches, so stones rest on each other and the gaps become irregular pockets rather than straight slots.
Porosity is set by the spacing between neighbouring stones in a course (bisection on `gap`).
"""
import numpy as np


def _profile(dy, b, p):
    """Half-height of a superellipse of half-width b at offset dy (0 outside)."""
    r = 1.0 - np.clip(np.abs(dy) / b, 0, 1) ** p
    return np.where(np.abs(dy) < b, r ** (1.0 / p), 0.0)


def make_stones(H, Ly, rng):
    """Random stone sizes/shapes per course (no positions in z yet)."""
    courses, z = [], 0.0
    while z < H * 0.97:
        h = rng.uniform(0.26, 0.38) * H
        courses.append(h)
        z += h
    rows = []
    for h in courses:
        lengths, tot = [], 0.0
        while tot < Ly:
            L = rng.uniform(0.45, 0.9) * H
            lengths.append(L)
            tot += L
        lengths = np.array(lengths) * Ly / tot
        y = rng.uniform(0, Ly)
        row = []
        for L in lengths:
            row.append(dict(yc=(y + L / 2) % Ly, L=L, c=0.5 * h * rng.uniform(0.85, 1.1),
                            p=rng.uniform(2.2, 3.2), dx=rng.uniform(-0.08, 0.08), ax=rng.uniform(0.85, 1.0)))
            y += L
        rows.append(row)
    return rows


def stack(rows, gap, H, Ly):
    """Drop stones course by course. gap in (0, 0.5): fraction of each stone's slot left empty in y."""
    ny = int(round(Ly))
    yg = np.arange(ny) + 0.5
    top = np.zeros(ny)
    stones = []
    for row in rows:
        new_top = top.copy()
        for st in row:
            b = 0.5 * st["L"] * (1.0 - gap)
            dy = (yg - st["yc"] + Ly / 2) % Ly - Ly / 2
            prof = _profile(dy, b, st["p"]) * st["c"]
            inside = np.abs(dy) < b
            zc = np.max(top[inside] + prof[inside])          # resting on the highest contact point below
            new_top = np.where(inside, np.maximum(new_top, zc + prof), new_top)
            stones.append(dict(st, b=b, zc=zc))
        top = new_top
    scale = H / top.mean()                                  # stretch so the mean top is the nominal height
    for st in stones:
        st["zc"] *= scale
        st["c"] *= scale
    return stones


def rasterize(stones, nx, ny, nz, X0, T):
    m = np.zeros((nx, ny, nz), bool)
    xc0 = X0 + T / 2
    for st in stones:
        a, b, c, p = st["ax"] * T / 2, st["b"], st["c"], st["p"]
        xc = xc0 + st["dx"] * T
        xi = np.arange(int(np.floor(xc - a)), int(np.ceil(xc + a)) + 1)
        ki = np.arange(max(1, int(np.floor(st["zc"] - c))), min(nz, int(np.ceil(st["zc"] + c)) + 2))
        yi = np.arange(int(np.floor(st["yc"] - b)) - 1, int(np.ceil(st["yc"] + b)) + 2)
        if len(xi) == 0 or len(ki) == 0:
            continue
        X, Y, K = np.meshgrid(xi + 0.5, yi + 0.5, ki - 0.5, indexing="ij")
        r = (np.abs(X - xc) / a) ** p + (np.abs(Y - st["yc"]) / b) ** p + (np.abs(K - st["zc"]) / c) ** p
        inside = r <= 1.0
        xs, ys, ks = np.meshgrid(xi, yi % ny, ki, indexing="ij")
        m[xs[inside], ys[inside], ks[inside]] = True
    return m


def front_mask(stones, ny, nz):
    """2D silhouette seen along x (cheap version of rasterize for the porosity search)."""
    yg = np.arange(ny) + 0.5
    kg = np.arange(nz) - 0.5
    m = np.zeros((ny, nz), bool)
    for st in stones:
        dy = (yg - st["yc"] + ny / 2) % ny - ny / 2
        half = _profile(dy, st["b"], st["p"]) * st["c"]
        m |= np.abs(kg[None, :] - st["zc"]) < half[:, None]
    m[:, 0] = False
    return m


def frontal_porosity(front, H):
    """Fraction of the nominal wall face (y in [0,Ly), height in [0,H)) you can see through."""
    return 1.0 - front[:, 1:int(round(H)) + 1].mean()


def mark_anchors(stones, ny):
    """Flag stones on the ground and stones that make up the top line of the packed wall."""
    yg = np.arange(ny) + 0.5
    env = np.zeros(ny)
    tops = []
    for st in stones:
        dy = (yg - st["yc"] + ny / 2) % ny - ny / 2
        t = np.where(np.abs(dy) < st["b"], st["zc"] + _profile(dy, st["b"], st["p"]) * st["c"], -np.inf)
        tops.append(t)
        env = np.maximum(env, t)
    for st, t in zip(stones, tops):
        st["top"] = bool(np.any(t >= env - 0.5))
        st["ground"] = bool(st["zc"] - st["c"] < 0.02 * st["c"] + 1e-6)
    return stones


def shrink(stones, s):
    """Shrink every stone about its centre by s (y and z). Top-line stones keep their top,
    ground stones keep touching the ground, so every porosity shares the same top line."""
    out = []
    for st in stones:
        n = dict(st, b=st["b"] * s, c=st["c"] * s)
        if st["top"]:
            n["zc"] = st["zc"] + st["c"] * (1 - s)
        elif st["ground"]:
            n["zc"] = st["zc"] - st["c"] * (1 - s)
        out.append(n)
    return out


def build_wall(porosity, H, nx, ny, nz, X0, T, seed=7, verbose=True):
    rng = np.random.default_rng(seed)
    rows = make_stones(H, ny, rng)
    base = mark_anchors(stack(rows, 0.0, H, ny), ny)        # tightly packed, stones touching
    if porosity <= 0.0:
        stones = base
        m = rasterize(stones, nx, ny, nz, X0, T)
        # mortar: fill the core of the wall up to the local stone top so nothing can pass through
        xa, xb = X0 + int(round(0.15 * T)), X0 + int(round(0.85 * T))
        col = m.any(axis=0)
        for j in range(ny):
            ks = np.nonzero(col[j])[0]
            if len(ks):
                m[xa:max(xb, xa + 1), j, 1:ks.max() + 1] = True
        s = 1.0
    else:
        lo, hi = 0.4, 1.2                                    # bisection on the stone scale
        for _ in range(24):
            s = 0.5 * (lo + hi)
            phi = frontal_porosity(front_mask(shrink(base, s), ny, nz), H)
            lo, hi = (s, hi) if phi > porosity else (lo, s)
        stones = shrink(base, s)
        m = rasterize(stones, nx, ny, nz, X0, T)
    phi = frontal_porosity(m.any(axis=0), H)
    if verbose:
        print(f"[wall] stones={len(base)} scale={s:.3f} frontal porosity={phi:.3f}", flush=True)
    return m, phi
