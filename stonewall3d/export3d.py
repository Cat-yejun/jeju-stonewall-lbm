"""Build a rotatable 3D viewer (three.js) from lbm3d.py results.

    python export3d.py results_H40/solid.npz results_H40/p20.npz results_H40/p40.npz --out results_H40

Works with any results: the mean-flow scene (stones, streamlines through the gaps, reverse-flow zone) comes
from the time-averaged 3D field every run saves. The animated vortex volume appears only for results made by
an lbm3d.py that stores 3D frames (`snaps_q`).
Needs numpy and scikit-image (pip install scikit-image).
"""
import argparse, base64, gzip, json, os
import numpy as np
from skimage.measure import marching_cubes

from geometry import build_wall

HERE = os.path.dirname(os.path.abspath(__file__))


def pack(a):
    return base64.b64encode(gzip.compress(np.ascontiguousarray(a).tobytes(), 6)).decode()


def blur(a, n=1):
    """Separable 3-point box blur, n passes (edge-padded)."""
    a = a.astype(np.float32)
    for _ in range(n):
        for ax in range(3):
            p = np.pad(a, [(1, 1) if i == ax else (0, 0) for i in range(3)], mode="edge")
            sl = [slice(None)] * 3
            parts = []
            for s in (slice(0, -2), slice(1, -1), slice(2, None)):
                sl[ax] = s
                parts.append(p[tuple(sl)])
            a = (parts[0] + parts[1] + parts[2]) / 3.0
    return a


def to_three(x, y, z, m):
    """Simulation cells -> three.js world (units of H). X = downstream from the wall face, Y = up, Z = -span."""
    H = m["H"]
    return np.stack([(x - m["X0"]) / H, z / H, -(y / H - m["ny"] / (2 * H))], axis=-1).astype(np.float32)


def mesh_payload(verts, faces):
    nv = len(verts)
    return dict(nv=int(nv), nf=int(len(faces)), v=pack(verts.astype(np.float32)),
                f=pack(faces.astype(np.uint16 if nv < 65535 else np.uint32)), wide=bool(nv >= 65535))


def stones_mesh(m):
    H, X0, T = m["H"], m["X0"], m["T"]
    nx, ny, nz = m["nx"], m["ny"], m["nz"]
    wall, _ = build_wall(m["porosity_target"], H, nx, ny, nz, X0, T, seed=m["seed"], verbose=False)
    xa, xb, kb = X0 - 3, X0 + T + 3, min(nz, int(1.3 * H) + 3)
    sub = np.pad(wall[xa:xb, :, :kb], 1)                     # closed caps at the crop ends
    f = blur(sub, 1)
    step = 1
    v, fc, _, _ = marching_cubes(f, 0.5, step_size=step)
    v = v - 1.0                                              # undo the pad
    # index (i, j, k) -> cell coordinates; cell k has height k - 0.5
    return mesh_payload(to_three(v[:, 0] + xa + 0.5, v[:, 1] + 0.5, v[:, 2] - 0.5, m), fc)


class Field:
    """Trilinear sampler of the time-mean velocity stored at every 2nd cell (periodic in y)."""

    def __init__(self, d, m):
        self.u = d["mean3d_ds2"].astype(np.float32)          # (3, nx/2, ny/2, nz/2)
        self.solid = d["wall_ds2"]
        self.n = np.array(self.u.shape[1:])

    def __call__(self, p):                                   # p: (N, 3) in cells
        g = p / 2.0
        i0 = np.floor(g).astype(int)
        f = g - i0
        out = np.zeros((len(p), 3), np.float32)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ix = np.clip(i0[:, 0] + dx, 0, self.n[0] - 1)
                    iy = (i0[:, 1] + dy) % self.n[1]
                    iz = np.clip(i0[:, 2] + dz, 0, self.n[2] - 1)
                    w = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
                         * (f[:, 2] if dz else 1 - f[:, 2]))
                    out += w[:, None] * self.u[:, ix, iy, iz].T
        return out

    def is_solid(self, p):
        g = np.round(p / 2.0).astype(int)
        ix = np.clip(g[:, 0], 0, self.n[0] - 1)
        iy = g[:, 1] % self.n[1]
        iz = np.clip(g[:, 2], 0, self.n[2] - 1)
        return self.solid[ix, iy, iz] | (p[:, 2] < 1.0)


def trace(field, seeds, m, n_steps, h=1.0, keep=3):
    """Streamlines through the mean flow (geometric RK2, step h cells). Returns list of (pts, speed)."""
    UH, X0, H = m["UH"], m["X0"], m["H"]
    p = seeds.astype(np.float32).copy()
    alive = np.ones(len(p), bool)
    paths = [[p[i].copy()] for i in range(len(p))]
    speeds = [[float(np.linalg.norm(field(p[i:i + 1])[0])) / UH] for i in range(len(p))]
    for s in range(n_steps):
        idx = np.nonzero(alive)[0]
        if len(idx) == 0:
            break
        q = p[idx]
        u1 = field(q)
        n1 = np.linalg.norm(u1, axis=1)
        ok = n1 > 0.02 * UH
        mid = q + 0.5 * h * u1 / np.maximum(n1, 1e-9)[:, None]
        u2 = field(mid)
        n2 = np.linalg.norm(u2, axis=1)
        q2 = q + h * u2 / np.maximum(n2, 1e-9)[:, None]
        q2[:, 1] %= m["ny"]
        ok &= (n2 > 0.02 * UH) & ~field.is_solid(q2) & (q2[:, 0] < X0 + 10 * H) & (q2[:, 0] > X0 - 3 * H) \
            & (q2[:, 2] < 3 * H)
        p[idx] = q2
        for k, i in enumerate(idx):
            if not ok[k]:
                alive[i] = False
                continue
            if s % keep == 0:
                prev = next(pt for pt in reversed(paths[i]) if pt is not None)
                if abs(q2[k, 1] - prev[1]) > m["ny"] / 2:       # wrapped across the periodic edge: new segment
                    paths[i].append(None)
                paths[i].append(q2[k].copy())
                speeds[i].append(float(n2[k]) / UH)
    out = []
    for pts, sp in zip(paths, speeds):
        seg, sseg, j = [], [], 0
        for pt in pts:
            if pt is None:
                if len(seg) > 3:
                    out.append((np.array(seg), np.array(sseg)))
                seg, sseg = [], []
                continue
            seg.append(pt)
            sseg.append(sp[min(j, len(sp) - 1)])
            j += 1
        if len(seg) > 3:
            out.append((np.array(seg), np.array(sseg)))
    return out


def streamlines(d, m):
    H, X0, T, ny = m["H"], m["X0"], m["T"], m["ny"]
    field = Field(d, m)
    ys = (np.arange(16) + 0.5) * ny / 16
    zs = np.array([0.12, 0.3, 0.5, 0.7, 0.9, 1.15, 1.5]) * H
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    seeds = np.stack([np.full(Y.size, X0 - 2.0 * H), Y.ravel(), Z.ravel()], 1)
    lines = trace(field, seeds, m, n_steps=int(16 * H))
    # wake seeds: show the rolling flow behind the wall
    yw = (np.arange(6) + 0.25) * ny / 6
    xw = X0 + T + np.array([0.6, 1.5, 2.5]) * H
    zw = np.array([0.15, 0.45]) * H
    Xw, Yw, Zw = np.meshgrid(xw, yw, zw, indexing="ij")
    wake = trace(field, np.stack([Xw.ravel(), Yw.ravel(), Zw.ravel()], 1), m, n_steps=int(10 * H))
    allp, alls, offs = [], [], [0]
    for pts, sp in lines + wake:
        allp.append(to_three(pts[:, 0], pts[:, 1], pts[:, 2] - 0.5, m))
        alls.append(np.clip(np.round(sp / 1.4 * 255), 0, 255).astype(np.uint8))
        offs.append(offs[-1] + len(pts))
    return dict(n=len(offs) - 1, nWake=len(wake), off=pack(np.array(offs, np.int32)),
                p=pack(np.concatenate(allp)), s=pack(np.concatenate(alls)))


def reverse_zone(d, m):
    """Surface around the region where the time-mean wind blows backwards (u_x < 0) behind the wall."""
    H, X0 = m["H"], m["X0"]
    ux = d["mean3d_ds2"][0].astype(np.float32)
    solid = d["wall_ds2"]
    i0, i1 = X0 // 2, min(ux.shape[0], (X0 + 10 * H) // 2)
    k1 = min(ux.shape[2], int(2 * H) // 2)
    sub = np.where(solid[i0:i1, :, :k1], 0.05, ux[i0:i1, :, :k1]) / m["UH"]
    sub[:, :, 0] = sub[:, :, 1]                              # the ground row is not flow
    sub = np.pad(blur(sub, 1), 1, constant_values=1.0)
    if sub.min() >= 0:
        return None
    v, fc, _, _ = marching_cubes(-sub, 0.0)
    v = (v - 1.0) * 2.0
    vol = float((sub[1:-1, 1:-1, 1:-1] < 0).sum() * 8) / (H ** 2 * m["ny"])      # per unit wall length, in H^2
    return dict(mesh_payload(to_three(v[:, 0] + i0 * 2, v[:, 1], v[:, 2] - 0.5, m), fc), area=vol)


def vortex_frames(d, m):
    q = d["snaps_q"] if "snaps_q" in d.files else None
    if q is None or len(q) == 0 or "q3d" not in m:
        return None
    g = m["q3d"]
    ds, x0, (cx, cy, cz) = g["ds"], g["x0"], g["dims"]
    H = m["H"]
    wall, _ = build_wall(m["porosity_target"], H, m["nx"], m["ny"], m["nz"], m["X0"], m["T"], seed=m["seed"],
                         verbose=False)
    sub = wall[x0:x0 + cx * ds, :cy * ds, :cz * ds].copy()
    sub[:, :, 0] = True                                       # ground
    mask = sub.reshape(cx, ds, cy, ds, cz, ds).mean(axis=(1, 3, 5)) > 0.5
    frames = np.ascontiguousarray(q.transpose(0, 3, 2, 1))   # (F, z, y, x) = WebGL 3D texture order
    lo = to_three(np.float32(x0), np.float32(0), np.float32(-0.5), m)
    hi = to_three(np.float32(x0 + cx * ds), np.float32(cy * ds), np.float32(cz * ds - 0.5), m)
    return dict(dims=[cx, cy, cz], frames=len(q), data=pack(frames),
                mask=pack(np.ascontiguousarray(mask.transpose(2, 1, 0)).astype(np.uint8) * 255),
                lo=lo.tolist(), hi=hi.tolist(), qmax=g["qmax"],
                dtUnits=g["every"] / m["steps_per_unit"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    cases = []
    for f in a.files:
        d = np.load(f)
        m = json.loads(str(d["meta"]))
        n0 = m["n_spin"]
        Cd, Cm = d["Cd"][n0:], d["Cm"][n0:]
        c = dict(key=m["case"], porosity=round(m["porosity"], 3), porosityTarget=m["porosity_target"],
                 stats=dict(Cd=float(Cd.mean()), CdMax=float(Cd.max()), Cm=float(Cm.mean()), CmMax=float(Cm.max())),
                 stones=stones_mesh(m), lines=streamlines(d, m), reverse=reverse_zone(d, m),
                 vortex=vortex_frames(d, m),
                 domain=dict(span=m["ny"] / m["H"], thick=m["T"] / m["H"]))
        v = c["vortex"]
        print(f"{m['case']:>6}: stones {c['stones']['nf']} tris, {c['lines']['n']} streamlines, "
              f"reverse zone {'yes' if c['reverse'] else 'none'}, "
              f"vortex frames {v['frames'] if v else 'none (rerun with the new lbm3d.py to get them)'}", flush=True)
        cases.append(c)
    m0 = json.loads(str(np.load(a.files[0])["meta"]))
    info = dict(H=m0["H"], Re=m0["Re"], grid=[m0["nx"], m0["ny"], m0["nz"]])
    html = open(os.path.join(HERE, "viewer3d_template.html"), encoding="utf-8").read()
    html = html.replace("__DATA__", json.dumps(dict(cases=cases, info=info), separators=(",", ":")))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "stone_wall_scene3d.html")
    open(path, "w", encoding="utf-8").write(html)
    print("viewer:", path, f"({len(html)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
