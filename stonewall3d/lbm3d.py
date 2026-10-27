"""3D lattice-Boltzmann (D3Q19, regularized BGK + Smagorinsky LES) for wind over a dry-stone wall.

Runs on an NVIDIA GPU (device="cuda") or, slowly, on the CPU (device="cpu") through NVIDIA Warp.

    python lbm3d.py --case p40 --porosity 0.4                  # default: H=40 cells, GPU
    python lbm3d.py --case test --porosity 0.2 --H 12 --device cpu --t-spinup 10 --t-avg 10

Axes: x = wind direction, y = along the wall (periodic), z = height. Ground = solid cell layer k=0.
Boundaries: x=0 velocity inlet (1/7 power-law profile), x=end zero-gradient outlet behind a sponge layer,
top = free-slip, y = periodic, ground and stones = half-way bounce-back.
Wall force: momentum-exchange method, recorded per height level every step.
"""
import argparse, json, os, shutil, time
import numpy as np
import warp as wp

from geometry import build_wall

# ---------------------------------------------------------------- lattice
C = np.array([(0, 0, 0),
              (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
              (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
              (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
              (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1)], dtype=np.int32)
W = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12, dtype=np.float32)
OPP = np.array([0] + [i + 1 if i % 2 == 1 else i - 1 for i in range(1, 19)], dtype=np.int32)
MIRZ = np.array([int(np.nonzero((C == C[i] * np.array([1, 1, -1])).all(1))[0][0]) for i in range(19)], dtype=np.int32)
Q = 19

FLUID, WALL, GROUND = 0, 1, 2


@wp.func
def feq_i(w: float, cx: float, cy: float, cz: float, rho: float, ux: float, uy: float, uz: float):
    cu = 3.0 * (cx * ux + cy * uy + cz * uz)
    return w * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * (ux * ux + uy * uy + uz * uz))


@wp.kernel
def k_init(f: wp.array4d(dtype=float), flag: wp.array3d(dtype=wp.uint8), uin: wp.array(dtype=float),
           cx: wp.array(dtype=wp.int32), cy: wp.array(dtype=wp.int32), cz: wp.array(dtype=wp.int32),
           w: wp.array(dtype=float), noise: float, seed: int):
    x, y, z = wp.tid()
    ux = float(0.0)
    uy = float(0.0)
    uz = float(0.0)
    if flag[x, y, z] == wp.uint8(0):
        st = wp.rand_init(seed, (x * 7919 + y) * 7907 + z)
        u0 = uin[z]
        ux = u0 * (1.0 + noise * (2.0 * wp.randf(st) - 1.0))
        uy = u0 * noise * (2.0 * wp.randf(st) - 1.0)
        uz = u0 * noise * (2.0 * wp.randf(st) - 1.0)
    for i in range(19):
        f[i, x, y, z] = feq_i(w[i], float(cx[i]), float(cy[i]), float(cz[i]), 1.0, ux, uy, uz)


@wp.kernel
def k_step(f_in: wp.array4d(dtype=float), f_out: wp.array4d(dtype=float),
           flag: wp.array3d(dtype=wp.uint8), uin: wp.array(dtype=float),
           cx: wp.array(dtype=wp.int32), cy: wp.array(dtype=wp.int32), cz: wp.array(dtype=wp.int32),
           w: wp.array(dtype=float), opp: wp.array(dtype=wp.int32), mirz: wp.array(dtype=wp.int32),
           nx: int, ny: int, nz: int, nu0: float, cs: float, x_sponge: int, nu_sponge: float,
           force: wp.array3d(dtype=float), t: int):
    x, y, z = wp.tid()
    if flag[x, y, z] != wp.uint8(0):
        return
    if x == 0:                                   # inlet: fixed equilibrium
        u0 = uin[z]
        for i in range(19):
            f_out[i, x, y, z] = feq_i(w[i], float(cx[i]), float(cy[i]), float(cz[i]), 1.0, u0, 0.0, 0.0)
        return

    fl = wp.vector(length=19, dtype=float)
    fxw = float(0.0)
    fzw = float(0.0)
    hit = int(0)
    for i in range(19):
        xs = x - cx[i]
        ys = (y - cy[i] + ny) % ny
        zs = z - cz[i]
        j = i
        if xs >= nx:                             # outlet: zero-gradient
            xs = x
        if zs >= nz:                             # top: free-slip (specular reflection)
            zs = z
            j = mirz[i]
        val = float(0.0)
        fs = flag[xs, ys, zs]
        if fs != wp.uint8(0):                    # half-way bounce-back
            val = f_in[opp[i], x, y, z]
            if fs == wp.uint8(1):                # momentum exchange with the stone wall
                fr = val - w[i]                  # minus the rho=1 background (faces hidden by contacts)
                fxw = fxw - 2.0 * fr * float(cx[i])
                fzw = fzw - 2.0 * fr * float(cz[i])
                hit = 1
        else:
            val = f_in[j, xs, ys, zs]
        fl[i] = val
    if hit == 1:
        wp.atomic_add(force, t, z, 0, fxw)
        wp.atomic_add(force, t, z, 1, fzw)

    rho = float(0.0)
    mx = float(0.0)
    my = float(0.0)
    mz = float(0.0)
    pxx = float(0.0)
    pyy = float(0.0)
    pzz = float(0.0)
    pxy = float(0.0)
    pxz = float(0.0)
    pyz = float(0.0)
    for i in range(19):
        v = fl[i]
        ex = float(cx[i])
        ey = float(cy[i])
        ez = float(cz[i])
        rho += v
        mx += v * ex
        my += v * ey
        mz += v * ez
        pxx += v * ex * ex
        pyy += v * ey * ey
        pzz += v * ez * ez
        pxy += v * ex * ey
        pxz += v * ex * ez
        pyz += v * ey * ez
    ux = mx / rho
    uy = my / rho
    uz = mz / rho
    third = rho / 3.0
    # non-equilibrium stress
    nxx = pxx - third - rho * ux * ux
    nyy = pyy - third - rho * uy * uy
    nzz = pzz - third - rho * uz * uz
    nxy = pxy - rho * ux * uy
    nxz = pxz - rho * ux * uz
    nyz = pyz - rho * uy * uz

    nu = nu0
    if x > x_sponge:
        s = float(x - x_sponge) / float(nx - x_sponge)
        nu = nu0 + nu_sponge * s * s
    tau0 = 3.0 * nu + 0.5
    qn = wp.sqrt(nxx * nxx + nyy * nyy + nzz * nzz + 2.0 * (nxy * nxy + nxz * nxz + nyz * nyz))
    tau = 0.5 * (tau0 + wp.sqrt(tau0 * tau0 + 18.0 * wp.sqrt(2.0) * cs * cs * qn / rho))
    keep = 1.0 - 1.0 / tau

    for i in range(19):
        ex = float(cx[i])
        ey = float(cy[i])
        ez = float(cz[i])
        qq = ((ex * ex - 1.0 / 3.0) * nxx + (ey * ey - 1.0 / 3.0) * nyy + (ez * ez - 1.0 / 3.0) * nzz
              + 2.0 * (ex * ey * nxy + ex * ez * nxz + ey * ez * nyz))
        f_out[i, x, y, z] = feq_i(w[i], ex, ey, ez, rho, ux, uy, uz) + keep * w[i] * 4.5 * qq


@wp.kernel
def k_macro(f: wp.array4d(dtype=float), flag: wp.array3d(dtype=wp.uint8),
            cx: wp.array(dtype=wp.int32), cy: wp.array(dtype=wp.int32), cz: wp.array(dtype=wp.int32),
            u: wp.array4d(dtype=float)):
    x, y, z = wp.tid()
    if flag[x, y, z] != wp.uint8(0):
        u[0, x, y, z] = 1.0
        u[1, x, y, z] = 0.0
        u[2, x, y, z] = 0.0
        u[3, x, y, z] = 0.0
        return
    rho = float(0.0)
    mx = float(0.0)
    my = float(0.0)
    mz = float(0.0)
    for i in range(19):
        v = f[i, x, y, z]
        rho += v
        mx += v * float(cx[i])
        my += v * float(cy[i])
        mz += v * float(cz[i])
    u[0, x, y, z] = rho
    u[1, x, y, z] = mx / rho
    u[2, x, y, z] = my / rho
    u[3, x, y, z] = mz / rho


@wp.kernel
def k_accum(u: wp.array4d(dtype=float), acc: wp.array4d(dtype=float)):
    c, x, y, z = wp.tid()
    acc[c, x, y, z] = acc[c, x, y, z] + u[c, x, y, z]


@wp.kernel
def k_slice_xz(u: wp.array4d(dtype=float), y0: int, out: wp.array3d(dtype=float)):
    c, x, z = wp.tid()
    out[c, x, z] = u[c, x, y0, z]


@wp.kernel
def k_slice_xy(u: wp.array4d(dtype=float), z0: int, out: wp.array3d(dtype=float)):
    c, x, y = wp.tid()
    out[c, x, y] = u[c, x, y, z0]


@wp.func
def q_at(u: wp.array4d(dtype=float), flag: wp.array3d(dtype=wp.uint8), x: int, y: int, z: int, ny: int):
    """Q-criterion -0.5 * du_i/dx_j du_j/dx_i (central differences); 0 next to solids."""
    yp = (y + 1) % ny
    ym = (y - 1 + ny) % ny
    if (flag[x, y, z] != wp.uint8(0) or flag[x + 1, y, z] != wp.uint8(0) or flag[x - 1, y, z] != wp.uint8(0)
            or flag[x, yp, z] != wp.uint8(0) or flag[x, ym, z] != wp.uint8(0)
            or flag[x, y, z + 1] != wp.uint8(0) or flag[x, y, z - 1] != wp.uint8(0)):
        return 0.0
    a11 = 0.5 * (u[1, x + 1, y, z] - u[1, x - 1, y, z])
    a12 = 0.5 * (u[1, x, yp, z] - u[1, x, ym, z])
    a13 = 0.5 * (u[1, x, y, z + 1] - u[1, x, y, z - 1])
    a21 = 0.5 * (u[2, x + 1, y, z] - u[2, x - 1, y, z])
    a22 = 0.5 * (u[2, x, yp, z] - u[2, x, ym, z])
    a23 = 0.5 * (u[2, x, y, z + 1] - u[2, x, y, z - 1])
    a31 = 0.5 * (u[3, x + 1, y, z] - u[3, x - 1, y, z])
    a32 = 0.5 * (u[3, x, yp, z] - u[3, x, ym, z])
    a33 = 0.5 * (u[3, x, y, z + 1] - u[3, x, y, z - 1])
    return -0.5 * (a11 * a11 + a22 * a22 + a33 * a33 + 2.0 * (a12 * a21 + a13 * a31 + a23 * a32))


@wp.kernel
def k_qcoarse(u: wp.array4d(dtype=float), flag: wp.array3d(dtype=wp.uint8), x0: int, ds: int,
              nx: int, ny: int, nz: int, out: wp.array3d(dtype=float)):
    """Max-pool positive Q over ds^3 blocks of a crop starting at x=x0, y=0, z=0."""
    i, j, k = wp.tid()
    best = float(0.0)
    for a in range(ds):
        for b in range(ds):
            for c in range(ds):
                x = x0 + i * ds + a
                y = j * ds + b
                z = k * ds + c
                if x >= 1 and x < nx - 1 and z >= 1 and z < nz - 1:
                    best = wp.max(best, q_at(u, flag, x, y, z, ny))
    out[i, j, k] = best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, help="output name, e.g. solid / p20 / p40")
    ap.add_argument("--porosity", type=float, required=True, help="frontal open-area fraction of the wall (0 = mortared)")
    ap.add_argument("--H", type=int, default=40, help="wall height in cells (resolution)")
    ap.add_argument("--device", default="cuda", help="cuda / cuda:0 / cpu")
    ap.add_argument("--seed", type=int, default=7, help="stone layout seed (keep equal across cases)")
    ap.add_argument("--Re", type=float, default=15000, help="Reynolds number U_H*H/nu")
    ap.add_argument("--U0", type=float, default=0.08, help="free-stream speed above the boundary layer (lattice units)")
    ap.add_argument("--Cs", type=float, default=0.16, help="Smagorinsky constant")
    ap.add_argument("--up", type=float, default=4, help="domain length upstream of the wall [H]")
    ap.add_argument("--down", type=float, default=16, help="domain length downstream of the wall [H]")
    ap.add_argument("--span", type=float, default=4, help="domain width along the wall (periodic) [H]")
    ap.add_argument("--height", type=float, default=8, help="domain height [H]")
    ap.add_argument("--thick", type=float, default=0.3, help="wall thickness [H]")
    ap.add_argument("--t-spinup", type=float, default=20, help="spin-up time before averaging [H/U_H]")
    ap.add_argument("--t-avg", type=float, default=30, help="averaging / recording time [H/U_H]")
    ap.add_argument("--snap-dt", type=float, default=0.25, help="snapshot interval [H/U_H]")
    ap.add_argument("--snap3d-every", type=int, default=2,
                    help="store a 3D vortex (Q-criterion) frame every N snapshots (0 = off)")
    ap.add_argument("--snap3d-ds", type=int, default=0, help="3D frame downsampling (0 = auto, about 13 voxels per H)")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--ckpt-every", type=int, default=5000, help="save a resumable checkpoint every N steps (0 = off)")
    ap.add_argument("--no-resume", action="store_true", help="ignore an existing checkpoint and start over")
    a = ap.parse_args()

    wp.init()
    dev = a.device
    H = a.H
    nx, ny, nz = int(round((a.up + a.down) * H)), int(round(a.span * H)), int(round(a.height * H)) + 1
    X0, T = int(round(a.up * H)), max(2, int(round(a.thick * H)))
    zeta = np.arange(nz) - 0.5                                      # height of cell centres above ground
    delta = 3.0 * H
    uin = (a.U0 * np.clip(zeta / delta, 0, 1) ** (1 / 7)).astype(np.float32)
    uin[0] = 0.0
    UH = float(a.U0 * (H / delta) ** (1 / 7))                       # reference speed at wall top
    nu0 = UH * H / a.Re
    t_unit = H / UH                                                 # steps per H/U_H
    n_spin, n_avg = int(a.t_spinup * t_unit), int(a.t_avg * t_unit)
    n_total = n_spin + n_avg
    snap_every = max(1, int(round(a.snap_dt * t_unit)))
    x_sponge = int(nx - 3 * H)
    mem = 2 * Q * nx * ny * nz * 4 + 8 * nx * ny * nz * 4
    print(f"[grid] {nx}x{ny}x{nz} = {nx*ny*nz/1e6:.1f} M cells, ~{mem/1e9:.1f} GB  | U_H={UH:.4f} "
          f"nu={nu0:.2e} tau0={3*nu0+0.5:.5f}  Re={a.Re:.0f}  steps={n_total} (H/U = {t_unit:.0f} steps)", flush=True)

    wall, phi = build_wall(a.porosity, H, nx, ny, nz, X0, T, seed=a.seed)
    flag = np.zeros((nx, ny, nz), np.uint8)
    flag[wall] = WALL
    flag[:, :, 0] = GROUND

    d = dict(device=dev)
    f0 = wp.zeros((Q, nx, ny, nz), dtype=float, **d)
    f1 = wp.zeros((Q, nx, ny, nz), dtype=float, **d)
    flag_d = wp.array(flag, dtype=wp.uint8, **d)
    uin_d = wp.array(uin, dtype=float, **d)
    cx, cy, cz = (wp.array(np.ascontiguousarray(C[:, k]), dtype=wp.int32, **d) for k in range(3))
    w_d = wp.array(W, dtype=float, **d)
    opp_d = wp.array(OPP, dtype=wp.int32, **d)
    mirz_d = wp.array(MIRZ, dtype=wp.int32, **d)
    force = wp.zeros((n_total, nz, 2), dtype=float, **d)
    U = wp.zeros((4, nx, ny, nz), dtype=float, **d)
    Uacc = wp.zeros((4, nx, ny, nz), dtype=float, **d)
    y_mid, z_half = ny // 2, int(round(0.5 * H)) + 1
    sl_xz = wp.zeros((4, nx, nz), dtype=float, **d)
    sl_xy = wp.zeros((4, nx, ny), dtype=float, **d)
    # 3D vortex frames: Q-criterion, max-pooled over ds^3 blocks, on a crop around and behind the wall
    ds3 = a.snap3d_ds if a.snap3d_ds > 0 else max(1, int(round(H / 13)))
    q_x0 = max(1, X0 - int(round(0.5 * H)))
    q_dims = (min(int(round(9.0 * H)), nx - 1 - q_x0) // ds3, ny // ds3, min(int(round(2.5 * H)), nz - 1) // ds3)
    q_d = wp.zeros(q_dims, dtype=float, **d)
    QMAX = 400.0                                          # Q in units of (U_H/H)^2, log-quantised to uint8
    q_scale = (H / UH) ** 2

    def q_frame():
        wp.launch(k_qcoarse, dim=q_dims, inputs=[U, flag_d, q_x0, ds3, nx, ny, nz, q_d], device=dev)
        q = np.clip(q_d.numpy() * q_scale, 0, QMAX)
        return np.round(255 * np.log1p(q) / np.log1p(QMAX)).astype(np.uint8)

    shape = (nx, ny, nz)
    snaps_xz, snaps_xy, snaps_q, n_acc, t_start = [], [], [], 0, 0
    os.makedirs(a.out, exist_ok=True)
    ckpt = os.path.join(a.out, f"{a.case}.ckpt")
    sig = dict(grid=[nx, ny, nz], H=H, porosity=a.porosity, seed=a.seed, Re=a.Re, U0=a.U0, Cs=a.Cs,
               n_spin=n_spin, n_total=n_total, snap_every=snap_every, snap3d=[a.snap3d_every, ds3])
    if os.path.isdir(ckpt) and not a.no_resume:
        info = json.load(open(os.path.join(ckpt, "state.json")))
        if info["sig"] != sig:
            raise SystemExit(f"[{a.case}] checkpoint {ckpt} was made with different settings; "
                             f"use the same options, or --no-resume to start over")
        t_start, n_acc = info["t"], info["n_acc"]
        f0.assign(wp.array(np.load(os.path.join(ckpt, "f.npy")), dtype=float, **d))
        force.assign(wp.array(np.load(os.path.join(ckpt, "force.npy")), dtype=float, **d))
        Uacc.assign(wp.array(np.load(os.path.join(ckpt, "uacc.npy")), dtype=float, **d))
        sx, sy = np.load(os.path.join(ckpt, "snaps_xz.npy")), np.load(os.path.join(ckpt, "snaps_xy.npy"))
        snaps_xz, snaps_xy = list(sx), list(sy)
        snaps_q = list(np.load(os.path.join(ckpt, "snaps_q.npy")))
        print(f"[{a.case}] resumed from step {t_start} ({t_start/t_unit:.1f} H/U)", flush=True)
    else:
        wp.launch(k_init, dim=shape, inputs=[f0, flag_d, uin_d, cx, cy, cz, w_d, 0.02, a.seed], device=dev)

    def save_ckpt(t_next):
        tmp = ckpt + ".tmp"
        os.makedirs(tmp, exist_ok=True)
        np.save(os.path.join(tmp, "f.npy"), f0.numpy())
        np.save(os.path.join(tmp, "force.npy"), force.numpy())
        np.save(os.path.join(tmp, "uacc.npy"), Uacc.numpy())
        np.save(os.path.join(tmp, "snaps_xz.npy"), np.array(snaps_xz, dtype=np.float16).reshape(-1, 3, nx, nz))
        np.save(os.path.join(tmp, "snaps_xy.npy"), np.array(snaps_xy, dtype=np.float16).reshape(-1, 3, nx, ny))
        np.save(os.path.join(tmp, "snaps_q.npy"), np.array(snaps_q, dtype=np.uint8).reshape(-1, *q_dims))
        json.dump(dict(t=t_next, n_acc=n_acc, sig=sig), open(os.path.join(tmp, "state.json"), "w"))
        if os.path.isdir(ckpt):                       # swap in the new one only once it is complete
            shutil.rmtree(ckpt)
        os.replace(tmp, ckpt)
        print(f"[{a.case}] checkpoint saved at step {t_next}", flush=True)

    t0 = time.time()
    for t in range(t_start, n_total):
        wp.launch(k_step, dim=shape, device=dev,
                  inputs=[f0, f1, flag_d, uin_d, cx, cy, cz, w_d, opp_d, mirz_d,
                          nx, ny, nz, nu0, a.Cs, x_sponge, 0.05, force, t])
        f0, f1 = f1, f0
        if t >= n_spin:
            k = t - n_spin
            do_acc, do_snap = k % 5 == 0, k % snap_every == 0
            if do_acc or do_snap:
                wp.launch(k_macro, dim=shape, inputs=[f0, flag_d, cx, cy, cz, U], device=dev)
            if do_acc:
                wp.launch(k_accum, dim=(4, nx, ny, nz), inputs=[U, Uacc], device=dev)
                n_acc += 1
            if do_snap:
                wp.launch(k_slice_xz, dim=(4, nx, nz), inputs=[U, y_mid, sl_xz], device=dev)
                wp.launch(k_slice_xy, dim=(4, nx, ny), inputs=[U, z_half, sl_xy], device=dev)
                snaps_xz.append(sl_xz.numpy()[1:].astype(np.float16))   # ux, uy, uz
                snaps_xy.append(sl_xy.numpy()[1:].astype(np.float16))
                if a.snap3d_every > 0 and (k // snap_every) % a.snap3d_every == 0:
                    snaps_q.append(q_frame())
        if a.ckpt_every > 0 and (t + 1) % a.ckpt_every == 0 and t + 1 < n_total:
            save_ckpt(t + 1)
        if t % 1000 == 0 or t == n_total - 1:
            wp.synchronize_device(dev)
            el = time.time() - t0
            done = t + 1 - t_start
            mlups = done * nx * ny * nz / max(el, 1e-9) / 1e6
            rho_chk = ""
            if t % 5000 == 0:
                wp.launch(k_macro, dim=shape, inputs=[f0, flag_d, cx, cy, cz, U], device=dev)
                un = U.numpy()
                if not np.isfinite(un).all():
                    raise SystemExit(f"[{a.case}] diverged at step {t}")
                rho_chk = f" rho[{un[0].min():.3f},{un[0].max():.3f}] |u|max/U_H={np.abs(un[1:]).max()/UH:.2f}"
            print(f"[{a.case}] step {t}/{n_total}  t={t/t_unit:.1f} H/U  {mlups:.0f} MLUPS  "
                  f"eta {el/done*(n_total-t-1)/60:.1f} min{rho_chk}", flush=True)

    # ---------------------------------------------------------------- save
    fh = force.numpy()                                   # (steps, nz, 2)
    qref = 0.5 * UH ** 2
    Fx = fh[:, :, 0].sum(1)
    Fz = fh[:, :, 1].sum(1)
    M = (fh[:, :, 0] * zeta[None, :]).sum(1)
    A = H * ny
    mean = Uacc.numpy() / max(n_acc, 1)                  # (4, nx, ny, nz)
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(
        os.path.join(a.out, f"{a.case}.npz"),
        Cd=(Fx / (qref * A)).astype(np.float32), Cl=(Fz / (qref * A)).astype(np.float32),
        Cm=(M / (qref * A * H)).astype(np.float32),
        fx_profile=(fh[n_spin:, :, 0].mean(0) / (qref * ny)).astype(np.float32),
        mean_xz_mid=mean[:, :, y_mid, :].astype(np.float32),
        mean_xz_spanavg=np.where(wall.any(1)[None], np.nan, mean.mean(2)).astype(np.float32),
        mean_xy_half=mean[:, :, :, z_half].astype(np.float32),
        mean3d_ds2=mean[1:, ::2, ::2, ::2].astype(np.float16),
        snaps_xz=np.array(snaps_xz), snaps_xy=np.array(snaps_xy),
        wall_xz=wall[:, y_mid, :], wall_xy=wall[:, :, z_half], wall_front=wall.any(0), wall_top=wall.any(2),
        wall_ds2=wall[::2, ::2, ::2],
        snaps_q=np.array(snaps_q, dtype=np.uint8).reshape(-1, *q_dims),
        meta=json.dumps(dict(case=a.case, porosity_target=a.porosity, porosity=float(phi), H=H, T=T, X0=X0,
                             q3d=dict(x0=q_x0, ds=ds3, dims=list(q_dims), qmax=QMAX,
                                      every=a.snap3d_every * snap_every),
                             nx=nx, ny=ny, nz=nz, UH=UH, U0=a.U0, Re=a.Re, nu=nu0, Cs=a.Cs,
                             n_spin=n_spin, n_avg=n_avg, steps_per_unit=t_unit, snap_every=snap_every,
                             y_mid=y_mid, z_half=z_half, seed=a.seed, device=dev,
                             wall_seconds=time.time() - t0)))
    s = slice(n_spin, None)
    print(f"[{a.case}] done. porosity={phi:.3f}  Cd mean={(Fx[s]/(qref*A)).mean():.3f} "
          f"max={(Fx[s]/(qref*A)).max():.3f}  Cm mean={(M[s]/(qref*A*H)).mean():.3f}", flush=True)
    if os.path.isdir(ckpt):                               # finished: the checkpoint is no longer needed
        shutil.rmtree(ckpt)


if __name__ == "__main__":
    main()
