"""Turn lbm3d.py results into an interactive HTML viewer, a GIF and summary figures.

    python postprocess.py results/solid.npz results/p20.npz results/p40.npz --out results
"""
import argparse, base64, gzip, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VMAX = 6.0                       # vorticity colour range in units of U_H/H


def load(path):
    d = np.load(path)
    m = json.loads(str(d["meta"]))
    return d, m


def crop(m):
    H, X0 = m["H"], m["X0"]
    xs, xe = max(0, X0 - int(2.5 * H)), min(m["nx"], X0 + 10 * H)
    ztop = min(m["nz"], int(2.6 * H) + 1)
    return xs, xe, ztop


def vort_side(s, H, UH):           # s: (3, nx, nz) ux,uy,uz -> omega_y
    ux, uz = s[0].astype(np.float32), s[2].astype(np.float32)
    return (np.gradient(ux, axis=1) - np.gradient(uz, axis=0)) * H / UH


def vort_plan(s, H, UH):           # s: (3, nx, ny) -> omega_z (periodic in y)
    ux, uy = s[0].astype(np.float32), s[1].astype(np.float32)
    dux_dy = (np.roll(ux, -1, 1) - np.roll(ux, 1, 1)) / 2
    return (np.gradient(uy, axis=0) - dux_dy) * H / UH


def q8(v, lo, hi, solid):
    q = np.round((np.clip(v, lo, hi) - lo) / (hi - lo) * 254).astype(np.uint8)
    q[solid] = 255
    return q


def pack(a):
    return base64.b64encode(gzip.compress(np.ascontiguousarray(a).tobytes(), 9)).decode()


def case_payload(path):
    d, m = load(path)
    H, UH, ds = m["H"], m["UH"], max(1, int(round(m["H"] / 20)))
    xs, xe, ztop = crop(m)
    side_solid = d["wall_xz"].copy()
    side_solid[:, 0] = True
    side_solid = side_solid[xs:xe, :ztop]
    plan_solid = d["wall_xy"][xs:xe]
    side = np.stack([q8(vort_side(s, H, UH)[xs:xe, :ztop], -VMAX, VMAX, side_solid) for s in d["snaps_xz"]])
    plan = np.stack([q8(vort_plan(s, H, UH)[xs:xe], -VMAX, VMAX, plan_solid) for s in d["snaps_xy"]])
    side, plan = side[:, ::ds, ::ds], plan[:, ::ds, ::ds]
    # time-mean speed: span-averaged side view, mid-height plan view
    sm = d["mean_xz_spanavg"]
    sp_side = np.sqrt(np.nan_to_num(sm[1]) ** 2 + np.nan_to_num(sm[3]) ** 2)[xs:xe, :ztop] / UH
    span_solid = np.isnan(sm[1])[xs:xe, :ztop] | side_solid
    pm = d["mean_xy_half"]
    sp_plan = np.hypot(pm[1], pm[2])[xs:xe] / UH
    ms = q8(sp_side, 0, 1.4, span_solid)[::ds, ::ds]
    mp = q8(sp_plan, 0, 1.4, plan_solid)[::ds, ::ds]
    n0 = m["n_spin"]
    Cd, Cm = d["Cd"][n0:], d["Cm"][n0:]
    k = max(1, int(m["steps_per_unit"] / 20))                      # ~20 samples per H/U
    sm_cd = np.convolve(Cd, np.ones(k) / k, mode="same")[::k]
    wf = d["wall_front"]
    tops = np.array([np.nonzero(wf[j])[0].max() - 0.5 if wf[j].any() else 0.0 for j in range(wf.shape[0])]) / H
    front = wf[:, :int(1.25 * H)]                                   # (ny, nz)
    return m, dict(
        key=m["case"], porosity=round(m["porosity"], 3),
        topMean=float(tops.mean()), topMax=float(tops.max()),
        side=dict(nx=side.shape[1], nz=side.shape[2], frames=pack(side.transpose(0, 2, 1)), mean=pack(ms.T)),
        plan=dict(nx=plan.shape[1], ny=plan.shape[2], frames=pack(plan.transpose(0, 2, 1)), mean=pack(mp.T)),
        nframes=int(side.shape[0]), frameUnits=m["snap_every"] / m["steps_per_unit"],
        cd=[round(float(v), 3) for v in sm_cd], cdUnits=k / m["steps_per_unit"],
        front=dict(ny=front.shape[0], nz=front.shape[1], bits=pack(np.packbits(front.T[::-1].ravel()))),
        x0=(m["X0"] - xs) / H, lx=(xe - xs) / H, lz=ztop / H, ly=m["ny"] / H, zhalf=(m["z_half"] - 0.5) / H,
        stats=dict(Cd=float(Cd.mean()), CdMax=float(Cd.max()), CdStd=float(Cd.std()),
                   Cm=float(Cm.mean()), CmMax=float(Cm.max()), Cl=float(d["Cl"][n0:].mean())),
    ), d


def figures(items, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    for fam in ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic"]:
        try:
            from matplotlib import font_manager
            font_manager.findfont(fam, fallback_to_default=False)
            plt.rcParams["font.family"] = fam
            break
        except Exception:
            pass
    plt.rcParams["axes.unicode_minus"] = False

    def lab(m):
        return "틈 없는 담" if m["porosity_target"] <= 0 else f"돌담 공극률 {m['porosity']*100:.0f}%"

    # GIF: per case, side section (left) and plan section at H/2 (right)
    n = len(items)
    F = min(len(d["snaps_xz"]) for m, d in items)
    fig, axs = plt.subplots(n, 2, figsize=(13, 2.5 * n), constrained_layout=True,
                            gridspec_kw=dict(width_ratios=[1, 1]), squeeze=False)
    ims = []
    for r, (m, d) in enumerate(items):
        H, UH = m["H"], m["UH"]
        xs, xe, ztop = crop(m)
        ext_s = [(xs - m["X0"]) / H, (xe - m["X0"]) / H, 0, ztop / H]
        ext_p = [(xs - m["X0"]) / H, (xe - m["X0"]) / H, 0, m["ny"] / H]
        ws = d["wall_xz"][xs:xe, :ztop].T.astype(float)
        wpn = d["wall_xy"][xs:xe].T.astype(float)
        ws[ws == 0] = np.nan
        wpn[wpn == 0] = np.nan
        a, b = axs[r]
        i1 = a.imshow(vort_side(d["snaps_xz"][0], H, UH)[xs:xe, :ztop].T, origin="lower", extent=ext_s,
                      cmap="RdBu_r", vmin=-VMAX, vmax=VMAX, aspect="equal")
        a.imshow(ws, origin="lower", extent=ext_s, cmap="Greys", vmin=0, vmax=1.3, aspect="equal")
        i2 = b.imshow(vort_plan(d["snaps_xy"][0], H, UH)[xs:xe].T, origin="lower", extent=ext_p,
                      cmap="RdBu_r", vmin=-VMAX, vmax=VMAX, aspect="equal")
        b.imshow(wpn, origin="lower", extent=ext_p, cmap="Greys", vmin=0, vmax=1.3, aspect="equal")
        a.set_title(f"{lab(m)} · 옆 단면 (가운데)", loc="left", fontsize=10)
        b.set_title(f"{lab(m)} · 위에서 본 단면 (높이 H/2)", loc="left", fontsize=10)
        for ax in (a, b):
            ax.set_yticks([])
        ims.append((i1, i2, m, d, xs, xe, ztop))

    def upd(f):
        out_ = []
        for i1, i2, m, d, xs, xe, ztop in ims:
            i1.set_data(vort_side(d["snaps_xz"][f], m["H"], m["UH"])[xs:xe, :ztop].T)
            i2.set_data(vort_plan(d["snaps_xy"][f], m["H"], m["UH"])[xs:xe].T)
            out_ += [i1, i2]
        return out_
    gif = os.path.join(out, "stone_wall_3d.gif")
    animation.FuncAnimation(fig, upd, frames=F).save(gif, writer=animation.PillowWriter(fps=10), dpi=70)
    plt.close(fig)

    # force summary
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    names = [lab(m) for m, d in items]
    st = [(d["Cd"][m["n_spin"]:], d["Cm"][m["n_spin"]:]) for m, d in items]
    x = np.arange(n)
    for ax, j, title in [(axs[0], 0, "항력계수 Cd"), (axs[1], 1, "전도 모멘트 계수 Cm")]:
        ax.bar(x - 0.18, [s[j].mean() for s in st], 0.36, label="평균", color="#2a78ae")
        ax.bar(x + 0.18, [s[j].max() for s in st], 0.36, label="최대(돌풍)", color="#b8432c")
        ax.set_xticks(x, names)
        ax.set_title(title, loc="left")
        ax.grid(alpha=0.3, axis="y")
    axs[0].legend(frameon=False)
    fig.savefig(os.path.join(out, "forces_3d.png"), dpi=130)
    plt.close(fig)
    return gif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-gif", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    payload, items = [], []
    for f in a.files:
        m, p, d = case_payload(f)
        payload.append(p)
        items.append((m, d))
        s = p["stats"]
        print(f"{m['case']:>6}  porosity {m['porosity']:.3f}  Cd {s['Cd']:.3f} (max {s['CdMax']:.3f}, std {s['CdStd']:.3f})"
              f"  Cm {s['Cm']:.3f} (max {s['CmMax']:.3f})  Cl {s['Cl']:.3f}")
    m0 = items[0][0]
    info = dict(H=m0["H"], Re=m0["Re"], grid=[m0["nx"], m0["ny"], m0["nz"]], Cs=m0["Cs"],
                tAvg=m0["n_avg"] / m0["steps_per_unit"], thick=m0["T"] / m0["H"], vmax=VMAX)
    data = json.dumps(dict(cases=payload, info=info), separators=(",", ":"))
    html = open(os.path.join(HERE, "viewer_template.html"), encoding="utf-8").read().replace("__DATA__", data)
    path = os.path.join(a.out, "stone_wall_3d.html")
    open(path, "w", encoding="utf-8").write(html)
    print("viewer:", path, f"({len(html)/1e6:.1f} MB)")
    if not a.no_gif:
        print("gif:", figures(items, a.out))


if __name__ == "__main__":
    main()
