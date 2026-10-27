"""Pack simulation results into a self-contained interactive HTML page."""
import sys, json, gzip, base64, numpy as np

cases = sys.argv[1:] if len(sys.argv) > 1 else ["solid", "p20", "p40"]
DS = 2                                       # spatial downsample for the animation
out = {"cases": []}
for k in cases:
    d = np.load(f"out_{k}.npz")
    H = int(d['H']); X0 = int(d['X0']); U0 = float(d['U0']); q = float(d['qdyn'])
    xs, xe, ytop = X0 - int(2.5*H), X0 + 10*H, int(2.6*H)
    snaps = d['snaps'][:, xs:xe, :ytop]                     # (F, nx, ny)
    snaps = snaps[:, ::DS, ::DS]
    F, nx, ny = snaps.shape
    v = np.clip(snaps*H/U0, -8, 8)
    qv = np.round((v + 8)/16*254).astype(np.uint8)
    qv[np.isnan(snaps)] = 255
    qv = np.ascontiguousarray(qv.transpose(0, 2, 1))       # (F, ny, nx), row 0 = ground
    # time-averaged speed (static field)
    sp = np.hypot(d['ux'], d['uy'])[xs:xe:DS, :ytop:DS].T/U0
    qs = np.round(np.clip(sp/1.3, 0, 1)*254).astype(np.uint8)
    qs[d['solid'][xs:xe:DS, :ytop:DS].T] = 255
    n0 = int(d['avg_start'])
    Cd = d['Fx']/(q*H); Cm = d['M']/(q*H**2)
    Cd_s = np.convolve(Cd, np.ones(25)/25, mode="same")
    win = Cd_s[n0:]                                          # step index i <-> sim step i+1
    # wall rectangles (in cropped lattice cells)
    wall = d['wall'][xs:xe, :ytop]
    cols = np.nonzero(wall.any(1))[0]
    rows = wall[cols[0]]
    rects, y = [], 0
    while y < len(rows):
        if rows[y]:
            y0 = y
            while y < len(rows) and rows[y]: y += 1
            rects.append([int(cols[0]), y0, int(len(cols)), y - y0])
        else: y += 1
    out["cases"].append(dict(
        key=k, porosity=round(float(d['porosity']), 3), nx=nx, ny=ny, frames=F,
        ds=DS, H=H, x0=X0 - xs, lx=int(xe - xs), ly=ytop,
        vort=base64.b64encode(gzip.compress(qv.tobytes(), 9)).decode(),
        speed=base64.b64encode(gzip.compress(qs.tobytes(), 9)).decode(),
        cd=[round(float(x), 3) for x in win[::10]],
        frameStep=100, cdStep=10,
        stats=dict(Cd=float(Cd[n0:].mean()), CdMax=float(Cd[n0:].max()), CdStd=float(Cd[n0:].std()),
                   Cm=float(Cm[n0:].mean()), CmMax=float(Cm[n0:].max())),
        walls=rects, U0=U0))
out["tUnit"] = out["cases"][0]["U0"]/out["cases"][0]["H"]   # tU/H per step
html = open("viewer_template.html").read().replace("__DATA__", json.dumps(out, separators=(",", ":")))
open("stone_wall_wind.html", "w").write(html)
print("html size MB", len(html)/1e6)
