import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

cases = sys.argv[1:] if len(sys.argv) > 1 else ["solid", "p20", "p40"]
D = {k: np.load(f"out_{k}.npz") for k in cases}
label = {k: ("틈 없는 담 (공극률 0%)" if float(D[k]['porosity']) == 0
             else f"구멍 뚫린 담 (공극률 {float(D[k]['porosity'])*100:.0f}%)") for k in cases}
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
col = {cases[0]: "#c0392b"}
for k, cc in zip(cases[1:], ["#2e86c1", "#17a589"]): col[k] = cc

H = int(D[cases[0]]['H']); X0 = int(D[cases[0]]['X0']); U0 = float(D[cases[0]]['U0'])
q = float(D[cases[0]]['qdyn'])
xs, xe, ytop = X0 - int(2.5*H), X0 + 10*H, int(2.6*H)
ext = [(xs - X0)/H, (xe - X0)/H, 0, ytop/H]

stats = {}
for k in cases:
    d = D[k]; n0 = int(d['avg_start'])
    Cd = d['Fx'][n0:]/(q*H); Cm = d['M'][n0:]/(q*H**2)
    stats[k] = dict(Cd=Cd.mean(), Cd_std=Cd.std(), Cd_max=Cd.max(), Cm=Cm.mean(), Cm_max=Cm.max())
    print(k, {a: round(float(b), 3) for a, b in stats[k].items()})

# --- 1. mean flow: speed + streamlines --------------------------------------
fig, axs = plt.subplots(len(cases), 1, figsize=(11, 3.1*len(cases)), constrained_layout=True, squeeze=False); axs=axs[:,0]
for ax, k in zip(axs, cases):
    d = D[k]; ux = d['ux'][xs:xe, :ytop].T; uy = d['uy'][xs:xe, :ytop].T
    sp = np.hypot(ux, uy)/U0; sp[d['solid'][xs:xe, :ytop].T] = np.nan
    im = ax.imshow(sp, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1.3, aspect="equal")
    X = np.linspace(ext[0], ext[1], ux.shape[1]); Y = np.linspace(0, ext[3], ux.shape[0])
    ax.streamplot(X, Y, ux, uy, color="w", linewidth=0.6, density=1.6, arrowsize=0.6)
    wl = d['wall'][xs:xe, :ytop].T.astype(float); wl[wl == 0] = np.nan
    ax.imshow(wl, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1.2, aspect="equal")
    ax.set_title(f"{label[k]} — 시간평균 풍속과 유선", fontsize=11, loc="left")
    ax.set_ylabel("높이 y/H")
axs[-1].set_xlabel("담으로부터의 거리 x/H  (바람: 왼쪽 → 오른쪽)")
fig.colorbar(im, ax=axs, shrink=0.6, label="|u| / U∞")
fig.savefig("fig1_mean_flow.png", dpi=140); plt.close(fig)

# --- 2. mean pressure ------------------------------------------------------
fig, axs = plt.subplots(len(cases), 1, figsize=(11, 3.1*len(cases)), constrained_layout=True, squeeze=False); axs=axs[:,0]
for ax, k in zip(axs, cases):
    d = D[k]; p = d['p'][xs:xe, :ytop].T
    pinf = d['p'][5, ytop]
    cp = (p - pinf)/q; cp[d['solid'][xs:xe, :ytop].T] = np.nan
    im = ax.imshow(cp, origin="lower", extent=ext, cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="equal")
    wl = d['wall'][xs:xe, :ytop].T.astype(float); wl[wl == 0] = np.nan
    ax.imshow(wl, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1.2, aspect="equal")
    ax.set_title(f"{label[k]} — 시간평균 압력계수 Cp", fontsize=11, loc="left"); ax.set_ylabel("y/H")
axs[-1].set_xlabel("x/H")
fig.colorbar(im, ax=axs, shrink=0.6, label="Cp  (빨강: 미는 압력, 파랑: 빨아당기는 흡입)")
fig.savefig("fig2_pressure.png", dpi=140); plt.close(fig)

# --- 3. forces ----------------------------------------------------------------
fig = plt.figure(figsize=(12, 4.4), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1, 1])
ax = fig.add_subplot(gs[0])
for k in cases:
    d = D[k]; n0 = int(d['avg_start']); Cd = d['Fx']/(q*H)
    t = np.arange(len(Cd))*U0/H
    m = np.convolve(Cd, np.ones(20)/20, mode="same")
    ax.plot(t[max(0,n0-2000):], m[max(0,n0-2000):], color=col[k], lw=1, label=label[k])
ax.set_xlabel("무차원 시간 tU∞/H"); ax.set_ylabel("항력계수 Cd = F/(½ρU²H)")
ax.set_title("담이 받는 바람 힘의 시간 변화", loc="left", fontsize=11)
ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.3)
names = [label[k].split(" (")[1].rstrip(")").replace("공극률 ", "") for k in cases]
ax2 = fig.add_subplot(gs[1])
x = np.arange(len(cases))
ax2.bar(x - 0.18, [stats[k]['Cd'] for k in cases], 0.36, color=[col[k] for k in cases], label="평균")
ax2.bar(x + 0.18, [stats[k]['Cd_max'] for k in cases], 0.36, color=[col[k] for k in cases], alpha=0.45, label="최대(돌풍)")
ax2.set_xticks(x, names); ax2.set_title("항력계수 Cd", loc="left", fontsize=11)
ax2.legend(fontsize=8, frameon=False); ax2.grid(alpha=0.3, axis="y")
ax3 = fig.add_subplot(gs[2])
ax3.bar(x - 0.18, [stats[k]['Cm'] for k in cases], 0.36, color=[col[k] for k in cases])
ax3.bar(x + 0.18, [stats[k]['Cm_max'] for k in cases], 0.36, color=[col[k] for k in cases], alpha=0.45)
ax3.set_xticks(x, names); ax3.set_title("전도 모멘트 계수 Cm", loc="left", fontsize=11)
ax3.grid(alpha=0.3, axis="y")
fig.savefig("fig3_forces.png", dpi=140); plt.close(fig)

# --- 4. vorticity animation ------------------------------------------------
nfr = min(len(D[k]['snaps']) for k in cases)
fig, axs = plt.subplots(len(cases), 1, figsize=(9, 2.6*len(cases)), constrained_layout=True, squeeze=False); axs=axs[:,0]
ims = []
for ax, k in zip(axs, cases):
    v = D[k]['snaps'][0][xs:xe, :ytop].T*H/U0
    ims.append(ax.imshow(v, origin="lower", extent=ext, cmap="RdBu_r", vmin=-8, vmax=8, aspect="equal"))
    wl = D[k]['wall'][xs:xe, :ytop].T.astype(float); wl[wl == 0] = np.nan
    ax.imshow(wl, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1.2, aspect="equal")
    ax.set_title(label[k] + " — 와도(소용돌이)", fontsize=10, loc="left"); ax.set_yticks([])
def upd(i):
    for im, k in zip(ims, cases): im.set_data(D[k]['snaps'][i][xs:xe, :ytop].T*H/U0)
    return ims
animation.FuncAnimation(fig, upd, frames=nfr, interval=80).save(
    "vorticity.mp4", writer=animation.FFMpegWriter(fps=12, bitrate=2500), dpi=120)
plt.close(fig)
np.save("stats.npy", stats, allow_pickle=True)
