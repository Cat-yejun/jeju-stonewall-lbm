"""2D D2Q9 lattice-Boltzmann (BGK + Smagorinsky LES) - wind over a wall on the ground.
Side view: x = wind direction, y = height. Ground at y=0 (no-slip).
Usage: python lbm_wall.py <case_name> <n_gaps> <gap_cells>
"""
import sys, numpy as np

case = sys.argv[1]; n_gaps = int(sys.argv[2]); gap = int(sys.argv[3])
NX, NY = 580, 160
H, T = 40, 6                # wall height, thickness (cells)
X0 = 120                    # wall left x
U0 = 0.08                   # inflow speed (lattice units)
NU = 0.0008                 # molecular viscosity -> Re = U0*H/NU = 4000
CS = 0.16                   # Smagorinsky constant
NSTEP, AVG_START = 16000, 6000

c = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]])
w = np.array([4/9]+[1/9]*4+[1/36]*4)
opp = [0,3,4,1,2,7,8,5,6]

# --- geometry -----------------------------------------------------------
wall = np.zeros((NX, NY), bool)
wall[X0:X0+T, 1:H+1] = True
if n_gaps:                                   # evenly spaced horizontal gaps
    centers = np.linspace(1, H+1, n_gaps+2)[1:-1]
    for yc in centers:
        y0 = int(round(yc - gap/2)); wall[X0:X0+T, y0:y0+gap] = False
ground = np.zeros((NX, NY), bool); ground[:, 0] = True
solid = wall | ground
porosity = 1 - wall[X0, 1:H+1].sum()/H

# inflow: 1/7 power-law atmospheric boundary layer
yy = np.arange(NY) + 0.5
uin = U0*np.clip(yy/(1.5*H), 0, 1)**(1/7); uin[0] = 0

def feq(rho, ux, uy):
    cu = 3*(c[:,0,None,None]*ux + c[:,1,None,None]*uy)
    usq = 1.5*(ux**2+uy**2)
    return rho*w[:,None,None]*(1+cu+0.5*cu**2-usq)

rho = np.ones((NX, NY)); ux = np.tile(uin, (NX,1)); uy = np.zeros((NX,NY))
ux[solid] = 0
f = feq(rho, ux, uy).astype(np.float32)
tau0 = 3*NU + 0.5
cx = c[:,0,None,None].astype(float); cy = c[:,1,None,None].astype(float)
ys = np.arange(NY)[None,:] - 0.5           # height above ground surface

Fx_hist, Fy_hist, M_hist = [], [], []
ux_sum = np.zeros((NX,NY)); uy_sum = np.zeros((NX,NY)); p_sum = np.zeros((NX,NY)); nsum = 0
snaps = []

for it in range(NSTEP):
    rho = f.sum(0); ux = (f*cx).sum(0)/rho; uy = (f*cy).sum(0)/rho
    # inlet (x=0) & top: equilibrium with prescribed profile
    ux[0] = uin; uy[0] = 0; rho[0] = 1
    ux[:, -1] = uin[-1]; uy[:, -1] = 0
    fe = feq(rho, ux, uy)
    f[:,0] = fe[:,0]; f[:,:,-1] = fe[:,:,-1]
    # Smagorinsky: local tau from non-equilibrium stress
    fn = f - fe
    Pxx = (fn*cx*cx).sum(0); Pyy = (fn*cy*cy).sum(0); Pxy = (fn*cx*cy).sum(0)
    Q = np.sqrt(Pxx**2 + Pyy**2 + 2*Pxy**2)
    tau = 0.5*(tau0 + np.sqrt(tau0**2 + 18*np.sqrt(2)*CS**2*Q/rho))
    fpost = f - fn/tau
    fpost[:, solid] = f[:, solid]              # solids are not collided
    # stream
    for i in range(9):
        f[i] = np.roll(fpost[i], (c[i,0], c[i,1]), axis=(0,1))
    # outlet: zero-gradient
    f[:, -1] = f[:, -2]
    # momentum-exchange force on wall: populations that just arrived in wall nodes
    fw = f[:, wall]                          # (9, nwall)
    # only links coming from fluid count: neighbour in -c_i must be fluid
    if it == 0:
        from_fluid = np.zeros((9, wall.sum()), bool)
        for i in range(9):
            nb = np.roll(np.roll(~solid, c[i,0], 0), c[i,1], 1)
            from_fluid[i] = nb[wall]
        yw = np.nonzero(wall)[1] - 0.5
    contrib = 2*fw*from_fluid
    Fx_n = (contrib*c[:,0,None]).sum(0); Fy_n = (contrib*c[:,1,None]).sum(0)
    # bounce-back in all solid nodes
    f[:, solid] = f[opp][:, solid]
    if it >= 1:
        Fx_hist.append(Fx_n.sum()); Fy_hist.append(Fy_n.sum()); M_hist.append((Fx_n*yw).sum())
    if it >= AVG_START:
        ux_sum += ux; uy_sum += uy; p_sum += rho/3; nsum += 1
        if it % 100 == 0:
            vort = (np.roll(uy,-1,0)-np.roll(uy,1,0) - np.roll(ux,-1,1)+np.roll(ux,1,1))/2
            snaps.append(np.where(solid, np.nan, vort).astype(np.float32))
    if it % 2000 == 0:
        print(case, it, flush=True)

qdyn = 0.5*1.0*U0**2
np.savez_compressed(f"out_{case}.npz", Fx=np.array(Fx_hist), Fy=np.array(Fy_hist),
    M=np.array(M_hist), ux=ux_sum/nsum, uy=uy_sum/nsum, p=p_sum/nsum, wall=wall, solid=solid,
    snaps=np.array(snaps[-120:]), porosity=porosity, H=H, U0=U0, qdyn=qdyn, X0=X0, T=T,
    avg_start=AVG_START)
print("done", case, "porosity", porosity)
