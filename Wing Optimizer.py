from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess 
@dataclass
class SweepConfig:
    reynolds_min:float = 30_000
    reynolds_max:float = 70_000
    reynolds_steps:int = 5_000
    n_crit: float      = 5.5
    M_min = 1
    M_max = 6
    M_step = 1
    P_min = 2
    P_max = 5
    P_step = 1
    T_min = 8
    T_max = 14
    T_step = 2
    density= .00054 #g/mm^3
    wall_thickness=.56
    #design space steps 
    span_min=10
    span_max=256*3
    chord_min=10
    chord_max=230
    datapoints=50
    Cl_init=.5
    valid_glide=4
    printable = .42
    chordconfig= np.linspace(chord_min,chord_max,datapoints)
    spanconfig=np.linspace(span_min,span_max,datapoints)
    maxtrunc:float = .02
    
    #MPXX max camber/100 p for position of max camber /10 and XX for thickness /100; using T for XX
def paraGen(M,P,T, n=200):
    #x defined based on angle so that it takes sensible steps along the curvature were a sequential step in x wouldnt make sense
    beta= np.linspace(0,np.pi,n)
    x=(1-np.cos(beta))/2
    #Base numbers for naca profile definition
    M = M/ 100.0
    P = P/10.0
    T = T/100.0
    frontcamber = M/P**2*(2*P*x-x**2)
    dydx_front = 2*M/P**2*(P-x)
    dydx_back = 2*M/(1-P)**2*(P-x)
    backcamber = M/(1-P)**2*(1-2*P+2*P*x-x**2)
    dydx = np.where(x < P, dydx_front, dydx_back)
    yc = np.where(x < P, frontcamber, backcamber)
    #These are coefficients for the NACA equation
    a_0 =.2969
    a_1 =-.126
    a_2 =-.3516
    a_3 =.2843
    a_4 =-.1036
    #forumla for height of camberline 
    yt= (T/.2)*(a_0*x**.5+a_1*x+a_2*x**2+a_3*x**3+a_4*x**4)
    
    theta = np.arctan(dydx)
    #define x upper and y upper and the lowers and then stitch them toogether for one array of coordinates across the camber line
    xu=x-yt*np.sin(theta)
    yu=yc+yt*np.cos(theta)
    xl=x+yt*np.sin(theta)
    yl=yc-yt*np.cos(theta)
    upper = np.column_stack([xu,yu])
    bottom = np.column_stack([xl,yl])
    coords = np.vstack([upper[::-1],bottom])
    #calculate perimeter length for weight calculation
    dx=np.diff(xu)
    dy=np.diff(yu)
    upperlength=np.sum(np.sqrt(dx**2+dy**2))
    dx=np.diff(xl)
    dy=np.diff(yl)
    lowerlength=np.sum(np.sqrt(dx**2+dy**2))
    perimeter=upperlength+lowerlength
    thickness= yu-yl
    return coords, perimeter, thickness,xu

def buildDesignSpace(perimeter, cfg):
    #Create grid for design space
    chord_grid,span_grid=np.meshgrid(cfg.chordconfig,cfg.spanconfig)
    mass=perimeter*chord_grid*span_grid*cfg.density*cfg.wall_thickness
    #
    wingArea=span_grid*chord_grid
    
    #unit correction
    wingArea_m2=(chord_grid/1000)*(span_grid/1000)
    mass_kg = mass/1000
    rho=1.225
    Velocity=np.sqrt((2*mass_kg*9.81)/(rho*wingArea_m2*cfg.Cl_init))
    Re=Velocity*(chord_grid/1000)/(1.48e-5)
    return Re,chord_grid,span_grid,Velocity,mass

def mfCheck(cfg,thickness,xu):
    thickness_mm=thickness[:,np.newaxis]*cfg.chordconfig
    printable = thickness_mm >= cfg.printable
    lastchord_valid=(200-1) - np.argmax(printable[::-1], axis=0)
    percent = 1.0 - xu[lastchord_valid]
    validbool = percent <= cfg.maxtrunc
    validchords = cfg.chordconfig[validbool]
    validpercent = np.sum(validbool)/validbool.size
    return validchords, validpercent

def writeAirfoil(coords,M,P,T,polardir):
    name= f'NACA{M}{P}{T:02d}'
    filename=f'{name}.dat'
    polarfile=os.path.join(polardir, filename)
    with open(polarfile, 'w') as f:
        f.write(name+'\n')
        np.savetxt(f,coords,fmt='%.6f')
    return filename,name
def xfoilAnalysis(cfg,polardir,filename):
    resweep = np.arange(cfg.reynolds_min,cfg.reynolds_max + cfg.reynolds_steps, cfg.reynolds_steps)
    polar_files= [f"{filename}_Re{int(Re)}.txt" for Re in resweep]
    futures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for Re,polars in zip(resweep,polar_files):
            future=pool.submit(_xfoilSingleRe, Re,polars,filename,polardir,cfg)
            futures.append(future)
        results={}
        for future in as_completed(futures):
            Re,polar=future.result()
            if polar is not None:
                results[Re]=polar
    return results
def _xfoilSingleRe(Re,polar,filename,polardir,cfg):
    commands= (
        f"LOAD {filename}.dat\n"
        f"OPER\n"
        f"ITER\n"
        f"200\n"
        f"VPAR\nN {cfg.n_crit}\n\n"
        f"VISC {Re:.0f}\n"
        f"PACC\n{polar}\n\n"
        f"ASEQ -4 12 .5\n"
        f"PACC\n"
        f"QUIT\n"
        )
    try:
        results = subprocess.run(
            ["xfoil"],
            input=commands,
            text=True,
            timeout=300,
            cwd=polardir,
            )
    except subprocess.TimeoutExpired:
        print(f"XFOIL timed out on {filename}")
        return (Re, None)
    if os.path.exists(os.path.join(polardir, polar)):
        return (Re, polar)
    else:
        return (Re, None)
     
#here we start the program provide the function with its params and sit printing profile

M=4
P=4
T=12
cfg = SweepConfig()
workdir = os.path.join(os.getcwd(),"xfoil_work")
os.makedirs(workdir, exist_ok=True)
polardir = os.path.join(workdir, f"NACA{M}{P}{T:02d}")
os.makedirs(polardir, exist_ok=True)
coords,perimeter,thickness,xu = paraGen(M,P,T)
Re,chord_grid,span_grid, Velocity,mass=buildDesignSpace(perimeter,cfg)
valid_chords,validpercent=mfCheck(cfg,thickness,xu)
filename,name=writeAirfoil(coords,M,P,T,polardir)
xfoilAnalysis(cfg,polardir,name)


#debugging 
np.savetxt('weight_grid.csv', mass, delimiter=',', fmt='%.5g')
plt.plot(coords[:,0], coords[:,1])
plt.axis('equal')
plt.show()