from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import namedtuple
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
    span_min=12
    span_max=256*3
    span_steps=14
    chord_min=10
    chord_max=230
    chord_steps=5
    Cl_init=.5
    valid_glide=4
    printable = .42
    chordconfig= np.arange(chord_min,chord_max,chord_steps)
    chordconfig= np.append(chordconfig,chord_max)
    spanconfig=np.arange(span_min,span_max,span_steps)
    spanconfig = np.append(spanconfig,span_max)
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
    return validbool, validpercent
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
    return results,resweep
def _xfoilSingleRe(Re,polar,filename,polardir,cfg):
    commands= (
        f"LOAD {filename}.dat\n"
        f"PANE\n"
        f"OPER\n"
        f"ITER\n"
        f"200\n"
        f"VPAR\nN {cfg.n_crit}\n\n"
        f"VISC {Re:.0f}\n"
        f"PACC\n{polar}\n\n"
        f"ASEQ -4 12 .5\n"
        f"PACC\n"
        f"\nQUIT\n"
        )
    polarpath = os.path.join(polardir, polar)
    if os.path.exists(polarpath):
        os.remove(polarpath)
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
def _parseData(resultspath):
    with open(resultspath, 'r') as r:
        lines = r.readlines()
        for i, line in enumerate(lines):
            if line.strip().startswith('---'):
                data_start = i+1
                break
        if data_start >= len(lines):
            return None
        alpha_list=[]
        CL_list=[]
        CD_list=[]
        CM_list=[]
        for data in lines[data_start:]:
            datarow = data.split()
            alpha_list.append(float(datarow[0]))
            CL_list.append(float(datarow[1]))
            CD_list.append(float(datarow[2]))
            CM_list.append(float(datarow[4]))
    return np.array(alpha_list),np.array(CL_list),np.array(CD_list),np.array(CM_list)
def polarTable(results,polardir):
    table={}
    Polar= namedtuple("Polar", ["alpha","CL","CD","CM"])
    for Re, polarfiles in sorted(results.items()):
        parsed = _parseData(os.path.join(polardir,polarfiles))
        if parsed is not None:
            table[Re]=Polar(*parsed)
    return table
def _findOperatingPoint(table,ReTheoretical):
    # 0=alpha, 1=CL, 2=CD, 3=CM
    Re_keys=sorted(table.keys())
    ReTheoretical =float(np.clip(ReTheoretical, Re_keys[0], Re_keys[-1]))
    for xfoilRe in Re_keys:
        if ReTheoretical > xfoilRe:
            xlower = xfoilRe
        else:
            xupper = xfoilRe
            break
    if Re_keys[0] == xfoilRe or Re_keys[-1] == xfoilRe:
        interCL = table[xfoilRe].CL
        interCM = table[xfoilRe].CM
        interCD = table[xfoilRe].CD
    else:
        alpha_lower = table[xlower].alpha
        alpha_upper = table[xupper].alpha
        alpha_common = np.intersect1d(alpha_lower, alpha_upper)
        CL_lower = table[xlower].CL[np.isin(alpha_lower, alpha_common)]
        CL_upper = table[xupper].CL[np.isin(alpha_upper, alpha_common)]
        CD_lower = table[xlower].CD[np.isin(alpha_lower, alpha_common)]
        CD_upper = table[xupper].CD[np.isin(alpha_upper, alpha_common)]
        CM_lower = table[xlower].CM[np.isin(alpha_lower, alpha_common)]
        CM_upper = table[xupper].CM[np.isin(alpha_upper, alpha_common)]
        interCL = (CL_upper-CL_lower)/(xupper-xlower)*(ReTheoretical-xlower)+CL_lower
        interCM = (CM_upper-CM_lower)/(xupper-xlower)*(ReTheoretical-xlower)+CM_lower
        interCD = (CD_upper-CD_lower)/(xupper-xlower)*(ReTheoretical-xlower)+CD_lower
    interSink = interCL**1.5/interCD
    idx = np.argmax(interSink)
   # if interCL[idx] <= 0 or interCD[idx] <= 0:
    #    return None
    interMetrics = [ReTheoretical,interCL[idx],interCM[idx],interCD[idx],interSink[idx]]
 
    return interMetrics
def scoreCandidates(validbool,chord_grid,span_grid,Velocity,mass,Re,table):
    results= []
    for i in range(span_grid.shape[0]):
        for j in range(np.argmax(validbool),chord_grid.shape[1]):
            if not validbool[j]:
                continue
            testRe=Re[i,j]
            _repeat=True
            max_iter=500
            _iter=0
            while _repeat:
                interMetrics = _findOperatingPoint(table,testRe)
                if interMetrics is None:
                    break
                V = np.sqrt(2*mass[i,j]*9.81 / (1.225 * (chord_grid[i,j]/1000) * (span_grid[i,j]/1000) * interMetrics[1]))
                Re_new = V * (chord_grid[i,j]/1000)/1.48e-5
                if abs(Re_new - testRe)/testRe < .05:
                    _repeat = False
                    if interMetrics[2]>-.15:
                        results.append((interMetrics[4],chord_grid[i,j],span_grid[i,j],interMetrics[0],interMetrics[1],interMetrics[2],interMetrics[3]))
                elif _iter >= max_iter:
                    _repeat= False
                    
                else:
                    testRe = 0.7*testRe + .3*Re_new
                    _iter += 1
                    print("converge")
    
    #if not results:
     #   return None
    results.sort(reverse=True)
    return results[:5]
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
validbool,validpercent=mfCheck(cfg,thickness,xu)
filename,name=writeAirfoil(coords,M,P,T,polardir)
results,resweep = xfoilAnalysis(cfg,polardir,name)
table=polarTable(results,polardir)
results=scoreCandidates(validbool,chord_grid,span_grid,Velocity,mass,Re,table)
print(results)

#debugging 
np.savetxt('weight_grid.csv', mass, delimiter=',', fmt='%.5g')
plt.plot(coords[:,0], coords[:,1])
plt.axis('equal')
plt.show()