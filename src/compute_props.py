import numpy as np
import pickle

from polarizations import compute_Pz, compute_Pz_scaled, Pzz_parallel, parallel_loop
from multiprocessing import Pool
from scipy.interpolate import BSpline, UnivariateSpline
from scipy.signal import argrelextrema
from sklearn.linear_model import RidgeCV

EPSILON_0    = 8.854187817E-12 #* u.farad / u.meter
GAS_CONSTANT = 8.3144621 #* u.joule / u.kelvin / u.mole
CONV_FACTOR  = 1.602**2 * 1E06/ (8.854 * 310 * 1.38064)

conversion = 1.602**2 /(1.38 * 310 * 8.854) * 1E06

#def Capacitance(eps_z_inv, dz=0.1):

#    z = np.linspace(0, nbin*dz, nbin)

#    return 1/np.trapz(eps_z_inv, z)*1E-09/EPSILON_0* 1e06*1e-04 # muF/cm2

def COVAR(Pz, Mz, alpha =2):

    return np.cov(np.vstack([Pz[:, :, alpha].T, Mz[:, alpha].T.reshape(1, -1)]))[-1][:-1]

def VAR(Mz, alpha =2):

    return np.var(Mz[:, alpha])

def eps_zz(Pz , Mz, V, alpha=2):

    Hzz = np.var(Mz[:, alpha])*conversion/V
    hzz = conversion*COVAR(Pz, Mz, alpha=2)

    return hzz, Hzz, hzz/(1. + Hzz)

def Potential(Pz, z):

    #nbin = Pz.shape[0]
    #z = np.linspace(0, nbin*dz, nbin)

    #return np.array([np.trapz(Pz[0:i], z[0:i])*1E-10*1.602*4*np.pi*-1 for i,_ in enumerate(z)])
    return np.array([np.trapz(Pz[0:i], z[0:i]) for i,_ in enumerate(z)])

def Pz2C(Pz, Mz, box, nbins):

    Vol = np.product(box.mean(axis =0))
    _, _, ezz = eps_zz(Pz, Mz, Vol)
    exx = 1 +( CONV_FACTOR * COVAR(Pz, Mz, 0)+ CONV_FACTOR * COVAR(Pz, Mz, 1) )/2
    zz  = np.linspace(0., box[:,2].mean(), nbins)

    return zz, exx, ezz

def slab_model(dmem, dint, eps_int, eps_bilayer):
    '''Two slab model to compute capacitance given, epsilon values and thicknesses of each layer'''
    dmem = dmem
    dint = dint/2
    return 1e09*EPSILON_0/(((dmem-2*dint)/eps_bilayer)+ 2*dint/eps_int)*1e02

def sigmoid(x, t, d):
    '''Define a custom sigmoid function'''
    return 1/(1+ np.exp((x-t)*10/d))

def sigmoid_model(dmem, dint, eps_int, eps_bilayer, wd = 0.001):
    '''Define a macroscopic sigmoid_model of dielectric to compute capacitance'''
    dmem = dmem
    dint = dint/2

    eps = lambda x: eps_bilayer + (eps_int -eps_bilayer)* sigmoid(x, dint, wd)
    z = np.linspace(0., dmem/2, 250)

    return 0.5*1E09*EPSILON_0/np.trapz(1/eps(z), z) *1e02

def cumu_c(ezz, zz):
    return np.array([1E09*EPSILON_0/np.trapz(1/ezz[0:i],zz[0:i])*1e02 for i,_ in enumerate(zz)])

def get_c(*args):

    if(len(args)==5):
        dmem, dint, eps_bilayer, eps_int, model = args
        model_c = model(dmem, dint, eps_int, eps_bilayer)
    else:
        dmem, dint, eps_bilayer, eps_int, model, wd = args
        model_c = model(dmem, dint, eps_int, eps_bilayer, wd = wd)

    return model_c

def findclosest(a, x):
    ''' Find closest value to scalar a in array x '''
    index = (np.abs(x-a)).argmin()
    val = x[index]
    return val, index

def Bmatspl(x, knots=None, q=3):
    '''
    Builds the B-matrix (design matrix) with b-splines as basis elements
    Input:
    ------
    x -> values at which the b-spline functions are evaluated
    knots -> 1-D numpy array with the internal knots for total spline
    q -> degree of every b-spline in the linear combination

    Output:
    -------
    splreg -> Splines (linear expansion)
    '''

    if knots is None:
        # default definition of internal knots (equispaced along the data set interval)
        knots = np.linspace(x[0], x[-1], len(x)//3, endpoint=True)[1:-1]

    nknots = len(knots)                 # number of internal knots of total spline
    nbsplines = nknots + q + 1          # number of b-splines in the basis set

    # add knots at external points of the interval
    spknots = np.insert(knots, 0, [x[0], x[0], x[0], x[0]])
    spknots = np.append(spknots, [x[-1]+1e-10, x[-1]+1e-10, x[-1]+1e-10, x[-1]+1e-10])

    a = np.ones(nbsplines)             # coefficients of the b-splines in the linear expansion

    # build the B matrix. Their elements are the values of every b-spline at every data point
    ### B is a 2-D numpy array: [(number of points in x) x (number of bsplines)]
    spl = BSpline(t=spknots, c=a, k=q)
    B = spl.basis_element(spknots[0:q+2], extrapolate=None)(x).reshape(-1, 1)
    for i in range(1, nbsplines):
        Bnew = spl.basis_element(spknots[i:i+q+2], extrapolate=None)(x).reshape(-1, 1)
        B = np.hstack((B, Bnew))
    B = np.nan_to_num(B, copy=False)

    return B

def thickness_finder(ezz, zz, nbins, sym=True, nbspl=30):
    if sym == True:    # if membrane is symmetric
        zzlow = zz[:len(zz)//2]
        # Reverse array
        zzupp = zz[::-1]
        zzupp = zzupp[:len(zz)//2]

        ezzlow = ezz[:len(ezz)//2]
        # Reverse array
        ezzupp = ezz[::-1]
        ezzupp = ezzupp[:len(ezz)//2]

        ezzmeanlow = np.mean((ezzlow, ezzupp), axis=0)
        ezzmeanupp = ezzmeanlow[::-1]

        # Concatenate
        if (nbins % 2) == 0:
            ezzmean = np.hstack((ezzmeanlow, ezzmeanupp))
        else:
            zzcentre = zz[len(zz)//2]
            ezzcentre = ezz[len(zz)//2]
            ezz = np.hstack((ezzmeanlow, ezzcentre, ezzmeanupp))

    zz=zz[2:-2]
    ezz = ezz[2:-2]
    knots = np.linspace(zz[0], zz[-1], nbspl-2, endpoint=True)[1:-1]
    # Build design matrix and smooth ezz by means of ridge regression
    Bmat = Bmatspl(zz, knots=knots, q=3)
    reg = RidgeCV(alphas=(np.logspace(-15, 15, 300001)), store_cv_values=True, normalize=True)
    reg.fit(Bmat, ezz)
    ezzsmooth = np.dot(Bmat, reg.coef_)
    # Interpolate function in finer grid
    dz = 1e-3
    zgrid = np.arange(zz[0], zz[-1], dz)
    ezzsmoothspl = UnivariateSpline(zz, ezzsmooth, k=3 , s=0)
    smooth = ezzsmoothspl.__call__(zgrid, nu=0)
    # Calculate derivatives
    dsmooth = np.gradient(smooth)
    d2smooth = np.gradient(dsmooth)
    # Find critical points 
    ### First derivative
    idx_min = argrelextrema(dsmooth, np.less_equal, order=len(zgrid)//10)
    idx_max = argrelextrema(dsmooth, np.greater_equal, order=len(zgrid)//10)

    idx11, val11 = findclosest(len(zgrid)//2, idx_min[0])
    idx12, val12 = findclosest(len(zgrid)//2, idx_max[0])

    ### Second derivative
    idx_min = argrelextrema(d2smooth, np.less_equal, order=len(zgrid)//10)
    idx_max = argrelextrema(d2smooth, np.greater_equal, order=len(zgrid)//10)

    idx21, val21 = findclosest(idx11, idx_max[0][idx_max[0]>idx11])
    idx22, val22 = findclosest(idx12, idx_max[0][idx_max[0]<idx12])

    #db= abs(zgrid[idx12] - zgrid[idx11])
    dh = abs(zgrid[idx22] - zgrid[idx21])

    return dh
