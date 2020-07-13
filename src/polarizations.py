import numpy as np
from multiprocessing import Pool

from dipole_moment import get_nresids, total_charge_vector, total_bond_vector, dipole_locs_eff, get_dipoles

def compute_Pz(moments, zcoords, nbins, Vinv, dz=0.2):
    '''Function computes profile given moments'''    # moment and coords are natomx1 vectors 
    # function returns nbinx1 vector
    dip_profile = np.zeros(nbins)
#    den_profile = np.zeros(nbins) 
    #dip_mean_profile = np.zeros(nbins) 

    for k in np.arange(nbins):

        zi = 0.+ (k) * dz
        zf = zi+dz
        slab = moments[np.where((zcoords>=zi)&(zcoords<zf))[0]]
        dip_profile[k] = np.sum(slab)
#        den_profile[k] = np.size(slab)
        #dip_mean_profile[k] = np.mean(slab) 

    return Vinv*dip_profile*ZL, dip_profile.sum()*ZL #, den_profile*Vinv

def compute_Pz_scaled(moments, zcoords, nbins, Axy, ZL):
    '''Function computes profile given moments'''

    ## Scale the coordinates for uniform binning  
    dz = 1./nbins  ;
    zcoords= zcoords/ZL

    Vinv =1/(Axy*dz*ZL)

    dip_profile = np.zeros(nbins)

    for k in np.arange(nbins):

        ## Compute slab
        zi = 0.+ (k) * dz
        zf = zi+dz
        slab = moments[np.where((zcoords>=zi)&(zcoords<zf))[0]]

        ## Add moments in the slab
        dip_profile[k] = np.sum(slab)

    return Vinv*dip_profile*ZL, dip_profile.sum()*ZL

def Pzz_parallel(traj, exclude, top_file, MOL, nbins, n_proc=4):
    '''Function that computes density and dipole moment profile for all the frames'''

    xyzs = traj.xyz
    nf = xyzs.shape[0]
    box = traj.unitcell_lengths
    #nbins = 60 
    dip_profile = np.zeros([nf, 3, nbins])

    ## Compute Basic stuff
    resids, nums = get_nresids(MOL +'/' +top_file)
    bonds = total_bond_vector(resids, nums, MOL)
    charges = total_charge_vector(resids, nums, exclude, MOL)

    ## Pooling 
    mypool = Pool(processes = n_proc)
    myargs = ((xyzs[frame], bonds, charges, box[frame], MOL, nbins) for frame in np.arange(nf))
    kout = mypool.map(parallel_loop, myargs)
    mypool.close()

    ## convert pool output to normal form
    dip_profile = np.array([kout_it[0] for kout_it in kout ])
    Mzz = np.array([kout_it[1] for kout_it in kout ])

    Zmean_total = box[:, 2].mean()
    Vmean = traj.unitcell_volumes.mean()
    return  dip_profile/Zmean_total, Mzz/Zmean_total # EDIT HERE

def parallel_loop(myargs):
    ''' Function that computes the moment profiles for a given frame --- in an easily parallellizable form'''
    ## Parse Args
    xyzs, bonds, charges, box, MOL , nbins = myargs
    Axy = box[0] * box[1]

    ## Initialize Matrices
    dip_profile_loc = np.zeros([3, nbins])
    Mzz_loc = np.zeros([3])

    ## Get moments and locations
    moments, rmp = get_dipoles(xyzs, bonds, charges, box, MOL)

    for i in np.arange(3):
        dip_profile_loc[i], Mzz_loc[i] = compute_Pz_scaled(moments[:,i], rmp[:, 2], nbins, Axy, box[2])

    return dip_profile_loc.transpose(), Mzz_loc
