import numpy as np
from parsing_gmx import parse_itp, parse_itp_charge, return_C, return_bjk, get_bjks

def get_nresids(top_file):
    with open(top_file) as ff:
        lines = ff.readlines()
        lines = ''.join(lines)
        start = lines.index('[ molecules ]\n')
        lines = lines[start:].strip().split('\n')

    out = [i.strip().split() for i in lines[2:]]
    out = [ (i, int(j)) for i, j in out]
    resids, nums = zip(*out)

    resids=np.array(resids)
    nums = np.array(nums)
    _, idx = np.unique(resids, return_index= True)
    nums = np.array([nums[resids==rr].sum() for rr in  resids[sorted(idx)]])
    return resids[sorted(idx)] , nums

def total_charge_vector(resids, nums, exclude, MOL):
    nat = np.sum(nums)
    total_charges =np.array([])

    for i, j in enumerate(resids):

        if j =='TIP3':
            charges = np.array([0.417, 0.417])

            if 'TIP3' in exclude:
                charges = charges * 0.

        elif j in ['POT', 'CLA', 'SOD']:
            charges = np.array([])

        else:
            charges = get_bjks(j, MOL)

            if j in exclude:
                charges = charges *0.

        if charges.size >0:

            vector = np.tile(charges, nums[i]).reshape(-1,)
            total_charges = np.concatenate([total_charges.reshape(-1,), vector])

    return total_charges

def total_bond_vector(resids, nums, MOL):

    total_bonds=np.array([], dtype=(int, int))
    bid=0

    for i, j in enumerate(resids):

        if j =='TIP3':

            bonds = np.array([[0, 1], [0 ,2]])
            nbonds = 2
            nats = 3

        elif j in ['POT', 'CLA', 'SOD']:

            bonds = np.array([], dtype = int)
            nbonds =0
            nats = 1

        else:
            #print(j)
            #bonds = parse_itp(MOL+'/toppar/' + j +'.itp' )
            bonds = parse_itp(MOL, j )
            nbonds = bonds.shape[0]
            nats = bonds.reshape(-1,1).max()+1

        for bb in np.arange(nums[i]):

            if bonds.size>0:
                total_bonds = np.concatenate([total_bonds.reshape(-1, 2) , bonds + bid] )
            bid = nats + bid

    return total_bonds

def dipole_locs_eff(bonds , xyzs, box, MOL):

    # Apply PBCs
    #if any(abs(dr_temp)- 0.6):
    #        dr_temp = np.array([ min([ddr, -np.sign(ddr)* box[i] + ddr], key = abs) for i, ddr in enumerate(dr_temp)])
    rr = [(xyzs[bb[0]], xyzs[bb[1]]) for bb in bonds ]
    rmp = np.array([(i[1]+i[0])*0.5 for i in rr])
    rvec = np.array([i[0]-i[1] for i in rr])

    return rmp.reshape(-1, 3) , rvec.reshape(-1, 3)

def get_dipoles(frame_xyz, bonds, charges, box, MOL):

    rmp, dr = dipole_locs_eff(bonds, frame_xyz, box, MOL)

    moments = np.multiply(charges.reshape(-1,1) , dr)

    return moments, rmp
