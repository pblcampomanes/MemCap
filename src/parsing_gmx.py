import numpy as np

## Functions to read gromacs topology files and compupte bond and charge vectors
## Assumes that the topology is in ./toppar/ folder, can be modified by chaning the TOPPAR below

TOPPAR='/toppar/'

def parse_itp(MOL, j):

    filename=MOL+TOPPAR + j +'.itp'

    with open(filename) as ff:

        lines = ff.readlines()
        lines=''.join(lines)
        start = lines.find('[ bonds ]')
        l1 = len('[ bonds ]')

        end = lines.find('[ pairs ]')
        end1 = lines.find('[ angles ]')
        end2 = lines.find('[ dihedrals ]')

        end = min([end, end1, end2])

        bonds = lines[start+l1+1:end].strip().split('\n')[1:]

        tops = np.array([k for bb in bonds for k in bb.strip().split()], dtype=int).reshape(-1, 3)
    return tops[:,:2] - 1

def parse_itp_charge(MOL, j ):
    filename=MOL+TOPPAR + j
    with open(filename) as ff:
        lines = ff.readlines()
        lines=''.join(lines)
        end = lines.find('[ bonds ]')
        end1 = lines.find('[ angles ]')
        end2 = lines.find('[ dihedrals ]')

        end = min([end, end1, end2])
        l1 = len('[ atoms ]')
        start = lines.find('[ atoms ]')

        bonds = lines[start+l1+1:end].strip().split('\n')[1:]
        #print(bonds) 
        charges = np.array([float(''.join(bb.strip()).split()[6]) for bb in bonds if bb[0] not in [';']]).reshape(-1,1)
        #tops = np.array([int(k) for bb in bonds for k in bb.strip().split()]).reshape(-1, 3)
    return charges.reshape(-1,)

def return_C(mol_file, MOL):
    charges= parse_itp_charge(MOL, mol_file+'.itp')
    bonds = parse_itp(MOL, mol_file)

    nb=bonds.shape[0]
    nat = charges.shape[0]
    #print(nat, nb)
    # Build Cijk Matrix
    Cijk = np.zeros([nat, nb])
    for i in np.arange(nat):
        for j in np.arange(nb):
"parsing_gmx.py" [dos] 90L, 2705C                                                                                    52,0-1        Top
