from ase import Atoms
import numpy as np


## ANCHORING ATOM  has coordinates (0,0,0)
## ANCHORING TO (-1,-1,-1)
## IF a displacemnet of 0 is assigned then the anchoring atom is removed

LIGANDS = {
    'CH3' : [
        (0,0,0), # ANCHORING C
        (0.23962342, -0.47699124, 0.78585262),
        (0.78584986, 0.23962732, -0.47698795),
        (-0.47699412, 0.78585121, 0.23962671 )
    ],
    'CH2' : [
        (0,0,0), # ANCHORING C
        (-0.39755349, 0.59174911, 0.62728004),
        (0.94520686, -0.04409933, -0.07963039)
    ],
    'OH' : [
        (0,0,0), # ANCHORING O
        (0.87535922, -0.3881659 ,  0.06790889)
    ]
    'NH2' : [
        (0,0,0), # ANCHORING N
        (0.7250916 , -0.56270993,  0.42151063),
        (-0.56261958,  0.4215284 ,  0.72515241)
    ]
}    

class LigandsAtoms():
    
    def __init__(self):
        

        super().__init__()
        
    def rotate_and_dr(mol=None, align_to=(0,0,1), dr=0.0, remove_anchoring=False):
        v = np.array(align_to)
        n = np.linalg.norm(v)
        
        ## BAD CASES
        if n == 0.0:
            v = np.array((1,1,1)) / np.sqrt(3)
        else:
            v = v / n
            
        if mol:
            if dr == 0.0:
                mol.rotate((1,1,1), v)
                ## REMOVE ANCHOR atom
                del mol[[atom.index for atom in mol if np.linalg.norm(atom.position) < 0.001]]
                return mol
            else:
                mol.rotate((1,1,1), v)
                mol.translate(dr*v)  ## DO NOT USE: return mol.translate() !!!!!!!!
                return mol
        
        
    def ligand(self, formula='CH3', align_to=(0,0,1), dr=0.0):
        mol = Atoms(formula, positions=LIGANDS[formula])
        return self.rotate_and_dr(mol=mol,align_to=align_to,dr=dr)
     
    
    