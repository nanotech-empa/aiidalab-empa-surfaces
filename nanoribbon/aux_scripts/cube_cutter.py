from glob import glob
import numpy as np
import gzip

for fn in glob("*.cube"):
    # parse
    lines = open(fn).readlines()
    header = np.fromstring("".join(lines[2:6]), sep=' ').reshape(4,4)
    natoms, nx, ny, nz = header[:,0].astype(int)
    cube = np.fromstring("".join(lines[natoms+6:]), sep=' ').reshape(nx, ny, nz)

    # plan
    dz = header[3,3]
    angstrom = int(1.88972 / dz)
    z0 = nz/2 + 1*angstrom # start one angstrom above surface
    z1 = z0   + 3*angstrom # take three layers at one angstrom distance
    zcuts = range(z0, z1+1, angstrom)

    # output
    ## change offset header
    lines[2] = "%5.d 0.0 0.0 %f\n"%(natoms,  z0*dz)
    ## change shape header
    lines[5] = "%6.d 0.0 0.0 %f\n"%(len(zcuts), angstrom*dz)
    with gzip.open(fn+".gz", "w") as f:
        f.write("".join(lines[:natoms+6])) # write header
        np.savetxt(f, cube[:,:,zcuts].reshape(-1, len(zcuts)), fmt="%.5e")