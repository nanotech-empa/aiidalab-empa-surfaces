# mc-empa-reactions

Generate replicas and do NEB calculations in nano@surfaces.

- [x] Generate the replicas
- [x] Search for replicas
- [x] Submit the NEB

## Replicas
In order to set up a nudged elastic band calculation a series of images can be generated starting from a structure in the database. For this purpose a collective variable is defined and its targets specified such that a series of geometry optimizations can be carried out.

CP2K supports a multitude of collective variables: ACID_HYDRONIUM_DISTANCE, ACID_HYDRONIUM_SHELL, ANGLE, ANGLE_PLANE_PLANE, BOND_ROTATION, COLVAR_FUNC_INFO, COMBINE_COLVAR, CONDITIONED_DISTANCE, COORDINATION, DISTANCE, DISTANCE_FROM_PATH, DISTANCE_FUNCTION, DISTANCE_POINT_PLANE, GYRATION_RADIUS, HBP, HYDRONIUM_DISTANCE, HYDRONIUM_SHELL, POPULATION, QPARM, REACTION_PATH, RING_PUCKERING, RMSD, TORSION, U, WC, XYZ_DIAG, XYZ_OUTERDIAG

Of those ANGLE_PLANE_PLANE and DISTANCE are implemented right now.

## NEB
The climbing image - nudged elastic band calculation is set up with a series of geometries (at least two) specified by their ids. The geometries are handed over to AiiDA as a FolderData object which is in turn transferred to the cluster.

## ToDo

- [ ] Parsing of the results fails right now.
