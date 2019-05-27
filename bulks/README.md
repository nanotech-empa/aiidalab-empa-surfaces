## Materials Cloud AiiDA Lab Tools for Scanning Probe Microscopy simulations
App to perform bulk cell optimization with CP2K
method used:
 &CELL_OPT
   OPTIMIZER BFGS
   TYPE DIRECT_CELL_OPT
   KEEP_SYMMETRY
   EXTERNAL_PRESSURE 0
   MAX_FORCE 0.0001
   MAX_ITER 500
 &END
