#!/bin/bash 

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

DFT_DIR="parent_calc_folder"
HARTREE="$DFT_DIR/aiida-HART-v_hartree-1_0.cube"

NX=$(sed '4q;d' $HARTREE | awk '{print $1;}')
NY=$(sed '5q;d' $HARTREE | awk '{print $1;}')
NZ=$(sed '6q;d' $HARTREE | awk '{print $1;}')

echo "gridN $NX $NY $NZ" >> params.ini

python $DIR/generateLJFF.py -i geom.xyz --data_format npy
python $DIR/generateElFF.py -i $HARTREE --data_format npy
python $DIR/relaxed_scan.py --data_format npy --disp --pos
python $DIR/plot_results.py --df --cbar --save_df --data_format npy
