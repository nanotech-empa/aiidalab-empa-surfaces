"""Preserve the source provenance of explicitly prepared NEB replica cells."""

import numpy as np
from aiida import engine, orm


@engine.calcfunction
def apply_cell_plan(structure, plan):
    atoms = structure.get_ase()
    request = plan.get_dict()
    atoms.set_cell(request["cell"], scale_atoms=False)
    atoms.positions += np.asarray(request["translation"])
    atoms.pbc = request["pbc"]
    return orm.StructureData(ase=atoms)
