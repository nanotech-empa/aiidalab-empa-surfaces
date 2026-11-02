# Poisson solver and reversible isolation settings

The shared DFT forms show the detected dimensionality, editable periodic axes
and Poisson solver, additional POISSON settings as JSON, and a cell-size preview.
The control is **off by default**. Turning it on applies the reviewed cell and
settings; turning it off restores the original cell, atomic positions and PBC.
Loading or editing another structure resets activation and its original-cell
snapshot. Structure analysis itself no longer changes the input cell or PBC.

MT is available for 0D molecules and 2D slabs, **not for 1D wires**. For 1D,
choose its periodic cell axis and an explicitly supported solver. ANALYTIC is
available with a slow-convergence warning. The guide links to CP2K's documented
solver prerequisites; alternative methods require their own scientific checks.

MT preparation retains larger cells and grows only isolated cell vectors to
twice the atomic span plus 15 Å. The preview reports old/new lengths and volume
change. It preserves cell angles and periodic vectors and uses a rigid
translation, without scaling interatomic distances. The padding is a starting
rule: electronic-density and vacuum convergence remain necessary.

Geometry optimisation, SCF, phonons, PDOS, SPM, adsorption-energy and replica
forms forward these settings. A new NEB band uses one cell and translation for
all supplied input replicas, with preparation provenance linked to each source.
Editing the band resets activation so its common cell can be reviewed again.
An existing NEB restart retains its original setup; change the cell through a
new band rather than silently changing a restart's wavefunction interpretation.

This GUI change requires its paired backend change (`electrostatics.py` and
explicit Poisson-setting forwarding). It does not modify live installations.
The dedicated GW/image-charge forms use a separate physical cell construction
and are outside this shared DFT control.
