# Poisson solver and reversible isolation settings

The shared DFT forms show the detected dimensionality, editable periodic axes
and Poisson solver, additional POISSON settings as JSON, and a cell-size preview.
Before **Apply selected settings** is clicked, submission uses **XYZ** periodicity,
the **PERIODIC** solver and the original cell and atomic positions. Editing the
dropdowns or JSON field previews a proposal; it does not yet change the submission.
Clicking the button applies the selected solver, axes and prepared cell. The
**Currently used for submission** panel states which settings will be used.
While applied, further edits update those settings. Clicking **Undo: restore
original settings** restores the original cell, atomic positions and PBC and
returns to XYZ/PERIODIC electrostatics.
Loading or editing another structure resets activation and its original-cell
snapshot. Structure analysis itself no longer changes the input cell or PBC.

MT is available for 0D molecules and 2D slabs, **not for 1D wires**. For 1D,
choose its periodic cell axis and an explicitly supported solver. ANALYTIC is
available with a slow-convergence warning. The guide links to CP2K's documented
solver prerequisites; alternative methods require their own scientific checks.

The expandable **Examples: molecule, slab and wire** guide shows:

| System | Periodic axes | Solver | Cell preparation |
| --- | --- | --- | --- |
| Isolated molecule | NONE | MT | Space around the molecule in all directions |
| Slab in the xy plane | XY | MT | Retain the in-plane vectors; prepare vacuum along z |
| Wire along z | Z | ANALYTIC | Retain the wire vector; prepare space in x and y |

Choose the corresponding axes for another orientation. The JSON field is optional:
leave `{}` to use CP2K's defaults with the solver and axes chosen in the dropdowns.
For MT only, `{"MT": {"ALPHA": 7.0}}` explicitly retains its documented default
ALPHA value; this is a syntax example, not a recommendation to tune ALPHA.
The basic ANALYTIC wire example uses `{}`. Set periodicity and solver using the
dropdowns rather than placing `PERIODIC` or `POISSON_SOLVER` inside this field.

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
