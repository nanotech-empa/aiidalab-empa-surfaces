# P01–P03: effective boundary model and honest NEB status

These are real GUI checks with independent read-only auditing. Run P01 without
science. Reuse the already planned S04/S06 outputs for P02/P03. Do not launch
eight calculations merely to cover all axis/solver controls.

## P01: classification, inactive edits and restart validity

1. Load the campaign CH3 fixture in a fresh SCF draft. Record its UUID, source
   PBC and 12 Å cell. Structural connectivity should be 0D while this chosen
   effective calculation is 3D XYZ/PERIODIC.
2. Change inactive NONE/MT, Y, XY and XYZ suggestions without applying them.
   Confirm the effective banner and working 12 Å model stay XYZ. Do not apply
   isolated preparation. Explicitly apply XYZ/PERIODIC and verify unchanged
   coordinates/cell. Check Apply versus Undo wording.
3. In disposable drafts inspect NONE, X/Y/Z, XY/XZ/YZ and XYZ and their offered
   solvers. Invalid combinations/overrides need a clear rejection before any
   submission. Restore the agreed XYZ model. An offered solver is not a native
   physical-validation PASS.
4. In the real linked NEB form, enter a valid NEB restart, then an existing
   non-NEB geometry workflow, then clear the field. The invalid restart must
   show its actual error and make the effective submission model unavailable.
   Valid/cleared input restores the correct inherited/fresh model. Do not set
   internal widget state or submit the invalid draft.
5. Check the oblique hBN cell summary reports lengths 10,10,15 Å and angles
   90,90,120 degrees, rather than labelling diagonal components as lengths.

PASS requires accurate labels and live transitions. FAIL includes a stale valid
banner, silent cell expansion, wrong lengths or a working/result boundary
mismatch. BLOCKED means a required route/fixture is unavailable.

## P02: partial results and legacy export, when retained evidence exists

Open the new S04 result through Search. Compare timeout, electronic convergence
and band convergence against every native BAND output. Partial finite profiles,
ordered images and accepted-step barrier history must remain viewable. Functional
plotting/export can pass while native convergence fails, provided status is honest.

If an independently verified retained legacy result has stored FFF structures
but native XYZ, open its View & export path. Require a clear stored-versus-actual
model notice and a labelled temporary calculation-aware ExtendedXYZ copy with
unchanged atoms/coordinates/cell, correct effective PBC and UUID annotations.
Audit that original nodes/files remain unchanged. If no such historical result
survives authorized cleanup, mark this legacy GUI subcase BLOCKED. Do not invent
a historical native calculation or claim old deleted records still reopen.

## P03: converged Low restart

Use S06's separately bounded Low restart when selected. Compare native accepted
steps, force/displacement criteria, electronic convergence, energy profile and
new result PBC with the real result page. Reopen a fresh page and repeat Show
and ExtendedXYZ download. Native input CELL/POISSON, working/result PBC and
export must agree about the actual 12 Å XYZ model. A corrected display never
retroactively changes an earlier calculation's scientific convergence.
