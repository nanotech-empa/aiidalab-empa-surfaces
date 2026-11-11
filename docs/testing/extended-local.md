# Surfaces: editor, hybrid SCF, microscopy, PDOS and cube cases

Read [scope and resource limits](README.md), [core cases](localhost.md), and the
[report template](report-template.md). Use actual GUI controls on the aligned
candidate. Keep historical observations in their original local reports.
Perform [DOC01](help-review.md) throughout. XPS is excluded and gold is editor-only.

Scientific PASS requires the relevant native convergence and independent audit.
Functional GUI acceptance is recorded separately, including the user-accepted
partial NEB coverage and F01 export with its disclosed raw anomaly. Every case
requires bounded child jobs, preserved outputs and reopening without recomputation.
Record GUI, science, resources, persistence and duplicate protection separately.
Source findings below are not reproduced GUI failures. Never repeat Submit to
resolve an uncertain outcome; identify the first parent UUID. Never mutate an
ORCA or historical scientific node for a test.

## E01 — Create CH4, set its molecular cell, add Au(111)

**Needs:** S00–S01; an agreed new campaign prefix. **Submits:** zero native jobs;
storing a new structure creates AiiDA data. Run before scientific branches.

1. Open **SCF energy → From SMILES**, enter `C`, and use its structure-generation
   action. Check CH4: one C, four H, five atoms, a sensible tetrahedral geometry
   and no duplicate atoms. Record the generated atom order; it is not an optimized
   geometry or a reproducibility reference until its coordinates are frozen.
2. In **Edit structure → Cell transform**, set **a(Å), b(Å), c(Å)** to 12 and
   **α, β, γ** to 90. Leave **Scale atoms position** unchecked and press
   **Apply cell parameters**. Check that bond lengths did not scale. Center the
   intact molecule with the available atom-position editor if needed; the
   viewer's **Center molecule** camera action is not a coordinate transformation.
3. Inspect **Structure details**, positions, cell and periodicity. Do not apply
   the isolated-system suggestion. In the inspected electrostatics widget,
   **Apply selected settings** activates preparation; without that action the
   original cell and XYZ/PERIODIC model are retained. If isolated preparation
   was accidentally applied, use **Undo: restore original settings**, recheck
   the original 12 Å geometry, and record the correction before any submission.
4. **Store in AiiDA** as a new CH4 input; record UUID/PK and reopen through
   **AiiDA database**. Confirm five atoms and the exact cell. Capture geometry
   identity before adding a slab so later comparisons have an immutable source.
5. In **Build slab**, choose **Surface = Au(111)**, then explicitly set **nx=2,
   ny=2, nz=3**; selection of the molecule/surface can change suggested sizes.
   Press **Add slab** once. Check one CH4 above the top layer, no collision or
   duplicate methane, and a valid slab cell. Do not force this slab into the
   molecular 12 Å cube: the builder uses its own slab/vacuum geometry.
6. The inspected builder adds two bottom H per rectangular surface unit as
   well as two Au per layer. For this selection expect **24 Au + 8 bottom H +
   CH4 = 37 atoms**, not a pure 24-Au slab plus methane. Identify bottom H
   separately from methane H in selections and representations. Record this
   legacy construction convention; it is not an adsorption-reference choice.
7. In a disposable unsaved draft, change nx, rebuild and inspect that the
   remembered molecule appears once and layer counts change correctly. Restore
   nx=2, ny=2, nz=3 and rebuild. Record whether earlier stored CH4 geometry was
   preserved; a new combined structure must get a different UUID on storage.
8. Store only the agreed final new combined fixture. Reopen both new structures
   and compare their cells, formulas and atom coordinates. No CP2K job is needed.

**PASS:** correct molecular/slab construction and geometry persistence, clear
atom mapping and no scientific job. **FAIL:** atom loss/duplication, distorted
methane on cell edit, wrong layer count, altered earlier stored geometry or a
silent scientific submission. **BLOCKED:** importer/editor unavailable. No
electronic, Au adsorption or slab-convergence claim follows from E01.

## E02 — Auto/manual representations and explicit persistence

**Needs:** E01's new stored structures. **Submits:** no native jobs; explicit
representation storage changes extras on these new campaign nodes only.

1. Open the combined CH4/Au fixture. Use **AutoRep**; inspect its reported atom
   selections and whether molecule versus slab styles match those selections.
   The intended automatic pattern is molecular ball-and-stick plus the remaining
   surface in spacefill. Bottom H may affect classification: record actual
   groups rather than assuming every H belongs to methane.
2. In the viewer's **Appearance → Representations** controls, use **Add
   representation**. Select the methane indices and a supported ball-and-stick
   type, choose slab indices and spacefill, vary **Size**, **Color**, **Show**,
   then **Apply representations**. Delete a disposable row and restore the
   final distinct styles. Check geometry is unchanged and all intended atoms
   are represented; hiding atoms changes display, not the chemical structure.
3. Exercise **Camera type** (orthographic/perspective), background, axes and
   **Supercell** display, restoring 1×1×1. Count atoms in Structure details to
   confirm display replication has not created a scientific supercell.
4. Use **Store representations**. Record the node UUID and the explicit save
   feedback. Close only this campaign page and reopen the same structure via
   **AiiDA database**, then through the available **View & export** route.
   Verify stored selections, types, sizes and colors reappear. The inspected saved style
   does not encode **Show** visibility, camera or background; record their reset
   behavior as separate persistence limits, not as passed saved-state coverage.
5. Make a pending representation edit and press **Store representations** without
   a separate Apply. The inspected callback applies pending edits before saving;
   verify the reopened style matches the final controls. Then use **AutoRep**,
   explicitly store, reopen, and verify that automatic selections also persist.
6. Load the earlier bare-CH4 node and return to the slab node. Saved styles must
   not bleed between nodes, and loading stored representations must not silently
   overwrite them with AutoRep. A new unsaved structure should not inherit stale
   saved atom-index selections. The save button must not pretend an unsaved
   structure has persistent node storage.
7. Independent read-only audit: `aiidalab_viewer_representations` extras contain
   the selected state; geometry attributes/repository and provenance links are
   unchanged. Explicitly saved representation extras are expected mutations.
8. Record explicit same-source Load after unsaved geometry/style edits as a
   separate subcase. Official widgets-base 3.0.1 has the known pending PR832
   limitation. Do not install PR832 or use a hidden cache reset. If reproduced,
   record the affected subcase as FAIL with that upstream reference, then use an
   intervening node selection to continue independent supported cases.

**PASS:** auto/manual rendering and explicit persistence behave as described,
with unchanged science and no computation. **FAIL:** stale indices, misleading
save, lost styles, wrong-node extras or geometry changes. **BLOCKED:** viewer
or storage unavailable. Unsaved edits need not survive closing a page.

## H01 — CH3 doublet, PBE then PBE0 in a 12 Å cube

**Needs:** S01; agreed planar CH3 fixture, four atoms, no artificial isolated-cell
expansion; installed C/H hybrid basis, potential and auxiliary data. **Tier:** Small, with PBE0 at four MPI ranks. **Submits:** two
serial fixed-geometry SCFs, at most 20 minutes total; no geometry optimization.
Use **one node**, PBE **2 MPI tasks/node**, PBE0 **4 MPI tasks/node**, and
**one thread/task** in both. Four ranks allocate four CPU cores. The initial
2 GiB RAM estimate is superseded by observed usage and available Mac headroom.

1. Open **SCF energy**, import/store the agreed CH3 radical and verify C1H3,
   12×12×12 Å, angles 90°, intact centered molecule and XYZ/PERIODIC. If using
   `[CH3]` in **From SMILES**, verify the radical's formula and stored geometry
   rather than trusting SMILES parsing as spin configuration.
2. Set **UKS**, **Net charge: 0**, **Multiplicity: 2**; use **Add spin set**, record C's index in
   **Atoms indices:** and a suitable initial **Magnetization value:** (proposed
   1). Check electron counts against actual pseudopotential valences; C(q4)+3H(q1)
   gives seven electrons, four alpha and three beta. Do not infer these settings
   from the molecular name or structure classifier.
3. Select **Protocol: Standard**, **DFT settings → Functional: PBE**. Keep
   unnecessary unfolding/overlap/Bader and smearing off; record dispersion and
   every actual basis/potential choice. Set 1 node / 2 MPI tasks / 1 thread,
   600 seconds and one CP2K attempt; use the fresh restart policy,
   final-form capture and **Submit once**. Verify native SCF convergence, spin
   occupations and finite energy; preserve this PBE parent and source structure.
4. Open a fresh draft for the **same geometry** and spin. Select **Functional:
   PBE0**. Capture changed basis/potential fields: the inspected defaults use
   UZH primary/ADMM basis data and UZH potentials. Missing local data is BLOCKED,
   not permission to install or substitute arbitrary files.
5. Explicitly set **HF fraction: 0.25**, **HF cutoff radius: 5.0 Å** and **HF max
   memory [MiB/rank]: 256**; record **ADMM purification** (default NONE) and OT
   minimizer. The current form's 10 Å / 80,000 MiB-per-rank defaults are unsuitable
   for this budget. For a 12 Å periodic cube, 5 Å is below half the cell length;
   this is a proposed finite-cell recipe, not an exchange-cutoff convergence
   study. CP2K documents the periodic truncated-HFX condition in its
   [hybrid example](https://manual.cp2k.org/trunk/methods/post_hartree_fock/low-scaling.html).
   The HF memory field is per MPI rank and covers HFX, not the entire process;
   four ranks give a 1 GiB HFX sub-budget, not a total-job memory cap. Observe
   total RAM and host headroom separately. See
   [CP2K HF memory](https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/DFT/XC/HF/MEMORY.html).
6. In an unsubmitted draft, switch PBE→PBE0→PBE and back, checking that visible
   and effective functional-specific settings track the choice without hidden
   stale values. Restore the final explicit PBE0 recipe, spin, 12 Å cell and
   **1 node / 4 MPI tasks / 1 thread**, 600 seconds and one CP2K attempt.
   **Submit once** under a new description/UUID. Keep the parent-folder field
   empty and **Skip automatic WFN restart search** checked for both runs.
7. Independent audit: compare native cell, UKS/multiplicity, functional mixing,
   truncated interaction radius, ADMM/basis data, HF memory and native HFX terms
   to the visible form. Require native convergence without accepting a fallback
   to PBE or silent spin change. Check finite energy and electron/spin counts.
   If spin contamination diagnostics are absent, record that limitation.
8. Reopen both results and compare the recorded energies and settings. Do not
   require PBE0 energy to be lower than PBE, or interpret their difference as a
   pure functional effect if basis/potentials also changed. Apply S11 identity,
   storage and duplicate checks. No vibrational or isolated-limit validation.

**PASS:** correct open-shell hybrid calculation, effective bounded settings,
native convergence and durable outputs. **FAIL:** wrong spin/cell/functional,
silent default reversion, false convergence or budget breach. **BLOCKED:**
hybrid data/build capability, resource controls or native diagnostics unavailable.
Reducing to a closed-shell molecule would drop radical coverage; do not call it
an equivalent fallback. CH3 is already small; investigate recipe/readiness first.

## F01 — Benzene AFM without substrate

**Needs:** S01; neutral singlet planar benzene, 12 atoms, explicit 12 Å cube,
plane at z=3 Å and centered in x/y; local CP2K/PPAFM. **Tier:** Small.
The workflow is OT → diagonalization → AFM helper, three serial native jobs,
with 600 seconds requested per stage and a 30-minute case ceiling. Record actual
helper allocation and actual worker parallelism separately. A disclosed raw
AFM center-pixel anomaly from the old benzene case is retained in local evidence
for user review. Do not mask data or infer realism from the image.
Both SPM paths request at least 100 added MOs; audit the actual
diagonalization size, memory and feasibility before running. Debug is not a fix.
The inspected default C/H basis is TZV2P-MOLOPT-GTH; record the actual AO count
and ensure requested occupied+empty states fit it. Do not shrink the basis
without rechecking that condition. For the initial Mac case set **use ELPA**
off: the backend otherwise requests `AVX2_BLOCK2`, which must not be assumed
compatible with an ARM-native worker. Verify the available native diagonalizer.

1. Open **Scanning probe microscopy** / `submit_spm.ipynb`, select the new
   stored benzene. This form has no structure editors: fix its 12 Å cell in
   S01 beforehand. Do not activate isolated preparation. Verify full molecule
   is inside the cell and sampling will not wrap across z.
2. Set **SPM type: AFM**, the local **CP2K code:** and **PPAFM code:**, PBE,
   neutral closed shell and Standard. Proposed coarse settings: **Scan dx (Å):
   0.25**, **Scan z_min (Å): 3.5**, **Scan z_max (Å): 5.0**, **Amplitude (Å):
   0.5**, **Cantilever (f0): 22352.5**. Freeze the actual probe model and units.
   These are offsets from the rounded topmost atom, not absolute box coordinates;
   scan bounds must leave enough z samples for the oscillation convolution.
3. Inspect the effective PPAFM parameters: the current helper uses an O probe,
   fixed charge/tip/stiffness parameters and PBC=False. Record them; do not
   describe the model as a quantitatively calibrated experimental CO tip.
   Check x/y grid extent equals the cell and z range/amplitude remain inside
   the source-data extent. Retain the existing force-field/model limitations.
4. Set one node, selected 2/4/8 MPI tasks and one thread per task, 600 seconds,
   and the agreed fresh restart policy. The corrected helper receives the same
   requested allocation; PPAFM's wrapper itself remains serial. Distinguish
   allocated tasks from actual helper process count. Capture the final form,
   **Submit once**, and audit every child's effective options, UUID and native
   outcome. Worker refresh alone is not evidence of successful forwarding.
5. Open `view_afm.ipynb` from Search/result link. Inspect all generated height
   panels and color bars. **Make ZIP** packages one PNG and one numerical text
   plane per height. The actual F01 archive has five PNGs and five `.txt` files.
   Check image/height/axis correspondence and text headers (`tipz`, x/y limits).
   Text uses `%.2e`, three significant digits. Compare it to the corresponding
   NPZ plane within that rounding; do not demand byte-identical floats.
   Retain `df.npz` and native files for full-precision and complete-metadata audit.
6. Independently inspect retrieved `df.npz` (or the legacy `df.npy`/`df_vec.npy`
   pair only if actually produced): finite nonempty arrays, lvec/grid units,
   shape versus scan range, amplitude-adjusted plotted heights and Δf units.
   Require spatial variation above the molecule exceeding numerical noise,
   sensible symmetry for the agreed geometry, and consistent changes with
   height. Do not prescribe the sign of every pixel or judge correctness by
   visual attractiveness. Inspect interpolation artifacts separately.
7. Reopen the same UUID, compare arrays/checksums, and use S11. Retain the
   diagonalization eigenvalues/energy reference for selecting T01's window;
   reusing this evidence is not an additional calculation or a new SCF PASS.

**PASS:** correct converged source, numerical AFM output and meaningful GUI
height/image behavior within limits. **FAIL:** empty/nonfinite data, incorrect
units/offsets, false success or budget violation. **BLOCKED:** helper limit,
missing worker/data or unknown scan conventions. This is a molecule-without-substrate
model control; no substrate contrast, measured resolution or experimental
agreement is claimed.

## T01 — Benzene STM without substrate

**Needs:** the same benzene fixture and frontier eigenvalues/energy-reference
record from F01's converged diagonalization, or another explicitly identified
compatible local reference. If none exists, scope a separate reference stage
before submission; do not silently add one. **Tier:** Small. **Submits:** OT →
diagonalization → STM helper, at most 30 min. The approved two-file correction
replaces the old computer-default allocation and 36,000 s helper limit with the
GUI options. Use a fresh kernel after any approved source alignment. Confirm actual
STM descendant options: one node, selected 2/4/8 tasks,
one thread, 600 s. STM remains MPI-enabled.

Use intentional Standard PBE without D3 at fixed benzene geometry, explicit
12 Å XYZ/PERIODIC, neutral RKS, ELPA off, one node/four MPI/one thread/600 s.
Start with −3.1..−2.1 eV, dE/FWHM 0.1 eV, heights 3/4 Å, extrapolation plane
2 Å and an s tip. Verify that the actual spectrum contains frontier states in
that window. The STM output guard must reject failed helpers and missing,
unreadable or empty output while accepting valid zero-intensity maps and
optional missing constant-current crossings. Neither condition establishes
physical correctness. Test single-map and repeat exports in the same page.

1. Open **Scanning probe microscopy**, select benzene, **SPM type: STM**, local
   **CP2K code:** / **STM code:**, PBE/Standard and unchanged cell/spin. Confirm
   the z=3 Å plane and no crossing of the z cell boundary. Set **use ELPA** off
   for this first Mac case as in F01; record the actual native diagonalizer.
2. Set **Emin, Emax (eV):** to the frozen −3.1..−2.1 eV window containing an
   identified frontier level from the agreed spectrum. Record the app/worker
   energy zero and final numerical endpoints. The default ±2 eV may lie in
   benzene's gap; a blank map is not a useful positive science case.
3. Propose **dE (eV): 0.1**, **FWHMs (eV): 0.1**, **Extrap plane (ang): 2.0**,
   **Const. H (ang): 3.0 4.0**, **p tip ratio: 0.0**. Inspect the effective
   x/y resolution (currently 0.15 Å) and extrapolation extent (at least 5 Å).
   The worker centers its in-memory orbital geometry and uses top-atom-relative
   heights. Native cube-plane extraction has its own source-coordinate frame.
   Check the relative offsets and consistent translation rather than equating
   absolute helper z values with the stored z≈3 Å plane. The native/stored
   calculation remains a 12 Å XYZ/PERIODIC cell; do not enlarge it silently.
4. Treat **Const. cur. (isoval)** separately. The default 1e-7 must be inside
   the calculated intensity range before a constant-current surface is expected.
   Use valid constant-height maps as the mandatory first subcase; record absent
   constant-current crossings as unvalidated, not proof of a broken LDOS.
5. Review resources and actual added-MO count, capture final settings and
   **Submit once**. Require electronic convergence and helper
   output. Reusing a WFN as a guess does not remove child-stage accounting.
6. Open `view_stm.ipynb`, using **pk / Load pk** if needed. Change **energy range
   (eV)** and **energies (eV)** within calculated coverage and inspect available
   height/current views. Record exact plotted units and data identity. Do not
   use **clear tmp** as a cleanup test or delete original results.
7. Independent audit: requested spectral coverage actually contains states;
   nonempty finite grids, positive LDOS/intensity within numerical tolerance for
   the selected s-tip model, correct energy/height indexing and spatial nodal
   pattern consistent with the chosen frontier subspace. Degenerate benzene
   orbitals may rotate; compare the summed subspace when testing symmetry.
   Higher-plane intensity should broadly decay; do not require pixelwise
   monotonicity in nodal regions. A pretty or ugly image alone is no oracle.
8. Reopen without recomputation and apply S11. Record constant-height,
   constant-current and p-tip coverage separately; the last two are not passed
   automatically by this s-tip constant-height control.

**PASS:** finite physically interpretable spectral maps with correct indexing,
native convergence and bounded execution. **FAIL:** false nonempty/success claim,
wrong units/indexing, incompatible WFN or resource breach. **BLOCKED:** helper
limits, spectral reference or local code absent. This does not predict the
tunneling junction or adsorption-induced orbital changes of real benzene on a
substrate.

## D01 — Atomic PDOS of benzene on primitive 4×4 hBN

**Needs:** selected P1, 44 atoms, agreed fixed geometry, matching PBE data and
cell; S01. **Tier:** Medium (32 GiB external memory budget selected by the user;
timing remains proposed). There is no overall memory field in the CP2K form.
**Submits:** full-system OT →
diagonalization, serially, at most 40 min. A relaxed adsorption geometry can be
reused if already qualified, but is not required for a fixed-geometry PDOS
integration control. Record that distinction.

1. Open **Projected density of states** / `submit_pdos.ipynb`, select the new structure and check one
   benzene (12 atoms) above one primitive-monolayer 4×4 sheet (32 atoms), neutral
   singlet, actual cell, periodicity and agreed numerical recipe.
2. Keep **Do overlap calculation:** unchecked. Inspect the automatically
   detected molecule and full-system fragments. Use **Add projection** and
   **Atoms selection:** / **Label:** to define disjoint `benzene` and `sheet`
   projections covering all 44 atoms; add a full `all` selection if exposed for
   the comparison. Record actual one-based indices; do not assume molecular
   recognition selected all and only benzene.
3. Choose a modest energy window around occupied/frontier states and inspect
   effective added-MO count. The backend imposes a minimum/size-dependent count;
   no fabricated GUI control may override it. For ordinary valences the full
   system has 158 electrons / 79 doubly occupied bands; verify actual potentials
   and occupations. Missing empty states must not be invented by broadening.
4. Set all applicable resources to the agreed Medium values, capture the form
   with **use ELPA** off for the first Mac case and **Submit once**. Check only
   the full-system electronic branch ran; a
   molecule-overlap job is not expected with its checkbox off.
5. Open `view_pdos.ipynb`. Inspect **Broadening fwhm**, **Energy range**,
   **Manual E_ref** / **Apply E_ref**, **Plot**, and **Cumulative plot**. Select
   the available benzene/sheet curves with equal scale/normalization. Change
   broadening (proposed 0.1 then 0.2 eV) and energy zero, then restore. Peaks
   should broaden/shift consistently without changing underlying eigenvalues.
6. Export **CSV** and **SVG** through **Export in:**. Check headers, energy
   convention, finite data, selected curves and plot correspondence. The CSV
   contains plotted/broadened data with finite printed precision; it is not
   necessarily native per-orbital PDOS. Preserve raw retrieved PDOS separately.
7. Independent audit: atom groups complete/disjoint, native PDOS convention and
   units established, sums of benzene+sheet agree with the same all-atom
   projection at each recorded orbital and after equal broadening. Only use a
   unit-weight sum rule if the actual native projection convention warrants it;
   distinguish per-spin DOS from occupation-weighted electron counts. Test
   raw components and integrated broadened areas over a sufficiently wide
   energy interval (proposed relative tolerance 1e-3, finalized in advance).
8. Reopen with identical source/result identities and apply S11. Do not interpret
   PBE gaps as quasiparticle gaps, peak mixing as proof of charge transfer, or
   finite Γ-only supercell spectra as a converged Brillouin-zone DOS.

**PASS:** converged source and correct projected spectra, controls and exports
within budget. **FAIL:** wrong atom selection, missing/unreported states, bad
normalization/units or false convergence. **BLOCKED:** fixture, codes, budgets or
native reference data absent. Smaller fallback: bare benzene, which loses
substrate-projection coverage.

**Optional D02 — molecular-orbital overlap, not selected:** checking **Do overlap
calculation** launches full-system and molecular diagonalizations concurrently,
then an overlap helper with independent site-sized limits. This is a distinct
feature from atomic PDOS. Execution stays BLOCKED under the one-job rule until
serialization and explicit helper bounds exist. A checkbox inspection cannot
pass molecular-overlap science.

## C01 — Adsorption charge-density difference cube and GUI

**Needs:** S07–S08 with the **Cubegen code:** selected on the original valid form,
local compatible cubehandler, serialized fragment jobs and retained **frozen
SCF** density cubes. A3 (benzene/4×4 hBN) is the proposed shared fixture; A1/A2
are explicit alternatives. **Submits:** the cubehandler is the sixth job of that
adsorption workflow, after component stages; one rank / 600 seconds. This case
does not add a second helper job or resubmit the parent merely to view a cube.
Its time/storage are included in S08+C01's agreed tier cap.

1. In S07 choose the local **Cubegen code:**, verify the `all`, `benzene`, `sheet`
   (or selected alternative) partition and neutral charge balance. Capture
   this together with each fragment's resources before S08 submission.
2. After S08, record the child cubehandler UUID/PK and output files. The
   inspected backend uses `all` with weight +1 and frozen fragments with −1,
   writes **ChargeDiff.cube**, then creates reduced `out_cubes` previews.
   A successful energy page without this output cannot pass the selected
   adsorption-plus-density scope.
3. Open the available cube link or the maintained
   `handle_cubes.ipynb?pk=<cubehandler-CalcJob-PK>` route. The PK is the **child
   cubehandler**, not the parent adsorption WorkChain. Record a missing natural
   Search/result link as a navigation gap rather than calling a backend API.
4. Select the charge-difference item in **Cube files:**. In the viewer choose
   equal-magnitude positive and negative **Isovalue** entries with distinct
   colors, then **Update isosurfaces**. Inspect molecule/sheet alignment and
   signed lobes. Adjust **Camera orientation** and isovalue, then restore the
   report's agreed display settings. Verify density sign/units before labelling
   colors as electron accumulation or depletion.
5. Enable **Show slicing plane in 3D** / **Show 2D plot**, place a plane through
   molecule and sheet with **Center (A)**, **Normal**, **Offset (A)** and **Apply**.
   Use consistent symmetric **vmin/vmax** with **Auto color range** off to compare
   signs; vary plane/offset/resolution and verify 2D/3D alignment. A plane at only
   the molecular height does not alone test the interface density.
6. Independent audit of the **full-resolution** source/output: identical cell,
   origin, voxel vectors, grid dimensions, units, cutoffs and frozen coordinates;
   correct component SCF UUIDs and weights; finite arrays and nonzero signed
   redistribution above output precision. Reconstruct
   `Δρ = ρall − ρfragment1 − ρfragment2` pointwise within file precision.
   Never subtract densities of the independently relaxed fragments.
7. For a neutral, charge-balanced partition, integrate Δρ using the voxel volume
   and actual density/unit convention: expect approximately zero net electron
   change (proposed full-grid absolute tolerance 1e-3 electron, to be finalized
   before execution against the native grid accuracy). Independently verify each
   component integral against its valence electron count. Spatial redistribution
   can be finite while the total integral is zero. Do not impose a guessed
   transferred charge or binding sign on an unrelaxed benzene geometry.
8. Compare reduced GUI previews qualitatively with the full grid. The helper
   uses `low_precision=True`; reduced previews need not satisfy the full-grid
   conservation tolerance. Do not accept a low-resolution image as its proof.
   Preserve full cubes and record sizes/checksums without bulk retrieval.
9. Close/reopen the cube route and compare UUID/files, geometry, signed display
   capability and slice data without new native jobs. Use [C02](cube-viewer.md)
   for shared preview/slice/full-grid rendering once that adapter is included in
   the qualified candidate. If absent, record that path as unsupported or blocked.
   An interactive preview is not a high-resolution-render PASS.

**PASS:** correct signed density difference, full-grid arithmetic/conservation,
usable aligned GUI views, durable outputs and bounded resources. **FAIL:** wrong
reference geometries, mismatched grids, total density shown as Δρ, missing signs,
bad integral or rerun on viewing. **BLOCKED:** missing full cubes/helper/code or
independent diagnostics. Energy arithmetic can pass separately, but complete
adsorption coverage cannot pass until this selected density case also passes.
