# Surfaces: core local GUI cases

Read [scope, systems and limits](README.md), [added scientific/editor cases](extended-local.md)
and the [report template](report-template.md). All submissions are actual GUI
operations by the primary. Source inspection and synthetic tests do not pass
these cases. Record current source/runtime fingerprints before each run.

Review means inspecting the existing form, not an invented wizard. Standard and
Low accuracy are in scope. Structural connectivity and effective boundary
conditions must remain distinct. Molecular CP2K cases use explicit 12 Å XYZ
cells without isolated preparation. Functional NEB plots can pass separately
from native convergence if their partial status is honest and data agree.
Perform [DOC01](help-review.md) alongside every case.

## S00 — Launch and identify the local execution context

**Needs:** selected campaign cases; existing configured instance.
**Submits:** nothing.

1. Open the on-surface chemistry app from the existing AiiDAlab launch route.
   Record the actual app title/URL and the **Surfaces workflows** launcher.
2. Locate **Geometry optimization**, **SCF energy**, **Adsorption energy**,
   **Replica chain**, **Nudged elastic band**, **Scanning probe microscopy**,
   **Projected density of states** and **Search**. Open each relevant
   form without pressing Submit. Leave unrelated cards unopened.
3. Inspect **CP2K code:** and its computer. Record the actual label and UUID
   when exposed; have the maintainer resolve missing identity separately.
4. Verify normal page scrolling, usable forms, atom viewer and status output.
   Record nested output scrolling as a usability finding, not a reason to redesign.
5. Confirm the local native-job slot is idle. The Mac is available overnight;
   keep one native scientific job at a time and preserve unrelated research and the current protected-node manifest. Do not stop or change ORCA, workers, containers or services.

**PASS:** correct notebook routes, usable controls and identifiable local context;
opening pages creates no scientific jobs. **FAIL:** unexpected submission,
broken navigation or wrong app route. **BLOCKED:** app/code inaccessible or
execution context cannot be established. A remote code is not a substitute.

## S01 — Structures, form edits and resource review

**Needs:** S00, chosen fixture manifest. **Submits:** no scientific jobs;
an explicitly saved new input structure may create a StructureData node.

1. In **SCF energy**, use **Import from computer** for the selected fixture, or
   **From SMILES** for a chosen molecular starting structure. For NH3 use `N`
   only if its generated formula is NH3/4 atoms; for H2 prefer an explicit fixture
   rather than relying on implicit-hydrogen conventions.
2. Check **Structure details**, atom order, coordinates, cell, PBC and system
   classification. Inspect **Edit structure** / the available cell editor.
   For free molecules use **Cell transform**, a=b=c=12 Å and α=β=γ=90°,
   **Scale atoms position** off, then **Apply cell parameters**. Verify positions
   and cell. Do not select isolated-system preparation or apply its suggestion:
   retain the explicit box/XYZ periodic model. Confirm effective native settings
   later. For slabs use the agreed slab cell. An XYZ without a cell is insufficient.
3. Use **AiiDA database** to reopen the newly stored fixture, when a later route
   requires database selection. Record its UUID and local PK. Use a fresh input
   copy for this campaign; several workflows append bookkeeping extras to their
   input structures. Never use an ORCA result as the input for this check.
4. Set **Protocol: Standard** as the starting proposal. Record the actual
   generated method/basis/cutoff later; do not infer them solely from the word
   Standard. **Low accuracy** is the user-authorized integration approximation.
   **Debug is excluded from this campaign**: its tolerances/step limits
   are deliberately unsuitable for that purpose.
5. Set **# Nodes = 1**. CP2K **# Tasks per node = 2, 4 or 8**, chosen for the
   case; **# Threads per task = 1** always. Four MPI ranks with one thread each
   allocate four CPU cores. H01 PBE uses 2 ranks, H01 PBE0 uses 4, and CP2K 6×6
   hBN uses 8. Small stages use **Walltime: 00:10:00**; larger electronic stages
   use twenty minutes only if agreed. Record the external 32 GiB larger-case
   budget and actual RAM/headroom; the earlier 2 GiB small-case estimate is not
   a stop threshold. Memory is not entered in these GUI resource panels.
   Check the parsed total. Post-processing starts at 1/1/1, ten minutes.
   Inspect the memory and retry prerequisite
   supplied by the maintainer; there is no memory field in these forms.
6. Change a Small draft Walltime to `00:09:00`, check its parsed value, and restore
   its agreed tier value.
   In a non-submitted draft, try malformed walltime and observe the validation,
   then restore. Do not press Submit just to find out whether an invalid budget
   launches a job. Record missing pre-submit validation as such.
7. Record **Workflow description** using the case prefix. Capture the final
   form after any edits. **Estimate resources** is an optional estimator with
   site-oriented defaults; it is not a Mac benchmark or authority to enlarge
   the budget. If used in a draft, restore all limits afterwards.

**PASS:** geometry/identity survives import and reopening, edited values are
visible and accurately parsed, final resources are explicit. **FAIL:** lost
cell/atom identity, stale displayed settings, or unexpected launch. **BLOCKED:**
fixture or resource prerequisites missing. Unsupported import formats are
uncovered, not implicitly accepted.

## S02 — Prepare a tiny molecular reference and NEB endpoints

**Needs:** S01, selected N1 or R1, bounded local resources/retries/storage.
**Submits:** N1: one endpoint geometry optimization and one reflected-endpoint
SCF (plus an endpoint-A SCF only if needed for an identical energy comparison);
R1: at most one independent H2 optimization. Run each serially.

1. Choose **Geometry optimization** and the new molecular fixture. Confirm
   neutral closed shell, explicit 12 Å cube/XYZ periodic model with no isolated
   preparation, **Protocol**, dispersion
   policy, no unintended constraints and fixed cell. Record any **DFT settings**
   overrides. For N1 the later NEB form does not expose all the same overrides,
   so select a protocol that can be matched by NEB.
2. Set resources as S01 and **Workflow description = <prefix>-S02 ...**.
   Review the form, press **Submit once**, and record the workflow UUID through
   its process link. Resolve an uncertain submission by that identity, not a retry.
3. Wait for electronic and geometry convergence. Open the geometry result through
   **Search** / its process link, inspect the final structure and retain its UUID.
4. For N1, prepare a separate endpoint B by reflecting the converged ammonia
   pyramid through its H plane while preserving atom identity and a common cell.
   Use the editor's perpendicular-plane **Mirror** on N only, with the exact
   converged H-plane normal and point from read-only audit. Type full precision:
   installed **From selection** rounds coordinates to two decimals. Do not
   invent approximate coordinates during the run. Inspect both endpoints in the
   real GUI. Their relationship is reflection, not a relabelling of H atoms.
5. Open **SCF energy** for endpoint B. Leave **Compute band unfolding**, **Write
   overlap matrix** and **Compute Bader charges** off. Match method/electrostatics,
   choose **Skip automatic WFN restart search** for an independent fresh control,
   leave **WFN restart folder:** empty, review and submit once within limits.
6. Compare endpoint A/B energies with matching settings and units. Proposed
   endpoint symmetry tolerance: 1 meV, finalized before execution. A missing
   gradient check on B is not proof it is a stationary minimum; use the reflected
   converged geometry and record the symmetry assumption.

**PASS:** native convergence, finite energies, correct molecular identity and
endpoint symmetry. **FAIL:** unconverged result, changed identity or unexplained
symmetry mismatch. **BLOCKED:** comparable protocol/reference or fixture missing.
This does not establish a vibrational minimum or an experimental inversion barrier.

## S03 — NEB replica editor and one-group resource layout

**Needs:** S02 endpoint pair. **Submits:** no native jobs; **From viewer** /
**Interpolate** may store new structure nodes and are not purely read-only.

1. Open **Nudged elastic band**. Load endpoint A into the structure viewer and
   use the initial row's **From viewer** action; set the last row to endpoint B
   through its PK field or **From viewer** action. Use **Show** to inspect both.
2. Use the initial row's green **+** to insert a row between A and B. Set
   **Factor: 0.5**, then **Interpolate**: this is the midpoint M. Use **Show**
   and record its new UUID/PK. Next insert after A, again Factor 0.5 between
   nearest defined neighbors A and M, to obtain Q1 (one-quarter overall). Insert
   after M, Factor 0.5 between M and B, to obtain Q3 (three-quarters overall).
   Final explicit row order: **A, Q1, M, Q3, B**. Fractions are local to each
   pair of neighboring defined rows; entering 0.25 between A and M would be wrong.
   Check each image's atoms, cell, mapping, signed umbrella height and minimum
   separation; M should be near planar. No boundary jump or collision is allowed.
   This tests GUI-generated replicas, not just CP2K's automatic interpolation.
3. Propose **# of replica = 5** (including endpoints), **# rep / group = 5**,
   **Band Type = CI-NEB**, **Align Frames = off**, **Rotate Frames = off**,
   **Optimize Endpoints = off**, **Steps before CI = 5**, spring constant 0.05
   in the native convention. Confirm every choice against the manifest.
4. Set the replica count/group **before** resetting resources to 1 node,
   4 MPI tasks/node, 1 thread/task, ten minutes. Count/group changes can resize the
   allocation. Enable **Show NEB resource helper**. Expect one replica group,
   four MPI tasks per group and **NPROC_REP = 4**, never zero machines for the
   possible initial SCF. No five-way CPU allocation is allowed.
5. In disposable draft rows, try an endpoint with different atom count or species
   order; inspect the visible error and restore the valid endpoint. Do not submit
   an invalid band. Test the lower image-count bound against explicitly supplied
   rows and inspect a non-NEB **Restart from PK:** selection without submitting.
6. Remove Q1 with its **×** row action in the draft, leaving endpoints intact;
   its stored node must remain in provenance. Reinsert after A and interpolate
   again at 0.5 between A and M. Record the new UUID and equivalent coordinates,
   then recheck order and resources. Use **Show** on every row; all five images
   must be explicit and the summary must report zero additional images to be
   interpolated by CP2K. Capture all five UUIDs/PKs. Identical-element permutations
   cannot be detected from species lists alone; inspect the atom correspondence.

### Grouping and processor layout regression

All rows below are **form-only checks**, with one thread per MPI task. After
setting each layout, use **Show** on an image and switch between Replicas and
Advanced settings. The visible **# rep / group**, resource helper and
**# processors / rep** must retain the same layout. The automated regression
also compares the parent, section and resource traits with the prepared builder.

| Images | Replicas/group | Nodes | MPI tasks/node | Groups | NPROC_REP |
| --- | --- | --- | --- | --- | --- |
| 5 | 5 | 1 | 1 / 2 / 4 / 8 | 1 | 1 / 2 / 4 / 8 |
| 5 | 1 | 1 | 5 / 10 / 20 / 40 | 5 | 1 / 2 / 4 / 8 |
| 6 | 3 | 1 | 8 | 2 | 4 |
| 6 | 2 | 1 | 12 | 3 | 4 |
| 6 | 1 | 1 | 24 | 6 | 4 |
| 6 | 6 | 1 | 4 | 1 | 4 |
| 5 | 5 | 2 | 4 | 1 | 8 |
| 6 | 3 | 2 | 4 | 2 | 4 |
| 5 | 1 | 5 | 4 | 5 | 4 |

Use a disposable six-image draft for the additional divisors. Also change a count
so that its previous group size no longer divides it, and verify the deliberate
fallback to one replica/group is reflected everywhere. Count/group changes can
resize the resource fields. Record the resulting allocation before resetting it.
An existing Daint code may be selected for these form checks if local metadata
restrict larger layouts. Do not submit these drafts. Restore the agreed localhost
five-image, five-per-group, one-node/four-MPI/one-thread layout before S04.

A fractional node per group describes how BAND shares its MPI allocation; it is
not a fractional machine request. The selected backend allocates the optional
initial SCF exactly NPROC_REP MPI ranks on the fewest whole nodes within the
BAND node/per-node limits. For example, BAND 1node/20MPI with five environments
uses SCF 1node/4MPI; BAND 4nodes/6MPI with three environments uses SCF 2nodes/4MPI.
Threads and the parent BAND allocation are retained. Inconsistent allocations
return exit392 before child submission. Record actual WFN reuse and whether an
initial SCF runs. A form-only layout PASS does not qualify the native SCF path.

**Initial-SCF allocation regression:** keep the NH3 S04 draft unchanged and open
a separate fresh NEB page. Use a bare SMILES-C methane structure in a 12 Å cell,
with no calculation ancestor, and a saved endpoint moving only H atom4 by
0.03 Å along z. Load the saved endpoints by PK. Insert/interpolate one row at
local factor1/3, then a second after it at factor0.5, giving four explicit images
at fractions0,1/3,2/3,1. Inspect all four Show actions. Source FFF may be retained;
review the effective original-cell XYZ/PERIODIC model and trace native TTT copies
to their source UUIDs. Choose Standard PBE-D3/RKS/charge0, CI-NEB/spring0.05/CIafter5,
frames/endpoints off and restart blank. Set localhost1node/4MPI/1thread/600s and
two replicas/group: two environments, NPROC_REP2. A new kernel after backend
alignment is required. Submit once after review. With naturally unavailable WFN,
expect serial SCF1node/2MPI then BAND1node/4MPI, each with its requested600s limit;
no retry loop is added. Audit ranks/threads, WFN copies, all four replica results
and SCF/BAND convergence separately. This is functional allocation coverage,
not methane barrier validation. Save evidence/reopening route and close only
completed-test tabs. The generic two-endpoint/count4 CP2K interpolation route is
also supported; its bisected fractions need not be uniform.

See the [backend resource recipe](https://github.com/nanotech-empa/aiida-nanotech-empa/blob/master/docs/neb-initial-scf-resources.md).

**PASS:** correct ordered images, useful rejection diagnostics, image floor,
one-group layout and final resources. **FAIL:** silent invalid mapping, lost row
state or unsafe layout. **BLOCKED:** editor cannot represent the chosen endpoints.
Passing this editor case does not pass NEB science.

## S04 — Tiny CI-NEB and energy profile

**Needs:** S03, confirmed local envelope. Use at most one initial SCF and one
BAND CalcJob per submission. Record actual WFN reuse and child count. A useful
Low-accuracy continuation may request 30 minutes in the GUI. Inspect actual
native walltime, which reserves workflow time and can be shorter. Record every
native convergence criterion; no custom threshold UI is requested.

1. Preserve S03's final band and set **Workflow description = <prefix>-S04 NH3**.
   Review all settings, particularly after any replica edit, then **Submit once**.
   Record the parent and each child UUID; initial SCF and BAND must be sequential.
2. Observe status through the submitted process link. Do not equate “finished”
   with converged; obtain native SCF and BAND convergence evidence from the
   maintainer and record it separately from functional visualization acceptance.
3. Open **Search → NEB result** or the maintained `view_neb.ipynb` route for the
   recorded result. Use **Load node:** / **Show** where offered.
   Inspect the ordered replica thumbnails, relative energy plot and distances;
   use **View & export** for an endpoint and the highest interior image.
4. Verify five output images and matching energy/distance arrays. The viewer
   subtracts the first-image energy; reconstruct forward and reverse barriers
   from the raw energies separately. Do not sum distances twice or interpret
   the horizontal coordinate as a bond length without checking its definition.
5. Require continuous umbrella inversion and a positive, finite interior maximum
   above both symmetric endpoints. Proposed symmetry tolerance is 1 meV; barrier
   value itself has no asserted literature target in this integration test.
   Inspect force/displacement convergence against the actual native thresholds,
   not the plotted smoothness or the parser's successful exit alone.
6. Compare the displayed barrier evolution with accepted native steps. An
   overall downward trend may oscillate; require honest matching data, not
   monotonic descent. Exclude unfinished trial evaluations. Check fresh-page
   reopening and exported coordinates/cell/PBC separately. Existing Advanced
   settings do not override force/displacement thresholds: Low accuracy relaxes
   the force criteria and SCF tolerance, with displacement limits unchanged.

**Functional PASS:** actual GUI plot/geometry/export/reopening works, accepted
energies and geometry match native data, evolution is meaningful and limitations
are disclosed. Nonconvergence alone does not fail this revised functional case.
**Scientific convergence:** report ACHIEVED or NOT ACHIEVED independently against
the actual native thresholds; no validated barrier claim from an unfinished band.
**Functional FAIL:** incorrect mapping/units, lost geometry, unusable results or
exports, or a new discrepancy that prevents interpreting the result. Keep known
reporting defects in the ledger without silently relabeling them as fixed.
**BLOCKED:** required result data or GUI unavailable. Preserve all original
attempts. Neither band proves a transition state without further analysis.

## S05 — One-CV replica chain

**Needs:** S01 and selected R1/reference, bounded retries. **Submits:** initial
SCF then a few constrained geometry steps, serially. Maximum five native jobs.

1. Open **Replica chain**; import/select H2 and check two atoms, neutral singlet,
   explicit fixed 12 Å cube/XYZ periodic model, without isolated preparation.
   Match the reference protocol.
2. Press **Add constraint**. Enter **Constraint = distance atoms 1 2**. Check
   **Target A** against the starting bond distance. Record **K eV/A^2** (the
   form starts at 40). This is a finite restraint; the actual distance need not
   equal its target exactly. Do not call it a rigid constraint.
3. Set **CVs targets** to starting distance +0.06 Å and **CVs increments = 0.02**,
   using the finalized fixture values. Only one CV may have a nonzero increment.
   Leave **Restart from PK:** empty. In a draft, inspect an inconsistent number
   of targets/increments and restore; do not launch to probe a missing guard.
4. Set resources, description and final form capture; **Submit once**. Count all
   children and retry attempts. The native stage cap and aggregate case cap apply
   to automatically launched work too; a prerequisite stop mechanism is required.
5. Open the replica result through **Search** / `view_replica.ipynb`, using
   **Load node:** and **Show** if necessary. Inspect `initial_scf` followed by
   `step_0000`, etc.; **Target**, **CV actual**, **Energy (scf) /au**,
   **Energy (force) /au**, and **dist to prev**.
6. Check actual distances from the exported structures independently. Proposed
   tolerance between measured geometry and stored `cvs_actual`: 0.001 Å. The
   two-decimal display is insufficient to prove that tolerance by itself.
   Tabulate target-minus-actual differences rather than hiding restraint offsets.
7. Check termination against the implemented “within one increment” rule:
   final distance should be within 0.02 Å of the requested final target for this
   proposal. Do not demand exact equality or count a no-op SCF as chain coverage.
   Require at least two changed, converged geometries. Inspect electronic energy
   and restraint-inclusive FORCE_EVAL energy separately; the plotted x axis is
   cumulative structural displacement, not necessarily the CV itself.

**PASS:** ordered, converged one-CV evolution with correct distances/energies,
at least two real steps, bounded serial children and preserved provenance.
**FAIL:** incorrect mapping, uncontrolled retries, no-op completion, incorrect
units or unreported nonconvergence. **BLOCKED:** resource policy or needed controls
unavailable. Multiple evolving CVs and dissociation are not covered.

## S06 — Optional replica continuation and NEB restart inspection

**Needs:** S05/S04 respectively; explicitly selected continuation scope.
**Submits:** replica continuation only, at most one additional native geometry
step if its target/termination rule supports that budget. NEB restart is a draft.

1. Open a fresh **Replica chain** draft and enter S05's **Restart from PK:**.
   Keep the same atom order, method, cell, restraint and single CV. Choose a new
   target that adds one planned increment beyond the saved last point; account
   for the within-one-increment stopping rule before submission.
2. Review the inherited and changed values. Submit once only after the bounded
   continuation is established. Record a **new parent UUID** and original UUID.
   Earlier outputs should remain the original nodes; new steps add provenance.
3. Reopen both parents and compare the old step identities and energies. Record
   whether the continuation restarted a WFN or recomputed; do not invent reuse.
4. Separately open **Nudged elastic band**, enter S04's **Restart from PK:** and
   inspect inherited replicas, count floor and the hidden fresh-band editor.
   Clear it and verify the fresh draft state. Do not submit another band.

**PASS:** continuation adds the intended new work without rewriting earlier
science, and NEB restart preview identifies the original band. **FAIL:** mutated
history, stale fresh-band values used on restart, or unexplained duplicated steps.
**BLOCKED:** saved outputs/required WFN absent or continuation cannot be bounded.

## S07 — Adsorption fragment partition and readiness

**Needs:** selected A3 (or explicit smaller A1/A2) stored as a new StructureData input through S01;
agreed geometry preparation. **Submits:** no native jobs.

1. Open **Adsorption energy**. It selects from **AiiDA database** (optional
   openBIS is outside this campaign); it does not offer this form's own XYZ uploader.
2. Retain the automatically created **all** fragment. In the structure viewer,
   select each physical fragment; enter **New fragment** name and inspect
   **Selected indices:** before **Add fragment**. Use `water_a`/`water_b` for A1
   or `sheet`/`h2` for A2, **`sheet`/`benzene` for A3**. Confirm one-based
   **Indices**, **Charge** and any **UKS**
   / **Multiplicity** values. Non-all fragments must partition every atom exactly
   once, with no overlaps/omissions, and their charges must sum to the full charge.
3. Set **Fixed atoms** explicitly: blank for the agreed A1 control; the exact
   substrate indices for A2/A3. Verify the same identities after fragment extraction.
   Do not assume the full adsorbed system will be relaxed by this workflow—it is not.
4. Set **Protocol**, and inspect the actual electrostatic controls for the
   common periodic cell. The builder historically initializes periodic XYZ and
   dispersion enabled; the native effective input is the authority, not a label.
5. Set resources separately on **all** and both fragment cards to the agreed
   tier (A3: Medium proposal). Select a configured local **Cubegen code:** for
   mandatory [C01 charge-density-difference coverage](extended-local.md#c01--adsorption-charge-density-difference-cube-and-gui). In **Advanced settings**, leave **WFN restart folder:** empty
   and enable **Skip automatic WFN restart search** for this fresh control.
6. In disposable draft fields inspect duplicate fragment names, overlapping
   indices, missing atoms and an out-of-range index; restore the valid partition.
   Record absent validation. Do not press Submit while any invalid state exists.
7. Capture the complete valid form with **Run stages one at a time** enabled
   and **Max CP2K attempts per stage: 1**. The serial option is supported by the canonical workflow. The unchecked/default mode retains
   concurrent behavior for existing callers; the local campaign uses serial ON.

**PASS:** correct visible partition, indices, reference choices and per-fragment
budgets, with gaps recorded separately. **FAIL:** displayed mapping or resource
values are wrong. **BLOCKED:** fixture/control missing. Validation subcases with
no guard remain uncovered/defective; passing partition inspection does not pass them.

## S08 — Adsorption/fragment energy arithmetic, paired with C01

**Needs:** S07; verified serialization, storage/retry limits and a prepared full
geometry. Demonstrate native serialization and independent energy/density checks on this candidate.
**Submits when ready:** three SCFs, two fragment relaxations and one cubehandler
without retries, all serial. With C01, A3 has a proposed 110 min aggregate cap
(20 min per electronic stage, ten for the helper); Small A1 has 60 min total.
These are stopping ceilings, not runtime predictions. Cube viewing adds no job.

1. Reopen the valid S07 form, recheck all cards and source geometry. Give it a
   unique **Workflow description**, then **Submit once**. Record all child UUIDs,
   start/end times and fragment identities. Stop the branch on overlapping jobs.
2. Open **Search → Adsorption energy** / `view_adsorption_energy.ipynb`. Enter
   **Workflow PK:** and **Load** where necessary. Inspect **Adsorption energy
   (frozen)**, **Adsorption energy**, and each **Deformation energy**.
3. Independently obtain unrounded native/parsed energies with their actual units.
   Define `Eall` at the supplied full geometry; `Ei_frozen` for extracted fragment
   geometries; `Ei_relaxed` after their permitted relaxation. Check:

   ```text
   E_frozen = Eall - sum(Ei_frozen)
   E_relaxed_reference = Eall - sum(Ei_relaxed)
   D_i = Ei_frozen - Ei_relaxed
   E_relaxed_reference - E_frozen = sum(D_i)
   ```

4. Require raw stored sums to agree within proposed 1e-8 Hartree (or precision
   justified from native text) and the eV display within 0.001 eV of conversion
   of the unrounded value. The inspected viewer currently rounds in Hartree
   first; preserve any mismatch as a display failure, including a false zero.
5. Relaxations must converge and normally give nonnegative deformation energies
   within the agreed numerical tolerance; investigate violations. A negative
   interaction/separation energy means binding under this sign convention.
   Do not require a guessed sign/magnitude for an unrelaxed A2/A3 starting geometry.
6. Verify equal numerical conventions, complete fragments and fixed-atom mapping.
   Distinguish electronic energies from free energies: no ZPE, entropy,
   counterpoise/BSSE or finite-size convergence has been demonstrated here.
7. Complete **C01** using this parent's cubehandler child and frozen-SCF density
   sources. Record separate energy and density verdicts; the selected adsorption
   branch is incomplete if its charge-density difference is missing or unaudited.

**PASS:** converged component calculations, correct arithmetic/sign/units/display,
valid geometry/reference meaning, serial execution and storage audit. **FAIL:**
false precision, unconverged component accepted, bad partition, changed references,
or concurrency/budget violation. **BLOCKED:** missing components or readiness gates.
A1 PASS is molecular fragment accounting; A2/A3 exercise surface adsorption.
C01 must also pass before reporting the full requested adsorption coverage.

## S09 — CP2K unfolding primitive-cell preview

**Needs:** selected U1 (or separately agreed smaller U1), S01.
**Submits:** no native jobs.

1. Open **SCF energy**, select the pristine hBN supercell and enable **Compute
   band unfolding**. **Write overlap matrix** should become enabled automatically.
   Leave **Compute Bader charges** and **Also retrieve sparse overlap matrix** off.
2. Enter the exact primitive **a1:** and **a2:** from the manifest; leave **a3:**
   blank for the 2D mapping. Choose **Path type: hexagonal**, **Path: G-K-M-G**,
   and **Basis atoms:** for one B and one N in the same primitive basis, using
   their actual one-based indices, not assumed `1 2` after arbitrary import.
3. Press **Check primitive cell**. Record exact snapped vectors, integer matrix,
   determinant, correction, basis counts, displacement max/mean and reciprocal
   preview. For primary **6×6 monolayer U1 (72 atoms)** expect det(M)=36,
   36 copies of each basis site and
   displacements near coordinate-rounding precision (proposed <1e-5 Å).
4. Confirm `M^T q` is integer for each of the **36** folded primitive points.
   With 6×6, both K and M are commensurate; verify those points in the actual map.
   A path does not create additional q points or a continuous band. Distinguish
   all folded points from on-path points; boundary-equivalent points may repeat.
   A separately selected 3×3 fallback has 9 points and K but not M; 2×2 has four
   points and M but not K. Do not retain the 6×6 acceptance counts after downsizing.
5. Change a primitive vector or basis atom; the preview should say it is stale.
   Restore and recheck. Try a duplicate basis index, an out-of-range index,
   a missing element, or singular vectors via **Check primitive cell** only;
   expect useful errors, then restore. A large snap correction is not scientific
   validation even if the numerical snapping algorithm finds an integer matrix.
6. Choose the local **Unfolding code:** and set its independent resource panel
   to 1/1/1, ten minutes. For U1 set CP2K to **1/8/1** (eight cores), **PBE**,
   twenty-minute electronic stages if agreed. Record the **32 GiB total external
   memory budget** and available worker headroom separately; there is no overall
   memory field to set in this form. Set **ADDED_MOS: 8**
   provisionally (four for the smaller 2×2 fallback), **Overlap digits: 14**,
   **AO threshold: 1e-10**. With ordinary B(q3)/N(q5), U1 has 288 electrons and
   144 occupied bands; verify actual data and energy coverage. Agree whether
   to enable **Self-consistent diagonalisation**; use it for the proposed pristine
   control, leave Fermi-Dirac smearing off. Check compatibility with the frozen
   closed-shell model and native CP2K data before submission.

**PASS:** correct tiling, basis map, allowed q set, useful rejection diagnostics
and preview invalidation. **FAIL:** stale or incorrect mapping/units. **BLOCKED:**
fixture/worker missing. A preview PASS is not electronic/unfolding validation.

## S10 — CP2K unfolding computation and stored viewer

**Needs:** S09, local CP2K/WFN-compatible unfolding worker and audit prerequisites.
**Submits:** OT SCF → diagonalization → unfolding, sequentially; proposed U1
ceiling 50 min total (20+20+10). Smaller cases use their separately agreed tier.

1. Recheck S09 and the complete final form. Set **Workflow description**,
   **Skip automatic WFN restart search**, and empty **WFN restart folder:** for
   the independent control. Press **Submit once**, recording parent/child UUIDs.
2. Require converged native electronic stages, WFN/AO-basis consistency and a
   successful postprocessor. Verify the stored input vectors and mapping against
   the last preview: the form passes vector text and the worker performs its
   mapping, so identical preview/submission meaning needs an audit.
3. Open the unfolding result from **Search** or the maintained
   `view_cp2k_unfolding.ipynb?pk=<new-workflow-PK>` page. If no search link appears,
   record that navigation gap; a documented result URL is still the real GUI.
4. Inspect printed primitive vectors, M, det(M), folded/on-path counts, path,
   basis indices and mapping displacements. Change **E min [eV]:**, **E max [eV]:**,
   **Scale:** and **Show k-map**; restore them. Use **Add projection** for a B/N
   atom selection and available s/p channels. Check stated spin identities.
   This nonmagnetic case does not qualify spin-polarized unfolding.
5. Independently inspect `unfolding_bands.npz` and `unfolding_projections.npz`
   in `unfolding_retrieved`. Check finite arrays, dimensions, eigenvalue units,
   reference-energy convention, mapping and all 36 q points for U1, not only plotted
   points. For the ideal translationally repeated model, propose weight bounds
   [-1e-6, 1+1e-6] and per-normalized-orbital sum over the **complete** q set
   within 1e-5 of one. Confirm normalization conventions before freezing these
   tolerances. Degenerate states may mix; compare their total subspace weights.
6. The viewer clips negative weights for marker drawing; a clean plot cannot
   establish raw positivity. Check raw data. Threshold stability or a denser
   reference requires a separately selected run, not an invisible recalculation.
7. Reopen the same result and verify identical arrays/identity and no new jobs.
   Missing GUI NPZ download is a download-coverage gap; maintainer repository
   inspection remains separate evidence, not a fabricated browser action.

**PASS:** native convergence, correct stored mapping/eigenvalues/weights, usable
viewer and durable compact outputs. **FAIL:** bad normalization, invented unsampled q data,
wrong energy zero, false success, resource/storage breach or rerun on viewing.
**BLOCKED:** diagnostics or compatible local files/code missing. No defect,
spinor, arbitrary 3D-lattice or continuous-path qualification is implied.

## S11 — Durable reopening, duplicates, provenance and storage

**Needs:** at least one new completed case; prior failures retained.
**Submits:** nothing during the ordinary reopening check.

1. Save the new parent/child UUIDs and local PKs, final-form captures and result
   checksums. Close only this campaign's browser pages; leave ORCA alone.
2. Reopen the app and **Search**. Find by the available search controls or the
   recorded result route/PK; do not assume UUID search exists. Resolve the PK
   back to the recorded UUID through exposed details or the separate audit.
3. Reopen the same result twice, use **Show**, **Load**, search refresh and plot
   controls where applicable, and follow **View & export**. Confirm unchanged
   scientific arrays, source links and geometry, and no extra scientific processes.
   Durable saved results are required; unsaved draft/plot preferences need only
   persist where the UI promises that behavior.
4. Inspect submission controls after a completed or active submission without
   pressing Submit again. Several direct forms permit another process; SPM
   disables its button but durable reopen protection is unproven. Record
   duplicate protection as **source-identified gap / behavioral test NOT RUN**.
   Do not force a second calculation to demonstrate the known gap.
5. After a future focused duplicate guard is implemented, schedule a specific
   retest: one valid tiny request, two rapid clicks, reopen the active request
   and repeat its action. Independent process counting must show exactly one
   parent and the expected children. Do not run this hazardous subcase until the
   guard exists and the retest is authorized; record FAIL if a duplicate escapes.
6. Have an independent maintainer audit raw convergence, settings, resource/retry
   history, work-directory/repository growth and provenance using only this
   campaign's UUIDs. Verify original sources were not overwritten. NEB viewing
   may add thumbnail extras to new replica nodes; distinguish this bookkeeping
   from immutable attributes/repositories/links. Do not open historical results
   merely to manufacture a mutation test.

**PASS:** durable results, expected job count, preserved scientific provenance
and audited budgets. Duplicate protection gets its own verdict and cannot pass
from a single click. **FAIL:** extra calculation on viewing, incorrect identity,
lost data or altered science. **BLOCKED:** inaccessible stored outputs/audit.
Complete the report with unsupported/unconfigured/unvalidated paths explicitly
listed and dependent cases left NOT RUN or BLOCKED as appropriate.
