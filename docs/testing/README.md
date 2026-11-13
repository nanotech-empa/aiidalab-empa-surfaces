# On-surface chemistry: modern GUI qualification

These are reproducible manual cases for the current canonical Surfaces app.
The primary operator performs the real browser actions. Code tests do not pass
manual cases. Start only after the existing installation matches the agreed
canonical source/dependency manifest and the executable release preflight passes.
Record canonical master, actual imports, installed metadata, reviewed candidate
exceptions and their fingerprints for every run. Relevant source changes
invalidate affected cases.

This suite retains the existing notebook/widget interface. It covers the editor,
representations, PBE/PBE0, NEB, replica chain, adsorption/density difference,
AFM/STM, PDOS and CP2K unfolding. BandUPpy belongs to the separate
[QE post-processing app](https://github.com/nanotech-empa/aiidalab-empa-qe-postprocess/tree/main/docs/testing).
Gold is editor-only. XPS and frozen backward-compatible releases are excluded.
Use Standard or Low accuracy only. Do not select isolated preparation for our
molecular CP2K cases: use explicit 12 Å cubes and effective XYZ/PERIODIC.

Historical Nov10–11 acceptance reports remain local evidence of the old composed
installation. The user later authorized deletion of specified test nodes and
workdirs. Do not claim those native results still reopen, or transfer their PASS
status to this candidate. Reuse only fixtures whose current identity is verified.
The new campaign has its own UUIDs and verdicts. Keep unrelated research intact.

Widgets-base is the official 3.0.1 release. Its known explicit same-source reload
limitation is tracked by optional upstream PR832, which remains uninstalled and
pending maintainer review. Record that limitation honestly and use an intervening
source selection where required. Never hide it with an EMPA private-cache patch.

## Coverage and ownership

| Cases | App / purpose | Evidence required |
| --- | --- | --- |
| S00–S01, E01–E02 | Launch, CH4 creation, Au(111), cell editing, auto/manual representations | Geometry versus display changes; saved representations on new structures; no scientific submission |
| H01 | CH3 PBE control and PBE0 | Neutral doublet, 12 Å cell, effective hybrid/ADMM recipe, exchange cutoff and memory, native convergence |
| T01, F01 | Benzene without substrate: STM and AFM | Nonempty numerical data, coordinate/height/energy conventions, appropriate qualitative checks, durable results |
| D01 | Benzene on primitive 4×4 hBN: PDOS | Atom projections, band/energy bookkeeping, broadening, exports; optional overlap is separate |
| S02–S04 | Tiny endpoint preparation, NEB editor, CI-NEB | Native electronic and band convergence, ordered images, barrier and endpoint checks |
| S05–S06 | One-CV replica chain, optional continuation | Actual/target CVs, restraint and electronic energies, ordered provenance, serial children |
| S07–S08, C01 | Fragment definitions, adsorption energy and density difference | Serial component calculations, independent energy arithmetic, signed Δρ from frozen fragments, full-grid conservation and GUI cube viewer |
| S09–S10 | CP2K unfolding setup, calculation and viewer | Primitive mapping, allowed folded points, raw weights/eigenvalues, compact stored outputs |
| S11 | Search, durable reopening, duplicate prevention, storage | Same UUIDs/data after reopening; no new jobs; independent resource/provenance audit |
| B00–B04 | Separate QE post-processing / BandUPpy app | Local QE template, mapping, folded QE and reference jobs, stored BandUPpy spectra and reopening |
| DOC01, alongside every selected case | GUI help and information for both AiiDAlab chatbots | Missing/misleading help, concrete corrections, source/version labels, separate content/integration/answer-check status |
| P01–P03 | [Boundary/status controls](boundary-model.md) | Real linked restart transitions, explicit effective model, partial status and unchanged legacy records |
| C02 | [Shared cube viewer and renderer](cube-viewer.md) | Source-qualified preview, real plane/PNG export, one bounded full-grid render and reopening |

Execution statuses and actual UUIDs live in the current handoff. Source inspection
alone establishes available paths and readiness issues, not a browser PASS. See
[source evidence](source-evidence.md), [core Surfaces cases](localhost.md),
[editor and added scientific cases](extended-local.md), and the
[report template](report-template.md).

## Selected small integration systems

These are integration models, not production predictions. Freeze the exact
fixture and numerical settings before its first submission. The systems follow
the user's selections. Costs below are planning estimates, not measurements on
the aligned candidate. The larger-case 32 GiB budget is an observed OS/process
budget, not a CP2K-app memory field. A cap is a stopping limit, not a completion
promise. Keep one native job at a time.

| Candidate / cases | System and rationale | Estimated local cost / proposed tier | Prerequisites and scientific checks |
| --- | --- | --- | --- |
| **E1 / E01–E02** | Create CH4 (5 atoms), then place it on **Au(111)** with the built-in slab editor; try nx=2, ny=2, nz=3 | GUI only, seconds to minutes; **zero native jobs** | SMILES and shared viewer already available. Expected builder composition: CH4 + 24 Au + 8 bottom H = 37 atoms. Check extra H explicitly, placement, cell, auto/manual styles and saved representations. This is editor coverage, not Au adsorption science |
| **N1 / S02–S04** | NH3 inversion, 4 atoms, **five explicitly supplied images** | Small SCFs ~0.5–2 min, endpoint optimization ~1–5 min, band ~3–10 min; **Small** tier, convergence uncertain | Add midpoint and both quarter-path replicas with the **GUI + / Interpolate** controls. Symmetric endpoints, continuous path, positive barrier, native electronic and BAND convergence |
| **R1 / S05–S06** | H2 near-equilibrium stretch, 2 atoms, one distance CV | Initial SCF + about 2–4 restrained steps, ~0.5–3 min each; **Small**, at most 5 jobs / 30 min | One evolving CV avoids candidate concurrency. Target +0.06 Å, increment 0.02 Å, finite restraint; check actual versus target, energy definitions and real progression |
| **H1 / H01** | CH3 radical, 4 atoms, neutral doublet, fixed geometry in **12×12×12 Å**; PBE then **PBE0** | Two serial SCFs; ~0.5–2 min PBE, ~1–10 min PBE0; **Small** | C/H hybrid basis, potentials and ADMM data must exist. Explicit 25% exchange, proposed 5 Å HF cutoff, 256 MiB HF memory/rank. Check seven valence electrons with q4/q1 data, spin, native hybrid terms and convergence |
| **T1 / T01, F01** | Benzene, 12 atoms, no substrate, planar and wholly inside **12×12×12 Å** | Each branch OT → diagonalization → helper; ~1–10 min per electronic stage, helper ~1–5 min on a coarse grid; **Small**, 30 min per branch | Local STM/PPAFM workers and compatible native files. Both backends request at least 100 added MOs. Helper resource correction active; verify actual allocations. STM frontier-state maps and AFM numerical/height checks; no realistic supported-molecule image claim |
| **P1 / D01** | Benzene over **primitive-monolayer 4×4 hBN**, 12+32 = **44 atoms** | Two electronic stages, roughly 3–20 min each; **Medium**, 40 min case ceiling | PBE; agreed molecule height/orientation, cell and B/N/C/H data. Atomic PDOS first, **Do overlap calculation off**. Check benzene/substrate projections, common energy reference and exports |
| **A3 / S07–S08, C01** | Reuse the 44-atom benzene/4×4 hBN model for adsorption + **charge-density difference** | 3 SCFs + 2 fragment relaxations + cube helper; ~3–20 min/electronic child and <5 min helper; **Medium**, 110 min total cap, not a runtime prediction | Prepare a new or verified retained 44-atom input for this candidate. Keep sheet fixed, partition benzene/sheet. Serialize children; check frozen/relaxed energy identities and Δρ = ρall − ρbenzene − ρsheet using **frozen** fragments |
| **A1 / smaller adsorption control** | Water dimer, 6 atoms | Same six-stage sequence with density helper; ~0.5–5 min/electronic child; **Small**, 60 min cap | Cheaper meaningful fragment-energy and Δρ control if A3 is too costly. It does not validate surface adsorption. A2 (H2 on primitive 2×2 hBN, 10 atoms) remains a smaller surface alternative |
| **U1 / S09–S10** | Perfect **primitive-monolayer 6×6 hBN**, **72 atoms**, PBE, **8 Mac cores** | OT and diagonalization ~5–20 min each; unfolding ~1–10 min; **6×6 unfolding**, 50 min cap | Local CP2K and compatible unfolding worker; det(M)=36, 36 folded primitive q points, including Γ/K/M. Full basis mapping, raw weights and eigenvalues. More cores do not establish RAM or elapsed time |
| **B1 / B00–B04** | Perfect **primitive-monolayer 2×2 hBN**, **8 atoms**, in separate QE/BandUPpy app | Existing local template or 2 template jobs, then reference SCF/bands → folded QE → BandUPpy; ~1–10 min per QE stage; **Small**, at most 6 jobs / 60 min | Existing local workers/pseudopotentials, accessible QE density/WFN files; short G–M path, pristine reference, raw unfolding and durable reopening |

Use the campaign handoff for the serial case order and current verdicts.

For N1, a three-image band is the smaller calculation fallback, but still retain
the five-image GUI insertion check. For CP2K, **3×3 hBN (18 atoms)** retains K but
not M in the Γ-folded set; **2×2 (8 atoms)** retains M but not K. Either is a
meaningful smaller unfolding case if 6×6 cannot fit; neither has identical
reciprocal coverage. For QE, 2×1 hBN (4 atoms) remains a smaller nontrivial fallback.
For PDOS, bare benzene (12 atoms) tests projection plumbing but not molecule/sheet
coupling; a larger cell is not secretly replaced by an overcrowded benzene/2×2
sheet. Preserve failed attempts and discuss any replacement or raised limit.
No fallback moves work to remote compute.

## Freeze selected fixtures before execution

For each chosen fixture, fill a manifest in the report: formula and atom count;
one-based atom order; exact coordinates and file checksum; cell vectors, units
and PBC; charge/multiplicity and occupations; fixed atoms/CVs; method, dispersion,
basis/potential or pseudopotential identities; numerical protocol and tolerances;
primitive matrix/basis/path where relevant; expected stage and job counts.
Record new structure UUIDs. A generated SMILES geometry is a starting structure,
not an optimized reference.

**Molecules:** use an explicit **12×12×12 Å**, 90° cell; do not select an
isolated-system preset or apply the suggested isolated electrostatics. In the
inspected GUI, inactive electrostatic settings preserve the original cell and
use XYZ/PERIODIC; an isolated MT preparation can enlarge the cell to twice the
molecular span plus 15 Å. Check the actual stored and native cell, rather than
assuming the suggestion is active. This finite periodic-box test does not prove
isolated-limit convergence. Keep NH3/CH3/H2 intact near the center. For benzene
SPM, propose the molecular plane at z=3 Å, centered in x/y, so the requested
sampling and extrapolation lie inside the 12 Å box. Do not recenter blindly to
z=6 Å without rechecking the SPM extent.

**hBN:** “n×n” means repetitions of the **two-atom primitive monolayer**, not
sliders in the built-in **Build slab → hBN** tool. That tool uses an eight-atom,
two-layer rectangular unit: nx=6, ny=6, nz=1 would give **288 atoms**, not 72.
Prepare/review a cell-preserving primitive fixture later. An illustrative choice
is a1=(a,0,0), a2=(-a/2,sqrt(3)a/2,0), a≈2.50 Å; B at (0,0), N at fractional
(1/3,2/3), common z and about 15 Å cell height. Freeze the precise vectors,
centering and periodic model before execution. QE 2×2 and CP2K 6×6 share this
primitive lattice, **not an identical supercell**. The adsorption/PDOS 4×4 has
32 sheet atoms; add one benzene (12 atoms), initially flat at an agreed height
(e.g. 3.3 Å above the sheet), then qualify its geometry for the chosen purpose.
The Au editor fixture is separate and retains the builder's slab cell.

## Local resource and storage envelope

- Only local workers on the user's Mac (including its local Linux container).
  A notebook in a local browser does not prove that its selected code is local.
  Identify code UUID, computer UUID, transport and executable provenance through
  existing GUI information or a separate maintainer record. No remote/HPC/cloud/QPU.
- **One native scientific job at a time.** The user has made the Mac fully
  available overnight; ORCA qualification is reported complete. Preserve its
  historical records and configuration. One daemon worker is not proof that
  native jobs are serialized. Adsorption currently violates this limit (below).

**MPI tasks are processes, not threads.** Set **# Nodes = 1** and CP2K
**# Tasks per node = 2, 4 or 8** as selected for the case; always set
**# Threads per task = 1**. Four MPI ranks × one thread allocate **four CPU
cores**. H01 uses two ranks for PBE and four for PBE0; CP2K 6×6 hBN uses eight.
Keep this layout explicit in every form and native audit. The user withdrew
the additional OpenBLAS investigation/activation gate; it is not a prerequisite.

Resource tiers (all memory is an **external budget, total per active job**,
not per rank or a field to enter in the CP2K GUI):

| Tier | CP2K/QE allocation | External memory budget | Logical-stage cap including retries | Meaning |
| --- | --- | --- | --- | --- |
| **Small** | 1 node; 2 MPI ranks × 1 thread; H01 PBE0: 4 ranks × 1 thread | Observe actual RAM and available headroom; initial 2 GiB estimate superseded | 10 min | H01 PBE sampled about 3 GiB RSS; not a measured peak. QE 2×2 starts with two ranks |
| **Medium** | 1 node; 2 or 4 MPI ranks × 1 thread, selected in case review | **32 GiB** | 20 min electronic; 10 min helper | User-selected memory budget for 44-atom PDOS/adsorption and density cubes; timing remains proposed |
| **6×6 unfolding** | 1 node; **8 MPI ranks × 1 thread** | **32 GiB** | **20 min electronic; 10 min unfolding** | User's eight-core choice and 32 GiB budget; timing remains proposed |

Postprocessors start at 1 node / 1 task / 1 thread, with the same tier's total
external memory budget and a ten-minute cap. Verify that the local code supports
the selected MPI layout; keep one thread per task. Serial postprocessors remain
serial where required by their code. NEB uses all five
images in one replica group, NPROC_REP=4, sharing four ranks over time.

- These forms have **no overall job-memory field**. Do not instruct the GUI
  operator to enter 32 GiB, add a memory control, or fail a GUI case because the
  field is absent. Before execution record actual host/container capacity,
  existing external limits and the memory-sampling method/cadence. Report sampled
  maxima as such, not as true peaks. Distinguish a
  planning budget, monitored usage and an OS-enforced limit; the budget is not
  automatically enforced by the app. No container/service changes are part of
  this preparation. The budget permits up to 32 GiB; it does not predict that
  much use. Check available worker memory and **host/browser headroom**;
  if a selected case cannot fit, discuss a smaller one. HF max memory limits
  only the HFX subsystem per rank and is not the overall job-memory budget.
- Explicitly set every resource panel, including helpers. Small uses
  **Walltime: 00:10:00**, Medium/6×6 unfolding electronic panels **00:20:00** only if
  agreed. Verify parsed seconds and effective child inputs. Existing CP2K
  wrappers permit up to five iterations; retries must not reset these stage or
  case ceilings. Establish bounded execution/stop handling before running.
  Closing the browser does not stop AiiDA. Use the selected, UUID-scoped operator
  stop route in the local handoff; the reviewer remains read-only for science.
- Case ceilings: S02 30 min; S04 20; S05 30; S06 chain continuation 10 or
  Low NEB restart 30; H01 20;
  T01/F01 30 each; D01 40; S08+C01 110 (A3), 60 (A1); S10 50; B01+B03 60.
  Geometry preparation for A3, if selected, is a separately recorded stage
  (20 min maximum), not silently hidden in the adsorption budget. Reuse only
  valid, compatible earlier results and record reuse versus a fresh GUI run.
- Proposed storage ceilings: ordinary cases 250 MiB retrieved / 2 GiB new work;
  U1 500 MiB / 6 GiB; A3 including full density cubes 500 MiB / 4 GiB;
  QE template+BandUPpy 250 MiB / 4 GiB. Run staged subsets using actual free space
  and growth, rather than requiring the entire campaign's scratch at once.
  Start small stages within 1 GiB combined new work and retain 6 GiB headroom
  where feasible. Reassess before each larger case. Include exports and hybrid
  scratch. These are monitored budgets, not implemented controls. If a selected
  case cannot fit, use a meaningful agreed smaller case or report it blocked.
  Never delete scientific data automatically. Cleanup needs its own exact,
  user-authorized reviewed scope.
- Retain full WFN/AO/density/QE-save files in their original local work
  directories; avoid bulk AiiDA retrieval. NEB requests `*.xyz`, `*.out`, `*.ener`;
  adsorption emits stride-1 full cubes even without Cubegen. **Cubegen must be
  selected for C01**, whereas Bader and optional sparse-overlap retrieval stay
  off. Reduced cube previews are not substitutes for full-grid scientific data.

The NEB processor meaning and endpoint-inclusive image count follow the
[CP2K BAND documentation](https://manual.cp2k.org/trunk/CP2K_INPUT/MOTION/BAND.html).
Check the installed native version before running; current web documentation
does not establish that version's behavior.

## Source-supported constraints to verify in the actual GUI

- Enable **Run stages one at a time** and **Max CP2K attempts per stage: 1**
  for adsorption. Existing parallel callers are supported but outside this
  serial local case. Keep replica chains to one changing CV. Keep PDOS overlap
  off until its concurrency/resource path is separately qualified.
- Capture the complete pre-submit form and a unique case label. Resolve an
  uncertain submission from its process identity. Do not launch duplicates to
  demonstrate a missing submission lock.
- SPM helpers inherit GUI resources. Verify actual native allocations. AFM may
  run a serial numerical worker despite the scheduler allocation. Both SPM
  electronic paths request at least 100 empty orbitals. Check basis dimension.
- For ARM-local SPM/PDOS, start with **use ELPA** off and audit the actual
  diagonalizer. Do not assume AVX2 settings are appropriate.
- PBE0 changes basis/potential maps. Explicitly set 25% exchange, 5 Å cutoff
  and 256 MiB HF memory per MPI rank for H01. This is not hybrid/cell convergence.
- Separate structural connectivity from the effective calculation boundaries.
  Inactive suggestions must not change cell or electrostatics. Working/result
  PBC, native CELL/POISSON and ExtendedXYZ must describe the same actual model.
- Usable partial NEB data must remain viewable with honest timeout, electronic
  and band status. Functional visualization and native convergence have separate
  verdicts. Normal termination alone is insufficient.
- Adsorption eV display must match unrounded arithmetic. The cube link must
  resolve the actual completed helper, reject stale sources and show clear
  pending/unavailable feedback. Oblique preview and full-grid geometry must agree.
- Local code registration, native executable compatibility and the eight-atom
  QE template are readiness checks, not outcomes established by this document.

## How to execute later without changing the app style

The GUI operator uses maintained notebook entry points and visible controls.
Never execute notebook cells manually, call `engine.submit`/services, or use a
browser console as a replacement for an unavailable GUI action. Do not edit
numbered runtime notebook copies. Fixes belong to an implementation colleague;
the operator retests the exact revised build on new, clearly identified cases.

Before each submission, capture the full visible configuration, including
every child-stage resource panel. Edit a harmless field, check that the form
and any preview update or become stale, restore it, and capture the final form.
After submission, compare stored inputs to that final capture. Missing preview
invalidation is a defect only where that preview claims to represent the current
inputs; the absence of an ORCA-style review mechanism is not a defect.

Use **Workflow description** for `<prefix>-Snn ...`; the Surfaces workflow label
is set by the app, e.g. `CP2K_NEB`. BandUPpy has no equivalent description field
in the inspected form; associate its UUID in the external report instead of
inventing a control or changing old node metadata. Keep PKs as local GUI lookup
conveniences and UUIDs as durable evidence.

An expected rejection passes only its input-guard subcase. Native normal
termination, AiiDA exit 0, a “ready” badge, or a plausible plot alone cannot pass
scientific acceptance. Record GUI, scientific, resource/storage and reopening
verdicts separately. Unsupported, unconfigured and unvalidated are distinct
coverage states. Stop dependent cases on failure; preserve every result.

Use [the report](report-template.md) for the independent maintainer audit. No
maintainer API inspection should be described as an observation made in Chrome.

The testing agent must also perform [DOC01: GUI help and chatbot documentation
review](help-review.md) during every selected case, including blocked paths.
Identify needed tooltips, examples, workflow explanations and result guidance;
prepare concrete source-backed corrections. Record reusable information for
both the AiiDAlab-level and in-app assistants, with their actual documentation
sources identified separately. Do not confuse drafted knowledge with deployed
help, indexed chatbot content or validated chatbot answers. Keep this work in
the current old-style GUI and the preparation boundary.


## Page layout: one page scrollbar

Search results, plots and logs must expand in the normal page flow. A long
result must not create a second scrollbar inside a notebook or widget output,
including when the page restores a saved `output_scroll` state. Input controls
such as text editors retain their normal scrolling.

Each canonical notebook starts with an uncaptured call to
`surfaces_tools.page_layout.apply_page_layout()`. The integration test checks
every tracked notebook, including search, export, submission and result pages.
Keep that first cell when adding a page.

Run `python -m pytest -q tests/test_page_layout.py`. The Chromium regression
also runs when Playwright and a browser are already available;
`PLAYWRIGHT_CHROMIUM_EXECUTABLE` can select an existing browser. It checks saved
scrolling wrappers, asynchronous output growth and page scrolling. It does not
install a browser.

In the manual suite, search a date range with more than one screen of results.
Scroll while pointing at the table and at a long plot/output: the page must
move, with no separate output scrollbar or inset scrolling border. Check the
same behavior after a new search and after reopening the page.
