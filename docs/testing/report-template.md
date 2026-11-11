# Local GUI campaign report — fill only from observed evidence

Copy this template into a private campaign report outside the app-discovery
root. Do not store downloaded scientific files/screenshots in Git by default.
Keep original attempts and append retests; never replace a failure with a later
pass. Record times with timezone, normally **Europe/Zurich** for this user.

## Campaign authorization and environment

- Prefix / selected case IDs / authorization date:
- Canonical main/master commit per component and agreed PR inclusion decisions:
- Passed executable preflight receipt; expected-manifest and source fingerprints:
- Installed distribution versions/direct URLs versus actual imported paths/hashes:
- Exact reviewed unreleased qualification candidates and missing official releases:
- Relevant source changes since earlier runs; affected cases invalidated/requalified:
- Selected candidate(s) and explicitly deferred candidates:
- User-approved fixture/method/tolerance revision:
- Mac model/architecture/RAM; local container architecture/allocation, if used:
- Browser/version; actual app URLs and maintained notebook entry points:
- Surfaces revision plus dirty-source fingerprints:
- QE post-processing revision, if selected:
- Backend, CP2K, QE, BandUPpy, irrep, cp2k-spm-tools and widget versions relevant
  to the selected cases; mark unknown, not guessed:
- Actual local code and computer labels/UUIDs and evidence source:
- Confirmed execution slot outside ORCA; other active native jobs:
- CPU allocation: 1 node; CP2K 2/4/8 MPI tasks per node as selected; always
  1 thread per task (4 MPI ranks × 1 thread = 4 CPU cores). Record requested
  and native MPI/OpenMP counts; no additional BLAS investigation gate:
- Selected tier (Small / Medium / 6×6 unfolding); specifically agreed exceptions:
- Memory budget; requested vs monitored vs OS-enforced; evidence:
- Larger cases: user-selected **32 GiB total per active job**, recorded as an
  external budget; no overall CP2K GUI memory field. Actual host/container
  capacity, existing external enforcement (if any), sampled maximum RSS and
  headroom; do not describe this budget as automatically enforced:
- Walltime/logical-stage budget; retries/stop mechanism; evidence:
- Storage budgets and initial free space on each relevant host/container volume:
- Installation/configuration prerequisite gaps (no implicit permission to fix):

## Fixture manifest — one per selected system

| Field | Agreed value / source |
| --- | --- |
| Fixture name, formula, atom count | |
| New structure UUID / local PK | |
| Coordinate file and SHA-256; source/creation method | |
| Exact cell, PBC, units and electrostatic boundary model | |
| Atom order, fixed atoms, fragment partition, charge/spin | |
| Initial/final geometry identities; CV/restraint values and units | |
| Method, dispersion, basis/potential/pseudo identities and cutoffs | |
| SCF/geometry/BAND protocol and actual numerical thresholds | |
| Primitive vectors/matrix convention/basis indices and k path | |
| Native band count, occupations, sampling and energy-zero convention | |
| Proposed tolerances finalized before execution | |
| Expected logical stages/native jobs and maximum aggregate budget | |
| Explicit molecular 12 Å box; no isolated preparation; native-cell evidence | |
| Primitive-monolayer replication versus built-in slab convention; atom count | |
| PBE0 HF fraction/cutoff/per-rank memory; ADMM/primary data; actual spin | |
| SPM energy reference/window, heights, extent, probe model, grid and amplitude | |
| PDOS groups/projection convention, normalization, bands and export precision | |
| Density cube SCF sources, signs, cell/origin/voxel units and conservation target | |

## Case record — repeat for every case and retest

- Case ID / guide revision / original or retest number:
- Start/end time and timezone; GUI operator:
- Prerequisites satisfied, reused, missing or explicitly waived by user:
- Exact GUI actions, final-form capture and any edited/stale preview:
- Effective stored settings versus the final visible form:
- Parent UUID / local PK; template/reference/input/output UUIDs:
- Expected versus observed behavior and scientific quantities:
- Native convergence evidence (not just termination/exit code):
- New computation versus WFN restart versus stored-result reuse:
- Warnings, actual error text, screenshot/download references:
- Child job count, retries, native overlap and resource observations:
- Repository/work-directory growth; actual full-file/compact-file retention:
- Reopening route, same UUID/checksum and process-count observations:
- Completed-test browser tabs closed after recording the reopening route;
  active, pending user-review and unrelated tabs retained:
- Representation save action, node extras versus unchanged geometry; saved
  styles versus camera state that is not promised to persist:
- Defect reference, affected build, implementation colleague's supplied fix
  revision, and what must be retested; do not assign/pass based on hearsay:
- DOC01 help review: existing help consulted, confusion/missing explanation,
  proposed wording/location, evidence and reusable chatbot-knowledge topics;
  if no addition is needed, record why:

| Dimension | Verdict | Evidence / remaining limitation |
| --- | --- | --- |
| GUI path and input validation | NOT RUN | |
| Current-input/preview consistency | NOT RUN | |
| Native/scientific validation | NOT RUN | |
| CPU/memory/time/serial execution | NOT RUN | |
| Storage/provenance | NOT RUN | |
| Durable reopening | NOT RUN | |
| Duplicate prevention (separate safe subcase) | NOT RUN | |
| GUI help adequacy (DOC01; separate from science) | NOT RUN | |

Use **NOT RUN**, **PASS**, **FAIL**, **BLOCKED**, or **SKIPPED**. Additionally
record capability as **supported but unvalidated**, **unconfigured**, or
**unsupported/not exposed** where appropriate. An unavailable control cannot
be marked PASS; an expected rejection passes only the rejection subcase.
Separate a source-identified concern from a reproduced GUI defect.

## GUI help and chatbot documentation handoff

Complete [DOC01](help-review.md) alongside each selected case, even if the
scientific run is blocked. Do not leave this section out merely because no
calculation finished. Link concrete proposed wording or documentation patches;
distinguish required corrections from optional improvements.

| Finding / case / exact GUI location | Existing wording or missing help; user question | Classification / consequence | Proposed text and owning source | Evidence / revision / review date | Implementation and recheck status |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

- Documentation source and retrieval/integration route for the AiiDAlab-level
  assistant; mark unknown if not established:
- Documentation source and retrieval/integration route for the in-app assistant;
  do not assume it shares the first assistant's source:
- Canonical app-owned content location and links from GUI help:
- Scientific reviewer and unresolved claims/settings requiring review:
- Provisional campaign choices kept separate from general product guidance:

| Topic / question | Content drafted | Scientifically reviewed where needed | GUI help integrated/rechecked | AiiDAlab assistant source available / answer checked | In-app assistant source available / answer checked | Remaining gap |
| --- | --- | --- | --- | --- | --- | --- |
| | NOT RUN | NOT RUN | NOT RUN | NOT RUN / NOT RUN | NOT RUN / NOT RUN | |

For each later chatbot check, record exact question, actual answer, cited source
and revision, expected facts, discrepancies and verdict. Check each assistant
independently without triggering scientific actions. Neither prepared content
nor a fluent answer establishes scientific validity or current integration.

## Independent maintainer audit — by a person/session other than the GUI operator

- Reviewer, date/time, exact software revision and parent UUIDs:
- Method of read-only inspection, using the existing compatible runtime:
- Native SCF and geometry/BAND convergence, force/residual thresholds and warnings:
- Raw output versus parsed values, units, arrays, ordering and finite values:
- Endpoint identity/barrier symmetry, actual CVs/restraint energy, or fragment
  arithmetic, as appropriate to the case:
- Unfolding primitive mapping, raw weight checks/normalization assumptions,
  reference-band agreement and degeneracy/energy-zero handling:
- Molecular cell unchanged by inactive electrostatics; CH3 native spin, hybrid
  terms, cutoff and HF-memory settings versus whole-process memory:
- STM/AFM effective helper resource propagation, actual band count, numerical
  map validity, coordinates/units and height/energy reference:
- PDOS component sums, actual projection convention and CSV/plot correspondence:
- Full-resolution Δρ source geometry, pointwise arithmetic, density sign/unit
  convention and electron-integral residual; reduced-preview limitations:
- Actual scheduler request, wrapper behavior, native MPI/thread count, RSS
  samples and cadence/gaps. Distinguish a sampled maximum from a true peak;
  mark measurements absent rather than treating requests as enforcement:
- Child start/end times, actual retry count and aggregate elapsed/native time:
- Work directories, full WFN/AO/cube/QE save retention; retrieved file sizes,
  compact NPZ contents/checksums and absence of accidental large-file retrieval:
- Provenance/input/result identities unchanged; legitimate new nodes and known
  bookkeeping extras distinguished from immutable scientific attributes/links:
- No additional process from viewing/reopening; duplicate guard audit if selected:
- Verdict and limitations; discrepancies returned to the implementation colleague:

Do not run new calculations as an “audit”, mutate existing nodes, clear remote
folders, change services, or resume paused work. A browser operator must not
claim the maintainer's checks as GUI-observed evidence. If this independent
review is pending, scientific/resource acceptance remains pending.

## Final coverage ledger

| Case | Fixture | Original attempt UUID | Latest retest UUID | GUI | Science | Resources/storage | Reopen/duplicate | Remaining gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S00–S01 | | | | NOT RUN | N/A | NOT RUN | NOT RUN | |
| E01–E02 | | | | NOT RUN | Geometry only | NOT RUN | NOT RUN | |
| H01 | | | | NOT RUN | NOT RUN | NOT RUN | NOT RUN | |
| F01, T01 | | | | NOT RUN | NOT RUN | NOT RUN | NOT RUN | |
| D01 (D02 deferred) | | | | NOT RUN | NOT RUN | NOT RUN | NOT RUN | |
| S02–S04 | | | | NOT RUN | NOT RUN | NOT RUN | NOT RUN | |
| S05–S06 | | | | NOT RUN | NOT RUN | NOT RUN | NOT RUN | |
| S07–S08 | | | | NOT RUN | NOT RUN | NOT RUN | NOT RUN | |
| C01 | | | | NOT RUN | NOT RUN | NOT RUN | NOT RUN | |
| S09–S11 | | | | NOT RUN | NOT RUN | NOT RUN | NOT RUN | |
| B00–B04 | | | | NOT RUN | NOT RUN | NOT RUN | NOT RUN | |
| DOC01 (per selected case) | | | | NOT RUN | Content review only | N/A | NOT RUN | |

Expand grouped rows into individual selected cases. List separately: features
actually validated; prerequisites missing; unsupported paths; scientific claims
not tested; cases deferred for cost; defects awaiting fixes. XPS remains
**EXCLUDED**, not a pending case to resume from this report.
