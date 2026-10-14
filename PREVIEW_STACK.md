# Proposed Surfaces 2.0.0a3 preview metadata

This is the refreshed, unreleased test assembly on `integration/surfaces-v2.0.0a0`.
Its implementation comes from the canonical component PRs below. Published alpha
snapshots remain immutable; no new tag or production deployment is made here.
This metadata-only preparation proposes version 2.0.0a3 on the tested parent
assembly. It requires review in nanotech-empa/aiidalab-empa-surfaces#321; this
branch does not publish a new tag or change the installed application.

Python >=3.12, AiiDA >=2.8,<3 and AiiDAlab >=26.9.0 are the approved migration
targets. The fresh cell-reader retest uses Python 3.12.11/AiiDA 2.9.2 and the
installed AiiDAlab 26.9.0 with independent SQLite state. The host version was
rechecked on 2026-10-13; this task did not upgrade it or change its profile.

## Exact source pins

| Component | Commit | Ownership and status |
| --- | --- | --- |
| Backend existing AQE5 assembly | `b6b4126d9740f327b378651ab8a57f3f44d1d745` | Current master plus canonical nanotech-empa/aiida-nanotech-empa#200 through nanotech-empa/aiida-nanotech-empa#207, nanotech-empa/aiida-nanotech-empa#208 and nanotech-empa/aiida-nanotech-empa#228 |
| CP2K plugin preview | `bc927019f7b63f7882e2f2f3b165d51312ce0bb3` | Existing immutable preview pin retained; aiidateam/aiida-cp2k#231 still needs upstream review |
| CP2K input tools | `027cc072012e373265ba8368973a4a7158e33a19` | Upstream cp2k/cp2k-input-tools#106 merge; PyPI 0.9.1 still has older Pint metadata |
| CP2K SPM tools | `09139b5c674275fd240e3c4541f275575e784891` | Canonical nanotech-empa/cp2k-spm-tools#36, retaining the merged geometry/path/unit stack and adding negative-threshold rejection |
| Widgets base | `f2773b82017d503c40751f9c082f941eef83ee33` | Canonical aiidalab/aiidalab-widgets-base#820, origin-preserving structure handling |
| EMPA widgets | `8a85258bbb449ff99c1d0ee302fd4b6ae887c254` | Canonical nanotech-empa/aiidalab-widgets-empa#12, following runtime, sterics and graph validation components |
| Cubehandler | `f16f9c8685ea6379c2c89e93811c6ead71dd0f49` | Exact canonical modern-runtime revision from Surfaces #326/#327 |
| ALPS existing assembly | `da216394b9c5dd84b0b57ef14ff3b1ff929fbbdb` | Current main/Ritom definitions plus canonical nanotech-empa/aiidalab-alps-files#11; new and old preview code paths remain separate |
| EMPA setup existing assembly | `dbc365c914845accffbca9751e69d99e6a3e0746` | Canonical nanotech-empa/aiidalab-empa-setup#10 on nanotech-empa/aiidalab-empa-setup#8; modern platform metadata and tested non-interactive registration |

The local test source and proposed remote SPM configurations both use
nanotech-empa/cp2k-spm-tools#36. The proposed ALPS wrappers have unique labels
and paths ending in `surfaces-v2-09139b5`. Earlier Daint checks at #34 passed
all four CLI `--help` calls plus CP2K cell-unit/angle fixtures. That tested
geometry/path implementation is unchanged at #36; the new threshold rejection
is covered by the local and CI regression suite. The verified uenv provides
Python 3.11.9/NumPy 2.1.2/SciPy 1.14.1/ASE 3.23.0/mpi4py 4.0.1, separately
from the app's Python >=3.12 runtime. Source review can proceed; code deployment
and a real ALPS app run remain separate production acceptance checks.

The updated combined tools assembly passes 132 tests; this GUI assembly passes
159. A fresh real localhost four-carbon graphene form/submit/result smoke also
passes with the exact tools pin and canonical backend package metadata supplied
only to the disposable test environment. It runs synchronously without a browser
or daemon and is a software-wiring check, not a converged-physics benchmark.

The AO overlap-threshold input is bounded below by zero in the owning
nanotech-empa/aiidalab-empa-surfaces#283 component and all maintained children.
Negative frontend widget-state updates are clamped to zero; zero and positive
values remain valid, without a finite positive upper limit. Four regression
cases exercise the actual widget and submitted SCF builder. This complements
negative-input rejection in nanotech-empa/cp2k-spm-tools#36. The complete GUI
assembly was revalidated on 2026-10-14 using independent AiiDA state.

Pint >=0.24.4,<0.25 reproduces the validated modern dependency chain. With
NumPy 2.0.0/Python 3.12.11, Pint 0.23 fails on removed `numpy.cumproduct`;
0.24 through 0.24.3 fail with current flexparser dependencies; 0.24.4 loads
and parses CP2K input successfully. The <0.25 upper bound is inherited from
the official input-tools source and aiida-pseudo metadata. Replace the exact
upstream source override with the first release containing cp2k/cp2k-input-tools#106.

The optional openBIS importer is supplied by aiidalab/aiidalab-eln#93 at
`01e977a67035dd39741b73e3458a919ddfe884fd`. Missing or older ELN installations
omit the importer without stopping notebook startup. The export companion is
nanotech-empa/aiidalab-openbis#123. No real openBIS network operation is claimed
by the importer-gate tests.

## Canonical Surface heads included

| Component | Exact head |
| --- | --- |
| [Surfaces #281](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/281) | `91a9f31d66656ca8a784588d6fbaeab60c1b950e` |
| [Surfaces #282](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/282) | `c75d9b4919ea0b378b45c4f911ccebdade828e29` |
| [Surfaces #283](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/283) | `46ca660e652335150e6eb5e48974630159958fa6` |
| [Surfaces #284](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/284) | `bb92c6fb52565fca8afdefcca065400a9acf774c` |
| [Surfaces #285](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/285) | `34027ac9397a5c537272cb4273a4d8dec8e2c4d7` |
| [Surfaces #286](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/286) | `ae97f3034ad68aa59bb7f4093ede3a40504783b7` |
| [Surfaces #287](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/287) | `111eff56040fa273705ea2432935d50b96b48b82` |
| [Surfaces #288](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/288) | `6e3dc84a84f557537f0c10e6300e48a2a7b3d202` |
| [Surfaces #289](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/289) | `fea4f247466da963cf05fddd60999df32bc3ef9f` |
| [Surfaces #290](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/290) | `4768c5db567ee6a5d6be57453df0517362cdf779` |
| [Surfaces #292](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/292) | `e00d7226cd9c2f68b53ea3d534b63fcdfdd391fe` |
| [Surfaces #293](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/293) | `2cbb6e18b85b1c06e56d66833967bf7163e85c04` |
| [Surfaces #294](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/294) | `a49c1737791a3f0b4667687d0ac49a5ce9580f6d` |
| [Surfaces #295](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/295) | `67d9a4d4ac2780dc421448d3c80e7c85ff1fd1b8` |
| [Surfaces #296](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/296) | `69ed3edc4ff4cb0a8e11700f316d0db6c5158e52` |
| [Surfaces #297](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/297) | `5aaf2983541ec7e3fe68804d1cd6d237e54a87d0` |
| [Surfaces #298](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/298) | `883d2273b62e3a7de77f6d77d0ba9ac8e6fb6aa5` |
| [Surfaces #326](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/326) | `570860f1943d418f813e0f70a388d89172b4a4f5` |
| [Surfaces #327](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/327) | `8d9103cbf85bf088346a8f75e4952fc65da0df1a` |
| [Surfaces #317](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/317) | `b352b171bdf313cff9ac6bd5d1a3625d070eb2e6` |
| [Surfaces #318](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/318) | `fb507952008e8c0d7f79283adf692aefb3032cb8` |
| [Surfaces #300](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/300) | `9a4663c0f88835ab971017327d409af46a7aa442` |
| [Surfaces #309](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/309) | `502b8235bdf3db94f62f2e78b56018c7b1adbcc7` |
| [Surfaces #322](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/322) | `1c468b61ff66b536709fe9cc8939cd156e2b7ce7` |
| [Surfaces #255](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/255) | `bf84cccb9490b1f1e095f76e3b0de037900af470` |
| [Surfaces #268](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/268) | `1dc8a5c75af9d2329673d9c98745bddc282d7ca5` |

Surfaces #291 is closed as superseded: current master already provides the
validated NEB resource synchronization, and #292 now targets #290 directly.
The general-PDOS and GW-reuse work remains deferred and unchanged.

## Interaction and validation

For CP2K unfolding, supply 1-based primitive basis atom indices or leave the
field empty and press **Check primitive cell**. Detection can take time and runs
only after that explicit press with an empty input. Edits and submission never
trigger detection. Review the populated indices before submission; changes to
the primitive vectors invalidate automatically detected selections. Submitted
builders always carry explicit atom indices, so no silent basis detection is
requested from the backend. Empty band paths select the validated lattice
default; explicit paths remain available. The viewer uses retrieved path data.

Earlier canonical final component validation: 143 tests passed. Backend combined assembly:
121 tests pass. Current required repository hooks pass. Complete combined GUI
validation: 159 tests pass with the exact local backend and SPM source pins.
A real localhost smoke test ran the notebook-generated builder through its
submission widget, executing synchronously with `run_get_node` in the isolated
profile. The existing `cp2k@localhost`, sparse-overlap and unfolding executable
definitions supplied CP2K 2023.1 and the source-pinned SPM CLIs. A four-carbon
graphene debug calculation completed OT SCF, diagonalization, serial sparse
overlap and unfolding; weights have shape (2, 18), and the real result notebook
loads the stored default G-K-M-G path. Explicit basis atoms bypassed detection.
This verifies software wiring with coarse test inputs, not converged physics or
a browser/daemon/ALPS submission.

aiidateam/aiida-cp2k#234 proposes Python >=3.12/AiiDA >=2.8 and removes misleading
CI versions. Its current native Docker jobs pass 59 tests each on Python 3.12.15
with AiiDA 2.8.0 and 2.9.2. Refreshed aiidateam/aiida-cp2k#231 uses the upstream input-tools fix
and Pint 0.24.4 in CI; both NumPy 2.5.4 Docker jobs pass all 66 tests.
Documentation and pre-commit.ci pass for both. Maintainer reviews remain pending.

Backend nanotech-empa/aiida-nanotech-empa#202 through
nanotech-empa/aiida-nanotech-empa#208 now pass every native, unit, patch/project
coverage and pre-commit.ci check at their recorded current heads. The full runs
exposed UCX selecting unsupported InfiniBand verbs on GitHub VMs; the canonical
examples job sets UCX_TLS=tcp,self,sm. Corrected runs pass all 37 cases, without
changing production HPC settings, scientific inputs or coverage thresholds.
The builder/error contracts in nanotech-empa/aiida-nanotech-empa#204 and
nanotech-empa/aiida-nanotech-empa#206 now have focused coverage.

Release review remains: upstream support-policy/parser review in
aiidateam/aiida-cp2k#234 and aiidateam/aiida-cp2k#231; scientific review of
nanotech-empa/cp2k-spm-tools#24, nanotech-empa/cp2k-spm-tools#32,
nanotech-empa/cp2k-spm-tools#33 and nanotech-empa/cp2k-spm-tools#34;
canonical app/backend approvals; and review/deployment validation of
nanotech-empa/aiidalab-empa-setup#10 and nanotech-empa/aiidalab-alps-files#11.
This test assembly does not authorize a merge, tag or production rollout.

## Merge preparation, 2026-10-14

The canonical SCF slice now declares a `scf-preview` extra for the exact common
backend and already validated input-tools/Pint fix. Its CI installs that extra.
Basic current-widget API support belongs to the first AutoRep component; the
following component retains stored-representation preservation and its round-trip
regression. The final app source and scientific notebook cells are unchanged.

Surfaces and Nanoribbons now use the same exact backend, widgets-base and EMPA
widget sources. This refreshed source passes all 159 Surface regressions with
those common sources; Nanoribbons passes all 25, EMPA widgets all 146, and the
updated SPM guard source all 136. Local test profiles and configuration are
independent of the running instance. No tag or deployment is created here.

The proposed ALPS code definitions now use the same SPM guard commit and new
`surfaces-v2-09139b5` labels. Earlier remote CLI/cell-reader tests used #34; the
new threshold guard has local/CI regressions, and the geometry/path implementation
is unchanged. A real ALPS app submission remains a deployment acceptance check.
