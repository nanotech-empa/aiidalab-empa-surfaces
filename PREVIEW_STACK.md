# Maintained Surfaces preview assembly

This is the refreshed, unreleased test assembly on `integration/surfaces-v2.0.0a0`.
Its implementation comes from the canonical component PRs below. Published alpha
snapshots remain immutable; no new tag or production deployment is made here.
The existing package metadata remains 2.0.0a2 until the separate metadata-only
release preparation in nanotech-empa/aiidalab-empa-surfaces#321 is approved.

Python >=3.12, AiiDA >=2.8,<3 and AiiDAlab >=26.9.0 are the approved migration
targets. Tests use Python 3.12.11/AiiDA 2.9.2 and a temporary AiiDAlab 26.9.0
package with independent SQLite state. The shared host still has AiiDAlab 26.6.0;
this refresh does not upgrade it or change its profile.

## Exact source pins

| Component | Commit | Ownership and status |
| --- | --- | --- |
| Backend existing AQE5 assembly | `b6b4126d9740f327b378651ab8a57f3f44d1d745` | Current master plus canonical nanotech-empa/aiida-nanotech-empa#200 through #207, #208 and #228 |
| CP2K plugin preview | `bc927019f7b63f7882e2f2f3b165d51312ce0bb3` | Existing immutable preview pin retained; aiidateam/aiida-cp2k#231 still needs upstream review |
| CP2K input tools | `027cc072012e373265ba8368973a4a7158e33a19` | Upstream cp2k/cp2k-input-tools#106 merge; PyPI 0.9.1 still has older Pint metadata |
| CP2K SPM tools | `cd97fc02f49c172b21811e5da5389368d6ce5415` | Canonical nanotech-empa/cp2k-spm-tools#34, including the maintained unfolding geometry/path stack |
| Widgets base | `f2773b82017d503c40751f9c082f941eef83ee33` | Canonical aiidalab/aiidalab-widgets-base#820, origin-preserving structure handling |
| EMPA widgets | `8a85258bbb449ff99c1d0ee302fd4b6ae887c254` | Canonical nanotech-empa/aiidalab-widgets-empa#12, following runtime, sterics and graph validation components |
| Cubehandler | `f16f9c8685ea6379c2c89e93811c6ead71dd0f49` | Exact canonical modern-runtime revision from Surfaces #326/#327 |
| ALPS existing assembly | `39711c167a0d5d6109db8120b895412a8af181dd` | Current main/Ritom definitions plus canonical nanotech-empa/aiidalab-alps-files#11; new and old preview code paths remain separate |
| EMPA setup existing assembly | `dbc365c914845accffbca9751e69d99e6a3e0746` | Canonical nanotech-empa/aiidalab-empa-setup#10 on #8; modern platform metadata and tested non-interactive registration |

The local and new remote SPM configurations both use tools #34. The new ALPS
wrappers have unique labels/paths ending in `surfaces-v2-cd97fc0`; four modules
load successfully in Daint's configured uenv from an independent temporary
source checkout. Installed remote codes were not changed. That uenv still uses
Python 3.11.9; source deployment and an ALPS app run remain separate release
checks, and the experimental configuration PR stays draft.

Pint >=0.24.4,<0.25 reproduces the validated modern dependency chain. With
NumPy 2.0.0/Python 3.12.11, Pint 0.23 fails on removed `numpy.cumproduct`;
0.24 through 0.24.3 fail with current flexparser dependencies; 0.24.4 loads
and parses CP2K input successfully. The <0.25 upper bound is inherited from
the official input-tools source and aiida-pseudo metadata. Replace the exact
upstream source override with the first release containing #106.

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
| [Surfaces #283](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/283) | `9e2a2b28833535ffb0133e27bab8edf36d614d0e` |
| [Surfaces #284](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/284) | `8e270af219ae7a1ac2d70eb3ed323c8f03c1fe53` |
| [Surfaces #285](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/285) | `1ecd470bfd1228be603c2d34a426924bd73e017e` |
| [Surfaces #286](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/286) | `7167b437389740e9efa353d7c8ccad59421f539f` |
| [Surfaces #287](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/287) | `dc2b777a62265cc80fc2d04b45a90337236fdfe0` |
| [Surfaces #288](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/288) | `94ddd38aa8179d41e76d4a7bc90954e6fbe29496` |
| [Surfaces #289](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/289) | `6021e6ecc2a3866d20767581c8addd0eb452a285` |
| [Surfaces #290](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/290) | `57fee2d9a8c48f1670b6f55e4b4edc0fd3d51b68` |
| [Surfaces #292](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/292) | `d8dc8828858c6f51925d9cfc8b4336e0a6eddaf2` |
| [Surfaces #293](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/293) | `e91d8c4bf5495aeeedc3897d7fc5c8aced764c67` |
| [Surfaces #294](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/294) | `2e7d0af0adba2e5a3360a72fefe4f4f9b0fb4fc2` |
| [Surfaces #295](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/295) | `9787563e93f40ffcbc1ca3cb4990acc1cf17436d` |
| [Surfaces #296](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/296) | `2b023aa322bb5d4a6fb4d608542017672583278c` |
| [Surfaces #297](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/297) | `2fb4e6e28a17f1cfa365eb52ae080af0a95ff62c` |
| [Surfaces #298](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/298) | `4980aa37db6456af10cdff05270c54f2c9a7df51` |
| [Surfaces #326](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/326) | `968c9c17cf0ee4edf8b4eae1e05efdfe740006cf` |
| [Surfaces #327](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/327) | `57962a152122d5b4037d672ce37bb7e82a9364ef` |
| [Surfaces #317](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/317) | `acc30d09113914563c7dc30a1d7632c466cdaec1` |
| [Surfaces #318](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/318) | `6344f86fd6db2fe8ca1a41462eb811e991b6bcfc` |
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

Canonical final component validation: 143 tests pass. Backend combined assembly:
121 tests pass. Current required repository hooks pass. Complete combined GUI
validation: 155 tests pass with the exact local backend and SPM source pins.
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
with AiiDA 2.8.0 and 2.9.2. Refreshed #231 uses the upstream input-tools fix
and Pint 0.24.4 in CI; both NumPy 2.5.4 Docker jobs pass all 66 tests.
Documentation and pre-commit.ci pass for both. Maintainer reviews remain pending.

Backend #204/#206 coverage checks now pass with real builder/error-path tests.
The #202/#208 full runs exposed UCX selecting unsupported InfiniBand verbs on
GitHub VMs; their canonical CI sets UCX_TLS=tcp,self,sm. The current #208
full run passes all 37 native examples and every unit, coverage and hook check.
The #202 full run and later stack runs are pending; #202 project coverage is
not final until its example upload finishes. #203 is already fully green.
