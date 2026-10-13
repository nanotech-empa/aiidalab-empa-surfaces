# Maintained Surfaces preview assembly

This is the refreshed, unreleased test assembly on `integration/surfaces-v2.0.0a0`.
Its implementation comes from the canonical component PRs below. Published alpha
snapshots remain immutable; no new tag or production deployment is made here.
The existing package metadata remains 2.0.0a2 until the separate metadata-only
release preparation in nanotech-empa/aiidalab-empa-surfaces#321 is approved.

Python >=3.12 and AiiDA >=2.8,<3 are the approved migration targets. This refresh
is tested on Python 3.12.11/AiiDA 2.9.2 with independent SQLite state, without
changing the live application or profile.

## Exact source pins

| Component | Commit | Ownership and status |
| --- | --- | --- |
| Backend existing AQE5 assembly | `db55fee7af2f019c3e8df8fc754befd9b552e41d` | Current master plus canonical nanotech-empa/aiida-nanotech-empa#200 through #207, #208 and #228 |
| CP2K plugin preview | `bc927019f7b63f7882e2f2f3b165d51312ce0bb3` | Existing immutable preview pin retained; aiidateam/aiida-cp2k#231 still needs upstream review |
| CP2K SPM tools | `cd97fc02f49c172b21811e5da5389368d6ce5415` | Canonical nanotech-empa/cp2k-spm-tools#34, including the maintained unfolding geometry/path stack |
| Widgets base | `f2773b82017d503c40751f9c082f941eef83ee33` | Canonical aiidalab/aiidalab-widgets-base#820, origin-preserving structure handling |
| EMPA widgets | `8a85258bbb449ff99c1d0ee302fd4b6ae887c254` | Canonical nanotech-empa/aiidalab-widgets-empa#12, following runtime, sterics and graph validation components |
| Cubehandler | `f16f9c8685ea6379c2c89e93811c6ead71dd0f49` | Exact canonical modern-runtime revision from Surfaces #326/#327 |
| ALPS configuration (unchanged) | `3ef32074be647ac72393d0ffd26022a64e600d7f` | Existing nanotech-empa/aiidalab-alps-files#9 preview; remote SPM source alignment remains before deployment |
| EMPA setup (unchanged) | `b196b60a309f8f518e8bada6f8eb99575c135533` | Existing nanotech-empa/aiidalab-empa-setup#9 preview; no remote codes are changed here |

The local SPM-tools dependency above supplies the same preview geometry and path
helpers tested by the form. The ALPS/setup snapshots still describe their earlier
remote SPM preview. Their deployed executable/source alignment must be checked
before using this refresh for new scientific submissions on ALPS.

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
| [Surfaces #284](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/284) | `f3672076b0ec69943a7703415b05d83e4cb59449` |
| [Surfaces #285](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/285) | `435b0885fc361da8c5cf36126cc0902b2f4935d8` |
| [Surfaces #286](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/286) | `e4248fb194913bd6c40accbda9c954a6d3f33fd0` |
| [Surfaces #287](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/287) | `b7fa8ddb52ded3e42b01661179da05b45cbe300e` |
| [Surfaces #288](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/288) | `78d22ee520d501b9e76fe6e0fff10d0e501db818` |
| [Surfaces #289](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/289) | `b3a34cd9cc4095a2308d0bb5f4224373fc96856c` |
| [Surfaces #290](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/290) | `72a2be12673869b73e76085053b61aa129d6e182` |
| [Surfaces #292](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/292) | `1ede431475699070af77e1897404c1f2bb0e26b4` |
| [Surfaces #293](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/293) | `91167f655a312b68660900c22643729089a01b73` |
| [Surfaces #294](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/294) | `02219623e4ff868bfb9f522d1ee2b1affef63f5e` |
| [Surfaces #295](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/295) | `61024058d00c2d272192c7cbea01e740a0b15be1` |
| [Surfaces #296](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/296) | `5194818d9480c6854a500ae65a51c046d74960d2` |
| [Surfaces #297](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/297) | `98d459eb7b2d587b3cccdbedaedf442c95b15eda` |
| [Surfaces #298](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/298) | `c7363048a622feb837dbfb3d10430565c63f9554` |
| [Surfaces #326](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/326) | `26f747283b933204ee588a59b0be620cccbf592c` |
| [Surfaces #327](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/327) | `903bc891d0c10bad02baa370ee5fadab3d1399af` |
| [Surfaces #317](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/317) | `bdb369f952768b516b6d6e45a346a864cea1041f` |
| [Surfaces #318](https://github.com/nanotech-empa/aiidalab-empa-surfaces/pull/318) | `e2aa21e6ee307f21a86aeb0882be53d14d3ff381` |
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
108 tests pass. Current required repository hooks pass. Complete combined GUI
validation: 155 tests pass with the exact local backend and SPM source pins.
No new live scientific submission or browser session is claimed.

aiidateam/aiida-cp2k#234 proposes Python >=3.12/AiiDA >=2.8 and removes misleading
CI versions. Its native Docker jobs pass 59 tests each on Python 3.12.15 with
AiiDA 2.8.0 and 2.9.2; documentation and pre-commit.ci pass. This support-policy
proposal still requires upstream maintainer agreement.

Remaining CI review: the backend #202 example rerun still fails in four SPM
orbital/STM post-processing examples while the CP2K SCF stages complete.
Backend #208 has the same four failing examples; #204 coverage and #206 patch
coverage remain red. These results are not a fully green release validation.
