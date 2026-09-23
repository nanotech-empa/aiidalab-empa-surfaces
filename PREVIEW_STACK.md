# Surfaces 2.0.0a3 preview stack

This prerelease makes the reviewed-but-unmerged Empa feature stack available
without changing the stable `1.1.0` installation path. Dependencies are pinned
to immutable Git commits so every test container receives the same code.

## Components

| Repository | Source branch | Commit used here | Content |
| --- | --- | --- | --- |
| `aiidateam/aiida-cp2k` via `cpignedoli/aiida-cp2k` | `integration/surfaces-v2.0.0a0` | `bc927019f7b63f7882e2f2f3b165d51312ce0bb3` | PRs #230, #231, and #232 |
| `nanotech-empa/cp2k-spm-tools` | `integration/surfaces-v2.0.0a0` | `8bb2a13fd73af12c7ee5ec1eb688a3b07ed89276` | latest `main`, PRs #21 and #27, plus the tested #24 adaptation to merged #30 |
| `aiidalab/aiidalab-widgets-base` via `cpignedoli/aiidalab-widgets-base` | `feature/preserve-eln-structure-origin` | `f2773b82017d503c40751f9c082f941eef83ee33` | upstream `master` at `4d0076f78ed9316e499aa7ae2fb704f087ef5cb9` plus PR #820, including the provenance test typing fix |
| `nanotech-empa/aiida-nanotech-empa` | `integration/surfaces-v2.0.0a0-aqe5-current` | `1a1ce00bfa62913b2665a357374aa2be759eec8e` | current `master`, the CP2K preview stack, PR #216 for AQE5, and PR #208 for BandUPpy 1 |
| `nanotech-empa/aiidalab-alps-files` | `integration/surfaces-v2.0.0a0` | `3ef32074be647ac72393d0ffd26022a64e600d7f` | PR #9; complete tested Daint preview code matrix and exact `cp2k-spm-tools` source ref |
| `nanotech-empa/aiidalab-empa-setup` | `integration/surfaces-v2.0.0a0` | `b196b60a309f8f518e8bada6f8eb99575c135533` | PR #9; Python 3.12 preview gate and non-interactive complete code creation |

The Surfaces branch contains PR #280, the feature commits represented by PRs
#281 through #298, PR #300, PR #309, the approved final head of PR #312, and the SCF structure-manager fix from
PR #314. Alpha 2 additionally includes PR #317 (PDOS SVG export and incomplete
selection handling) and PR #318 (the optional shared openBIS structure importer).
The importer is skipped when `aiidalab-eln` is absent or an older installation
does not provide `OpenbisStructureImporterWidget`.

## Alpha 2 dependency update

Python >=3.12, AiiDA >=2.8,<3, and the widgets-base ipywidgets 8.1 stack are
required. The CP2K, SPM, nanotech-empa, ALPS, and setup pins remain those tested
for alpha 1; this update does not change their scientific workflow code.

Widgets-base is pinned to the exact updated head of
[PR #820](https://github.com/aiidalab/aiidalab-widgets-base/pull/820), not to a
moving `master` or PR branch. Upstream `master` alone does not preserve the ELN
origin through structure edits. PR #820 transfers that metadata between AiiDA
structure extras and ASE `info`, without changing QE workflows.

Using this widgets-base revision exposed two failures in
`python -m pytest -q tests/test_auto_representations.py --tb=short`: AutoRep still
referenced the removed `DEFAULT_REPRESENTATION` attribute. Alpha 2 adapts AutoRep
to the current representation constructor, `ballstick` type, and encoded style
IDs, preserving the existing atom selections.

The openBIS importer remains optional. For a reproducible openBIS test setup,
install the shared importer from
[aiidalab-eln PR #93](https://github.com/aiidalab/aiidalab-eln/pull/93):

```bash
python -m pip install "aiidalab-eln @ git+https://github.com/aiidalab/aiidalab-eln.git@01e977a67035dd39741b73e3458a919ddfe884fd"
```

## Alpha 3 AQE5 update

Alpha 3 updates `aiida-nanotech-empa` to a two-parent integration of the
surfaces CP2K preview and the head of PR #208. That head contains current
`master`, PR #216 for aiida-quantumespresso 5, and the BandUPpy 1 workflow.

PRs #216 and #208 remain the canonical review and merge targets. The combined
commit pinned here is only an immutable alpha-test assembly; its integration
branch is not a destination for the component changes.

The exact commit pin remains necessary until the upstream AQE5 changes and the
CP2K preview stack are released. New QE workflows target AQE5; compatibility
with historical AQE4 results remains a viewer/import concern.

## Composition decisions

- The revised PR #196 was applied on current `aiida-nanotech-empa` `master`.
  Only the actual #197-#207 feature commits were then replayed. This preserves
  the AiiDA 2.8 restart/pause work already merged in PR #209 instead of
  reintroducing the stack's older copy.
- The accepted upstream implementation of widgets-base PR #525 is used. The
  older local completion branch is not merged over it.
- Surfaces commits that modified the deleted `submit_reactions.ipynb`, or its
  obsolete inline NEB resource helper, were omitted because current `master`
  provides the newer split notebooks and shared helper from PRs #308-#311.
- PR #312 is the authoritative NEB replica editor. The independent
  `ForcePeriodicWidget` and AutoRep additions were reapplied after taking its
  final files.

## ALPS isolation and rollback

The preview configuration creates isolated STM, overlap, sparse-overlap, and
unfolding labels whose names retain the tested `surfaces-v2.0.0a0` namespace.
It uses a separate remote source directory
and does not relabel or overwrite the existing `1.5.0` codes.

The stable Surfaces release remains `1.1.0`. The `2.0.0a3` alpha is opt-in. If a
defect is found, install `1.1.0` again and publish a corrected `2.0.0a4`; do not
move or rewrite an existing alpha tag.
