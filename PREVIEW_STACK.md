# Surfaces 2.0.0a1 preview stack

This prerelease makes the reviewed-but-unmerged Empa feature stack available
without changing the stable `1.1.0` installation path. Dependencies are pinned
to immutable Git commits so every test container receives the same code.

## Components

| Repository | Integration branch | Commit used here | Content |
| --- | --- | --- | --- |
| `aiidateam/aiida-cp2k` via `cpignedoli/aiida-cp2k` | `integration/surfaces-v2.0.0a0` | `bc927019f7b63f7882e2f2f3b165d51312ce0bb3` | PRs #230, #231, and #232 |
| `nanotech-empa/cp2k-spm-tools` | `integration/surfaces-v2.0.0a0` | `8bb2a13fd73af12c7ee5ec1eb688a3b07ed89276` | latest `main`, PRs #21 and #27, plus the tested #24 adaptation to merged #30 |
| `aiidalab/aiidalab-widgets-base` via `cpignedoli/aiidalab-widgets-base` | `integration/surfaces-v2.0.0a0` | `eb3672a7471f658371a38927d1563d78203ac028` | latest `master`, PRs #766, #768, #769, and #770; #525 is already merged upstream; validation import-order fixes |
| `nanotech-empa/aiida-nanotech-empa` | `integration/surfaces-v2.0.0a0` | `45c4307a9db39ac5debc7a2b87f0ad869a425c17` | revised PR #196 and the feature commits from #197 through #207; declared Ruff formatting applied |
| `nanotech-empa/aiidalab-alps-files` | `integration/surfaces-v2.0.0a0` | `3ef32074be647ac72393d0ffd26022a64e600d7f` | PR #9; complete tested Daint preview code matrix and exact `cp2k-spm-tools` source ref |
| `nanotech-empa/aiidalab-empa-setup` | `integration/surfaces-v2.0.0a0` | `b196b60a309f8f518e8bada6f8eb99575c135533` | PR #9; Python 3.12 preview gate and non-interactive complete code creation |

The Surfaces branch contains PR #280, the feature commits represented by PRs
#281 through #298, PR #300, PR #309, the approved final head of PR #312, and the SCF structure-manager fix from
PR #314.

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

The stable Surfaces release remains `1.1.0`. The `2.0.0a1` alpha is opt-in. If a
defect is found, install `1.1.0` again and publish a corrected `2.0.0a2`; do not
move or rewrite either alpha tag.
