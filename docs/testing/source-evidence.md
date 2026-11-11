# Source ownership and qualification evidence

This file maps the maintained sources. It is not an installation fingerprint or
evidence that the tested installation contains their current fixes. Reconstruct
the expected release manifest from agreed canonical PR decisions, then compare
declarations, distribution metadata, import paths, exact content and reviewed
overlays with the executable installation preflight. Never bless an installed
snapshot as the expected release.

| Component | Canonical source | Relevant contract |
| --- | --- | --- |
| Surfaces | `nanotech-empa/aiidalab-empa-surfaces`, master | Tracked submit/view notebooks and `surfaces_tools` widgets/utilities |
| Workflows/plugins | `nanotech-empa/aiida-nanotech-empa`, master | CP2K models, attempts/resources, serial adsorption, NEB partial status and helper staging |
| Shared viewer | `nanotech-empa/aiidalab-widgets-empa`, main | Maintained cube/plane/render widget, saved views and lifecycle |
| Cube processing | `nanotech-empa/cubehandler`, main | Oblique row-vector grids, reduction and renderer CLI contract |
| SPM/unfolding | `nanotech-empa/cp2k-spm-tools`, main | Native WFN processing, reciprocal paths, grids and unfolding |
| Separate QE app | `nanotech-empa/aiidalab-empa-qe-postprocess`, main | QE templates, explicit primitive-reference model, BandUPpy controls/results |
| External core | Official immutable releases | Core, AiiDAlab, CP2K/QE/pseudo plugins and widgets-base compatibility |

Do not copy older composed runtime files onto these branches. Develop focused
fixes from the latest canonical default branch, preserve newer validation and
require the affected checks before merge. Intentional held/deferred PRs remain
excluded. Own development candidates and required future official releases are
recorded separately in the campaign dependency-release ledger.

Only the primary browser operator can pass the manual cases. Record the exact
canonical commit, installed package metadata and actual imported source for
each run, plus the manifest/source fingerprint and preflight receipt. Requalify
affected cases after relevant changes. Historical reports and approved cleanup
receipts remain evidence of their original builds, not current reopening claims.
