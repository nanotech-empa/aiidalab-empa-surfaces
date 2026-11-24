# Helping a user with On-surface chemistry

The shared chat is embedded in the canonical submission, search, result and
structure-tool notebooks. `app_get_state` identifies the actual notebook and
selected structure. `app_update_draft` fills the actual controls, including the
declared nested scientific/resource/editor sections. Field keys inside compound
sections have numeric path segments; identify them using `controls.label` and
`option_labels`, not guessed keys. The current schema is authoritative. Re-read
it after structure, protocol, boundary, mode or section changes because the GUI
rebuilds controls. Preserve unrelated user edits.

## Open the matching page

Read the app's `workflow_pages` through `list_apps`, then use `open_app` with the
requested workflow. The source calculation and destination task are independent:
an optimized geometry is a valid source for STM or PDOS without reopening geometry
optimization. Carry its UUID as `process_uuid` and use the returned link directly.

| Requested task | Workflow | Notebook |
| --- | --- | --- |
| Geometry optimization | `geometry_optimization` | `submit_geometry_optimization.ipynb` |
| Single-point SCF | `scf` | `submit_scf.ipynb` |
| NEB | `neb` | `submit_neb.ipynb` |
| PDOS | `pdos` | `submit_pdos.ipynb` |
| STM / AFM / SPM | `stm` (alias `spm`, `afm`, `hrstm`) | `submit_spm.ipynb` |
| STM results | `stm_results` | `view_stm.ipynb` |
| PDOS results | `pdos_results` | `view_pdos.ipynb` |
| NEB results | `neb_results` | `view_neb.ipynb` |

The manifest also declares the other supported submission, search, result and
structure-tool notebooks. XPS links only to its Work in progress guide. On an
SPM form select the intended STM/AFM mode through its actual controls. Inspect
the saved optimized structure and original spin guess, load that structure and
verify the requested draft's charge, UKS, multiplicity and magnetic atoms; a page
link does not transfer those settings automatically. Never submit during a handoff.

When asked to prepare/fill a calculation, perform the requested draft edits and
read them back. `draft_actions` lists current edit/apply/estimate controls;
`app_prepare_action` invokes one with a fresh revision. `execution_actions`
lists the actual submission or other execution button. Review the complete
science/resources through `app_review_action`, then `app_execute_action` presents
the user's execution button. No submission occurs from draft edits or a pending
action. After execution, inspect messages and `submitted_processes` UUIDs; a
callback return alone is not evidence of a submitted or successful calculation.

## Load and prepare geometry

Use `app_search_structures` to find AiiDA StructureData/CifData by label or
description, and `app_load_structure` with the intended UUID. Empty searches
list recent structures with pagination. Use shared process inspection to locate
a calculation's actual output structure; do not substitute its input by assumption.

`app_import_structure` accepts supplied text files through the existing standard
reader or CDXML importer when that notebook provides it. Uploading CDXML in its
GUI tab is also supported. Inspect `context.cdxml`, select the drawing fragment
through `cdxml.structure_selector`, fill relevant repeat/hydrogenation controls,
then call `app_cdxml_action(action="create_model")`. Read conversion messages and
the resulting formula/cell. Upload alone does not create the final model. Do not
enable geometry cleanup or steric relief without a scientific reason agreed
with the user. Unsupported importer routes remain available in their existing GUI.

The declared geometry-editor fields and draft actions use the same GUI callbacks.
For atom operations use `app_select_atoms` with **one-based** indices, check the
read-back selection, set the displacement/angle or cell controls, and invoke the
named draft action. For a molecule on a slab, set the Build slab surface and size,
then use its Add slab action and inspect the resulting geometry. Never infer the
correct adsorption site/orientation or periodicity from the surface name alone.

The input-details sections expose charge, spin, constraints, cell/boundary,
NEB/replica/phonon and other controls as applicable. Dynamic rows use the listed
draft controls for add/apply/remove. Re-read after each structural change. For
CP2K functional/basis/HFX and effective electrostatics, use the installed
[DFT/boundary guide](../cp2k-electrostatics.md); a visual vacuum box alone does not
establish isolated electrostatics. The code selector's current UUID options and
resource widgets are available to fill; never invent a configured code.

## Orient from the user's goal

- Prepare a molecule/slab or combine them in submit_geometry_optimization.ipynb.
  Inspect geometry, cell, periodicity, selected CP2K Code, protocol, charge/spin,
  constraints, DFT settings, nodes/tasks/threads and wall time. Review wavefunction
  reuse before the existing Submit action. An initial spin guess is not proof
  of a magnetic ground state. Record the returned AiiDA UUID.
- Reuse a compatible completed geometry/SCF for submit_scf.ipynb,
  submit_pdos.ipynb or submit_spm.ipynb. Check the actual native outputs and
  required auxiliary Codes. Postprocessing availability does not imply every
  historical calculation retained its necessary files.
- For STM/orbitals/AFM inspect the saved source, cell, height, bias and spin
  conventions in the corresponding submit/view notebook. Rendering a reduced
  cube does not replace a full scientific cube or validate vacuum extrapolation.
- For reactions use submit_neb.ipynb or submit_replica_chain.ipynb, checking
  endpoint identity, atom order, images, constraints and resource budget. Existing
  paths and each image's status matter; a plotted path alone is not convergence.
- To continue a terminated EMPA CP2K NEB from Home, call
  `review_neb_continuation` for each parent. Check its saved band and allocation,
  then use its review ID with `submit_aiida_rerun` for user confirmation.
  An ordinary rerun starts from the recorded inputs. A paused process can resume
  through the process controls.
- For an editable continuation, use `open_app` with workflow `neb` and the
  source UUID. On an empty form, load the parent input structure with
  `app_load_structure` to display the NEB controls. Set `neb.restart_from`
  to the parent PK and verify the saved replica count in `context.neb`.
  Apply the parent replica count and `neb.replicas_per_group` before the node
  count. Check protocol, constraints, band settings and the effective allocation
  against the parent before reviewing submission. Read `context.neb.input_validation`
  and `changed_parent_settings`. An empty list for each of `sys_params`,
  `dft_params` and `neb_params` verifies equality with the parent, including
  the constraints. Check code, protocol and allocation separately. With 15 replicas, 40 nodes,
  8 ranks per node and NPROC_REP of 64, the original 320 ranks run five groups
  of three replicas. One replica per group makes the form adjust 40 nodes to 45.
- Use the matching adsorption-energy, phonon, NICS or other notebook only when
  the installed workflow and configured Codes support the intended calculation.
  XPS research is separate: xps_guide.ipynb does not authorize resuming old pilots.

## Submission, results and openBIS

Use search.ipynb and the matching view notebook. The common chat can search
AiiDA and inspect reports with bounded read tools. Explain a failed/paused job
from its actual evidence. Preserve running research jobs and existing nodes.
A submitted input is authoritative; later edited form fields do not describe it.

The openBIS structure importer is a source-selection route. Use typed material
searches to identify the correct concept/sample and explicit identifiers rather
than selecting an arbitrary duplicate name. `openbis_relationships` can follow
sample preparation or related simulations; inspect missing-link/truncation flags.

For simulation publication, open the shared openBIS import/export notebook with
`openbis_open_gui(workflow="export_simulations")`, select the stored calculation
UUID, project and collection, review supported result cards/structures/archives,
then press **Export**. A navigation link does not prefill fields or publish.
Inspect unsupported-result messages rather than fabricating an object. For an
archive import use the Import simulations GUI and inspect the receiving AiiDA
record; do not call the old chatbot's import function from conversation text.

## Read the effective CP2K inputs

In the shared chat, inspect `app_get_state.context.electrostatics`. The
`proposal` describes the dropdowns. The `effective` axes and solver describe
the input used for submission, including inherited NEB settings. On an ordinary
fresh form a molecule can intentionally use XYZ until Apply selected settings.
For advice, report both states and leave the form alone. For a request to set
isolated boundaries, explicitly set `electrostatics.apply_selected` to true
through `app_update_draft`, then read the effective axes and solver back.
Report a blocked state if the widget rejects the selection.

For resource advice read the selected Code UUID and `context.cp2k_resources`
when available. Use `inspect_cp2k_resources` to compare the Code with the CSCS
Daint recipe. Check memory per MPI rank for hybrid calculations. A request to
rerun a stored workflow with a different structure goes through the common
native AiiDA rerun tools and can be made from any chat.

### Stored STM maps into openBIS microscopy

In `view_stm.ipynb`, use the visible series and energy selection, then
`app_prepare_export` to open the shared review. Inspect `stm_selection`,
`microscopy_review`, destination fields and the `export_stm_microscopy` action.
Choose existing laboratory destination/material references with the normal
review controls. Use `app_review_action` and the user's confirmation before
`app_execute_action`; preparation must never publish. A changed revision requires
another review. Inspect final dataset IDs/preview status and report incomplete
steps honestly. Do not bypass the common exporter with direct server writes.
See [the user guide](../stm-microscopy-export.md) for units, original data,
oblique-grid interpolation and retry behavior. Never describe a native signal
as an absolutely calibrated experimental current.
