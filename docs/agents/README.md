# Helping a user with On-surface chemistry

The shared chat is embedded in the canonical submission, search, result and
structure-tool notebooks. `app_get_state` identifies the actual notebook and
selected structure. `app_update_draft` fills the actual controls, including the
declared nested scientific/resource/editor sections. Field keys inside compound
sections have numeric path segments; identify them using `controls.label` and
`option_labels`, not guessed keys. The current schema is authoritative. Re-read
it after structure, protocol, boundary, mode or section changes because the GUI
rebuilds controls. Preserve unrelated user edits.

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
