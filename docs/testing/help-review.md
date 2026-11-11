# DOC01: GUI help and assistant-source review

Perform this alongside every manual case, including blocked cases. Record the
actual control, wording and source revision before deciding whether help should
be added. Keep a software fix, a scientific explanation and an unsupported
feature distinct. Do not add implementation details to ordinary user steps.

Check these recurring questions against the actual app and native inputs:

- Structural connectivity versus effective calculation periodicity. Explain
  NONE, each 1D axis, 2D planes and XYZ, active Apply/Undo, valid Poisson choices
  and the fact that an inactive suggestion changes nothing.
- Molecular 12 Å periodic boxes versus isolated preparation. The campaign uses
  the former and makes no isolated-limit convergence claim.
- MPI ranks versus threads. One node with four MPI tasks and one thread per task
  allocates four cores. There is no total-memory entry in these CP2K panels.
- Saved representations versus camera, background and visibility. Explain the
  official widgets-base 3.0.1 same-source reload limitation and pending PR832.
- Standard/Low protocols, explicit PBE0 basis/potential changes, HF cutoff and
  per-rank HF memory. Debug and custom threshold controls are outside the suite.
- NEB local interpolation factors, endpoint-inclusive images, one-group resources,
  restart inheritance, partial plots, timeout, electronic and band convergence.
- Replica target versus actual CV, restraint energy, units, termination and
  multiple-CV concurrency. Adsorption frozen/relaxed references, partitioning,
  serial stages, attempts, deformation signs and charge-density difference.
- STM/AFM axes, energy zero, scan/extrapolation heights, constant-current missing
  crossings, numerical versus realistic-image validity, archive precision and
  one-map/repeat export behavior.
- PDOS compute/overlap window versus stored-spectrum display range, reference
  shifts, all/fragment projections, normalization and styles.
- Cube source provenance, reduced versus full grids, oblique cell/origin, signed
  colors, sampling, slice export, saved views and rendering resources. A render
  is a new bounded helper job; opening a stored preview should not submit one.
- CP2K unfolding versus the separate QE/BandUPpy workflow. Explain the primitive
  basis, matrix convention, permitted folded points, weights and reference model.

For each finding, propose a short full sentence at the control or a linked
app-owned help page. Record the canonical owner, evidence and any scientific
review still needed. Do not treat a drafted page as installed GUI help.

Use the [assistant-source procedure](assistant-sources.md) before claiming that
either the AiiDAlab-level or in-app chatbot can retrieve the information. Test
their answers separately, with exact questions, source citations and revisions.
Do not trigger calculations or external publication during an answer check.

PASS requires adequate source-backed help for the observed action. FAIL means
misleading or missing help with a concrete user consequence. BLOCKED means the
relevant source or assistant route cannot be established. Content preparation,
scientific review, GUI integration, retrieval and answer quality each have their
own status in the [report](report-template.md).
