# C02: shared cube preview, slices and bounded full-grid rendering

Needs completed C01 with accessible full cubes, the maintained shared-viewer
adapter included in the agreed candidate, and a compatible local renderer.
Missing source/code/resources means BLOCKED with clear feedback. Do not rerun
SCFs or replace unavailable native data. Render one job on one CPU and one
thread, maximum 600 s, with the native slot otherwise idle.

1. Open the adsorption result and click **View cube files**. Confirm the URL
   identifies its actual completed cubehelper, not the energy parent. Pending
   or unavailable outputs must not expose a stale link from a different workflow.
   Load another result, an invalid source and the original result to check reset.
2. Select ChargeDiff in the shared cube viewer. Record source UUID/files, full
   and preview shapes, cell, origin and boundary model. For an oblique cube,
   compare the reconstructed cell to the actual full-grid row vectors. If the
   retained preview is corrected from a qualified full source, require the
   derived notice and unchanged original files/nodes. CP2K periodic sampling
   and isolated ORCA sampling must remain deliberate and separately labelled.
3. In the 3D tab use equal-magnitude positive/negative isovalues with distinct
   colors. Record values/colors, camera, surface visibility and opacity. Change
   opacity and confirm the preview reflects it. A new cube must not inherit stale
   source identity or a previous render review.
4. In the 2D controls set a plane through the interface. Record center, normal,
   offset, extent, resolution, symmetric color range and signs. Change offset
   away from zero and verify the 3D plane and numerical slice move together,
   including any nonzero cube origin. Export the actual plane PNG through its
   GUI control and check axes, range and orientation against the displayed slice.
5. Save a named view, change it, reopen the same source and restore it. Check
   camera, signs/colors/range, plane, lighting and visibility against saved
   values. Keep display extras distinct from immutable science. A new source
   must not reuse an incompatible saved view silently.
6. Open Render, inspect full-grid source and the one-CPU/one-thread/600 s plan,
   and review it. Change a relevant source/display/resource control and require
   review invalidation. Restore/review once. Submit through the actual GUI only.
   Resolve uncertain submission by UUID. Do not create a duplicate to test a
   lock; confirm the button/state rejects repeated dispatch safely.
7. Audit the actual render input: qualified full source/checksum, row-vector
   cell/origin, camera orientation, signed isovalues/colors, opacity, lighting,
   visible surfaces and slice-plane settings. Preview and renderer use the same
   model and conventions despite different grid resolutions.
8. Open/download the nonempty render, compare those settings and plane alignment
   with the preview, then close/reopen the result by UUID. Viewing must not
   submit another render. Test missing full-source feedback in a disposable
   source selection without deleting data. Preserve files and original outputs.

PASS requires real plane/PNG export, shared preview alignment, one bounded render
and durable reopening with matching scientific geometry/display conventions.
FAIL includes shear, origin/plane mismatch, stale review/link, wrong source or
unintended extra jobs. A successful helper exit or an attractive image alone
does not pass. Record unsupported partial-axis sampling explicitly rather than
silently treating it as isolated or fully periodic.

Close completed-test tabs after recording result and reopening routes. Keep
active/user-review tabs. Use the [report template](report-template.md).
