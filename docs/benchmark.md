# CSCS CP2K benchmark preview

Open **CSCS CP2K benchmark** in the Surfaces launcher, select a structure and CP2K code,
then edit the node, MPI-task and OpenMP-thread lists. Each MPI-task count per node
must be divisible by **GPUs per node**; invalid lists are rejected in both the app
and workflow. The GPU field is a scheduling rule, not a GPU-allocation command;
the selected computer/code supplies GPU allocation and binding.

Initial defaults: nodes 1–10, MPI tasks per node 4/8/12/16, OpenMP threads per task
2/4/6/8, four GPUs per node: 160 combinations before CPU-capacity filtering.
The previous perfect-square-total-MPI restriction has been removed.
The legacy workflow input max_tasks_per_node is still supported when no explicit
list_tasks_per_node is supplied.

Use **Preview combinations** before **Submit benchmark**. The draft uses the
computer's default MPI processes per machine as its CPU-capacity limit. Review
that setting for the selected computer. Each calculation allocates the selected
number of CPUs per MPI task and sets OMP_NUM_THREADS after the code's prepend
text, so the requested thread count takes precedence. Executable wrappers must
also respect that value. All selected cases are submitted together.

The CP2K protocol is unchanged: periodic PBE+D3, OT CG, no input WFN.
The timing metric is the sum of the elapsed times for OT iterations 3 (CG) and
4 (LS), not total wall time. A calculation that fails or lacks either iteration
has no usable timing. Failed cases remain visible; an entirely failed grid
returns an error and a readable report. The viewer uses the smallest successful
node count as its speedup reference. A report multiplier is only an estimate.

Submission produces monitor and results links. Results can also be opened in
view_benchmark.ipynb using a UUID or PK, or through the Surfaces structure search.
Existing benchmark result dictionaries remain readable.

This port belongs to the existing drafts:
- nanotech-empa/aiida-nanotech-empa#182 (feature/benchmarking)
- nanotech-empa/aiidalab-empa-surfaces#251 (benchmarks)

The GUI requires the companion workflow with list_tasks_per_node; it is not yet
part of a published release. Validation uses the active Python 3.12 / AiiDA 2.9.2 /
ipywidgets 8 environment. The instance's AiiDAlab 26.6.0 remains below the agreed
26.9.0 platform baseline; platform and repository-wide runtime metadata alignment
are separate remaining work. This port does not upgrade the shared platform.
