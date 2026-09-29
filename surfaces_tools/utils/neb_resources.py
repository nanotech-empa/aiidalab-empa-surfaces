"""Derivation of the CP2K NEB replica/group/task layout.

CP2K describes ``MOTION/BAND/NPROC_REP`` as "the number of processors to be
used per replica environment".  CP2K then runs

    n_groups = total_tasks / nproc_rep

replica environments concurrently, each of which works through
``n_replica_per_group`` replicas in turn.  Inverting that, a layout which
keeps the whole allocation busy is

    n_groups     = ceil(n_replica / n_replica_per_group)
    nproc_replica = total_tasks // n_groups

Note there is no second division by ``n_replica_per_group``: the replicas
inside one group run one after another in the same environment, so they share
those tasks over time rather than splitting them.  Dividing again would leave
``(n_replica_per_group - 1) / n_replica_per_group`` of the machine idle.

Every input is floored at 1 before use.  The resource widgets are unbounded
integer fields, so a mistyped node count or an allocation smaller than the
replica count could otherwise produce ``NPROC_REP = 0``, which CP2K rejects.

``NebResourceHelper`` is the opt-in panel that puts the layout into words for
the submission form.  It lives here rather than in the notebook so that it can
be tested; nothing in a notebook cell can be.
"""

import math

import ipywidgets as ipw


def at_least(value, minimum=1):
    """Coerce ``value`` to an ``int`` no smaller than ``minimum``.

    Anything that is not convertible to an ``int`` (``None``, an empty widget
    value) falls back to ``minimum``.
    """
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return minimum


def resource_layout(
    n_replica,
    n_replica_per_group,
    nodes,
    tasks_per_node,
    threads_per_task,
):
    """Return the replica/group/task layout for a NEB run.

    All arguments are clamped to >= 1, so the returned ``nproc_replica`` is
    always a valid ``NPROC_REP`` value.
    """
    n_replica = at_least(n_replica)
    n_replica_per_group = at_least(n_replica_per_group)
    nodes = at_least(nodes)
    tasks_per_node = at_least(tasks_per_node)
    threads_per_task = at_least(threads_per_task)

    n_groups = math.ceil(n_replica / n_replica_per_group)
    total_tasks = nodes * tasks_per_node

    return {
        "n_replica": n_replica,
        "n_replica_per_group": n_replica_per_group,
        "n_groups": n_groups,
        "nodes": nodes,
        "tasks_per_node": tasks_per_node,
        "threads_per_task": threads_per_task,
        # Kept as a float: a non-integer value means the groups do not divide
        # the allocation evenly, which is worth showing to the user.
        "tasks_per_group": total_tasks / n_groups,
        "nproc_replica": at_least(total_tasks // n_groups),
    }


class NebResourceHelper(ipw.VBox):
    """Opt-in panel explaining how an allocation splits into replica groups.

    ``layout_provider`` is called on every refresh and must return a
    ``resource_layout`` result.  It is a callable rather than the resource
    widgets themselves because the caller owns those widgets and the clamping
    applied to them; this panel only reads what it is handed.

    Hook ``update`` up as a traitlets observer on anything that feeds the
    layout.  The checkbox is observed here, so callers do not repeat it.
    """

    def __init__(self, layout_provider, **kwargs):
        self._layout_provider = layout_provider
        self.show = ipw.Checkbox(
            value=False,
            description="Show NEB resource helper",
            indent=False,
            style={"description_width": "initial"},
        )
        self.text = ipw.HTML()
        self.show.observe(self.update, "value")
        super().__init__([self.show, self.text], **kwargs)
        self.update()

    def update(self, _=None):
        """Re-render the panel, or clear it while the box is unchecked."""
        self.text.value = self.describe_layout() if self.show.value else ""

    def describe_layout(self):
        """Render the current layout as one HTML sentence.

        Warns when the MPI tasks do not divide evenly between the groups,
        which is the case that leaves part of the allocation idle.  Note the
        test is on the tasks, not on the nodes: half a node per group wastes
        nothing as long as the tasks come out whole, and that is a layout the
        resources widget produces deliberately.

        ``threads_per_task`` is left out on purpose - it is clamped alongside
        the rest but plays no part in the grouping.
        """
        layout = self._layout_provider()
        tasks_per_group = layout["tasks_per_group"]
        text = (
            f"{layout['n_replica']} replicas at {layout['n_replica_per_group']} "
            f"per group run as <b>{layout['n_groups']} replica group(s)</b>. "
            f"{layout['nodes']} nodes x {layout['tasks_per_node']} tasks per "
            f"node are shared as {layout['nodes'] / layout['n_groups']:g} "
            f"node(s), i.e. {tasks_per_group:g} MPI tasks, per group, so "
            f"<b>NPROC_REP = {layout['nproc_replica']}</b>. "
            "The threads per task do not enter this split."
        )
        if tasks_per_group != int(tasks_per_group):
            text += (
                " <b>The tasks do not divide evenly between the groups</b>, so "
                f"{layout['nproc_replica']} tasks per group leaves part of the "
                "allocation idle. Adjust the total number of nodes or the "
                "number of replicas per group."
            )
        return f"<p>{text}</p>"
