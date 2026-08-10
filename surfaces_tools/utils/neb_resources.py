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
"""

import math


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
