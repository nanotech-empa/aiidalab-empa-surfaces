"""Summarize the saved timings of a CP2K benchmark without loading ORM nodes."""

import math


def summarize_timings(raw_timings):
    """Return timing extrema and all matching configurations for each node count.

    Keys encode nodes_tasks-per-node_threads-per-task; values contain the
    timing and scheduler job ID. Failed or unusable timings do not define an
    extremum, but remain in the tested count, including entirely failed groups.
    """
    groups = {}
    for key, (timing, job_id) in raw_timings.items():
        nodes, tasks, threads = map(int, key.split("_"))
        group = groups.setdefault(nodes, {"tested": 0, "configurations": []})
        group["tested"] += 1
        try:
            seconds = float(timing)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(seconds) or seconds <= 0:
            continue
        group["configurations"].append(
            {
                "tasks_per_node": tasks,
                "threads_per_task": threads,
                "time": seconds,
                "job_id": job_id,
            }
        )

    results = []
    for nodes, group in sorted(groups.items()):
        configurations = sorted(
            group["configurations"],
            key=lambda item: (item["tasks_per_node"], item["threads_per_task"]),
        )
        minimum = min((item["time"] for item in configurations), default=None)
        maximum = max((item["time"] for item in configurations), default=None)
        results.append(
            {
                "nodes": nodes,
                "tested": group["tested"],
                "successful": len(configurations),
                "min_time": minimum,
                "max_time": maximum,
                "best": [item for item in configurations if item["time"] == minimum],
                "worst": [item for item in configurations if item["time"] == maximum],
            }
        )
    return results
