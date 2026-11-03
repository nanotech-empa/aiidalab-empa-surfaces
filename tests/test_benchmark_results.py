"""Check benchmark ranges against configurations, failures and tied timings."""

import pytest

from surfaces_tools.utils.benchmark_results import summarize_timings


def test_groups_nodes_numerically_and_keeps_extreme_configurations():
    rows = summarize_timings(
        {
            "10_16_2": [1.5, "job-10"],
            "2_8_4": [4.0, "job-best"],
            "2_4_8": [9.0, "job-worst"],
            "2_12_2": [6.0, "job-middle"],
            "1_4_2": [12.0, "job-1"],
        }
    )
    assert [row["nodes"] for row in rows] == [1, 2, 10]
    row = rows[1]
    assert (row["min_time"], row["max_time"]) == (4.0, 9.0)
    assert (row["successful"], row["tested"]) == (3, 3)
    assert row["best"] == [
        {"tasks_per_node": 8, "threads_per_task": 4, "time": 4.0, "job_id": "job-best"}
    ]
    assert row["worst"] == [
        {"tasks_per_node": 4, "threads_per_task": 8, "time": 9.0, "job_id": "job-worst"}
    ]


def test_failed_cases_do_not_define_extrema_and_failed_nodes_remain_visible():
    rows = summarize_timings(
        {
            "1_4_2": [7.0, "ok"],
            "1_8_2": ["FAILED", "failed"],
            "2_4_2": ["FAILED", "failed-2"],
        }
    )
    assert (rows[0]["min_time"], rows[0]["max_time"]) == (7.0, 7.0)
    assert (rows[0]["successful"], rows[0]["tested"]) == (1, 2)
    assert rows[1] == {
        "nodes": 2,
        "tested": 1,
        "successful": 0,
        "min_time": None,
        "max_time": None,
        "best": [],
        "worst": [],
    }


def test_lists_all_ties_in_numeric_configuration_order():
    row = summarize_timings(
        {
            "1_16_2": [2.0, "fast-16"],
            "1_4_8": [2.0, "fast-4"],
            "1_8_4": [5.0, "slow-8"],
            "1_12_2": [5.0, "slow-12"],
        }
    )[0]
    assert [item["job_id"] for item in row["best"]] == ["fast-4", "fast-16"]
    assert [item["job_id"] for item in row["worst"]] == ["slow-8", "slow-12"]


def test_equal_timings_are_both_best_and_worst():
    row = summarize_timings({"1_4_2": [3.0, "a"], "1_8_2": [3.0, "b"]})[0]
    assert row["best"] == row["worst"]
    assert len(row["best"]) == 2


@pytest.mark.parametrize("timing", [None, float("nan"), float("inf"), 0, -1])
def test_unusable_timings_cannot_be_reported_as_best(timing):
    row = summarize_timings({"1_4_2": [timing, "bad"], "1_8_2": [3.0, "ok"]})[0]
    assert (row["min_time"], row["max_time"]) == (3.0, 3.0)
    assert row["successful"] == 1


def test_empty_results():
    assert summarize_timings({}) == []
