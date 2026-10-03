from ProcedureMem.analyze_p2_offload_oracle import analyze, exact_mcnemar_p_value


def result(task_id, index, reward):
    return {
        "task_id": task_id,
        "task_index": index,
        "task_type": "pick_and_place_simple-Apple-None-Desk-1",
        "query": "put an apple on the desk",
        "reward": reward,
        "steps": 1,
    }


def indexed(values):
    return {
        f"task-{index}": result(f"task-{index}", index, reward)
        for index, reward in enumerate(values)
    }


def test_analyze_decomposes_offload_membership_flips():
    # task 0: offload -> Edge success
    # task 1: offload -> neither success
    # task 2: Edge success -> offload
    # task 3: neither success -> offload
    # task 4: offload persists
    summary, rows = analyze(
        edge_no_memory=indexed([0, 0, 1, 0, 0]),
        cloud_no_memory=indexed([1, 1, 1, 0, 1]),
        edge_memory=indexed([1, 0, 0, 0, 0]),
        cloud_memory=indexed([1, 0, 1, 1, 1]),
    )

    comparison = summary["offload_set_comparison"]
    assert comparison["no_memory_count"] == 3
    assert comparison["memory_count"] == 3
    assert comparison["overlap_count"] == 1
    assert comparison["removed_count"] == 2
    assert comparison["added_count"] == 2
    assert comparison["jaccard_similarity"] == 0.2

    transitions = summary["offload_membership_transition_counts"]
    assert transitions["offload_to_edge_success"] == 1
    assert transitions["offload_to_neither_success"] == 1
    assert transitions["edge_success_to_offload"] == 1
    assert transitions["neither_success_to_offload"] == 1
    assert transitions["offload_value_persisted"] == 1
    assert [row["task_index"] for row in rows] == list(range(5))


def test_exact_mcnemar_p_value_handles_balanced_and_one_sided_changes():
    assert exact_mcnemar_p_value(0, 0) == 1.0
    assert exact_mcnemar_p_value(2, 2) == 1.0
    assert exact_mcnemar_p_value(5, 0) == 0.0625
