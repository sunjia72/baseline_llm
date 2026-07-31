from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

import run_baseline_llm as baseline_runner
from constraint_baseline import (
    COMPILED_CONSTRAINT_KIND,
    COMPILED_CONSTRAINT_SCHEMA_VERSION,
    COMPILED_WORKLOAD_KIND,
    COMPILED_WORKLOAD_SCHEMA_VERSION,
    compose_user_content,
    evaluate_generated_ids,
    family_order,
    load_workload,
    nfa_accepts_content,
    overlapping_positions,
    render_constraint_instruction,
    require_tokenizer_compatibility,
    sha256_value,
    terminal_eos_length_contract,
    validate_compiled_constraint_artifact,
)
from run_baseline_llm import (
    BASELINE_METHOD,
    MANIFEST_KIND,
    RESULT_SCHEMA_VERSION,
    _balanced_assignments,
    _apply_terminal_boundary_probe,
    _collect_samples,
    _configured_terminal_token_ids,
    _finalize_results,
    _model_slug,
    _parse_gpu_ids,
    _planned_tasks,
    _prepare_scoring_rows,
    _sample_seed,
    _slurm_command,
    _worker_slots,
    select_jobs,
)


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 98
    all_special_ids = [98, 99, 100]
    chat_template = "fake"

    def decode(
        self,
        token_ids: List[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        if token_ids == [9]:
            return " and"
        return "".join(f"<{token_id}>" for token_id in token_ids)


def tokenizer_fingerprint() -> Dict[str, Any]:
    files = {"tokenizer.json": "a" * 64}
    return {
        "path": "/models/test",
        "vocab_size": 128,
        "eos_token_id": 99,
        "pad_token_id": 98,
        "all_special_ids": [98, 99, 100],
        "files": files,
        "combined_sha256": sha256_value(files),
    }


def compiled_artifact(
    family: str,
    patterns: List[List[int]],
    *,
    separator: int = 9,
    rule: str | None = None,
    n_low: int = 2,
    n: int = 65,
) -> Dict[str, Any]:
    primitive_ids = list(
        dict.fromkeys(token for pattern in patterns for token in pattern)
    )
    token_classes = [
        {
            "name": f"piece_{index:03d}",
            "symbol_id": index,
            "token_ids": [token_id],
        }
        for index, token_id in enumerate(primitive_ids)
    ]
    token_classes.append(
        {
            "name": "separator",
            "symbol_id": len(token_classes),
            "token_ids": [separator],
        }
    )
    stop_symbol = len(token_classes) + 1
    body = {
        "schema_version": COMPILED_CONSTRAINT_SCHEMA_VERSION,
        "kind": COMPILED_CONSTRAINT_KIND,
        # This generic test NFA accepts iff the first tracked token occurs.
        "nfa": {
            "num_states": 3,
            "alphabet_size": len(token_classes) + 2,
            "initials": [0],
            "finals": [2],
            "transitions": [[0, 0, 1], [1, stop_symbol, 2]],
            "wildcard_transitions": [[0, 0], [1, 1]],
        },
        "token_classes": token_classes,
        "other_symbol_id": len(token_classes),
        "stop_symbol_id": stop_symbol,
        "n_low": n_low,
        "n": n,
        "length_contract": terminal_eos_length_contract(n_low, n),
        "prompt_token_ids": [10, 20],
        "tokenizer_fingerprint": tokenizer_fingerprint(),
        "instruction": {
            "family": family,
            "label": f"{family} label",
            "rule": rule or f"Satisfy the authenticated {family} test rule.",
        },
        "provenance": {"fixture": True},
    }
    return {**body, "sha256": sha256_value(body)}


def row(
    family: str = "family_alpha",
    *,
    patterns: List[List[int]] | None = None,
    separator: int = 9,
    rule: str | None = None,
) -> Dict[str, Any]:
    patterns = patterns or [[1], [2]]
    artifact = compiled_artifact(
        family,
        patterns,
        separator=separator,
        rule=rule,
    )
    return {
        "job_id": f"000_test_{family}",
        "design_index": 0,
        "family_index": 0,
        "base_instance_id": "test:0",
        "dataset": "test",
        "constraint": family,
        "k": None,
        "t": len(patterns),
        "n_low": artifact["n_low"],
        "n": artifact["n"],
        "seed": 1,
        "prompt_text": "Write exactly one short sentence.",
        "task_prompt_text": "Write exactly one short sentence.",
        "prompt_token_ids": list(artifact["prompt_token_ids"]),
        "separator_text": " and",
        "separator_token_id": separator,
        "tracked_keyword_token_ids": patterns,
        "tracked_token_texts": [" alpha", " beta"][: len(patterns)],
        "selected_keywords": [
            {"surface": name} for name in ("alpha", "beta")[: len(patterns)]
        ],
        "nfa_states": artifact["nfa"]["num_states"],
        "compiled_constraint": artifact,
    }


def workload(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_low = rows[0]["n_low"]
    n = rows[0]["n"]
    return {
        "schema_version": COMPILED_WORKLOAD_SCHEMA_VERSION,
        "kind": COMPILED_WORKLOAD_KIND,
        "dataset": "test",
        "length_interval_including_eos": [n_low, n],
        "length_contract": terminal_eos_length_contract(n_low, n),
        "family_counts": {
            family: sum(item["constraint"] == family for item in rows)
            for family in family_order(rows)
        },
        "total_instances": len(rows),
        "jobs": rows,
        "jobs_sha256": sha256_value(rows),
    }


def write_workload(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_generic_compiled_nfa_accepts_without_family_code() -> None:
    example = row("a_future_constraint")
    assert nfa_accepts_content(example, [7, 1, 8])
    assert not nfa_accepts_content(example, [7, 2, 8])
    result = evaluate_generated_ids(
        example,
        [7, 1, 8, 99],
        FakeTokenizer(),
    )
    assert result["strict_accepts"]


def test_overlap_and_multitoken_diagnostics() -> None:
    example = row(patterns=[[1, 1]])
    assert overlapping_positions([1, 1, 1], [1, 1]) == [0, 1]
    result = evaluate_generated_ids(
        example,
        [1, 1, 1, 99],
        FakeTokenizer(),
    )
    assert result["strict_accepts"]
    assert result["occurrence_counts"] == [2]


def test_multiple_model_native_terminal_ids() -> None:
    example = row(patterns=[[1]])
    result = evaluate_generated_ids(
        example,
        [1, 100],
        FakeTokenizer(),
        terminal_token_ids=[99, 100],
    )
    assert result["strict_accepts"]
    assert result["actual_terminal_token_id"] == 100
    assert result["content_token_ids"] == [1]

    midstream = evaluate_generated_ids(
        example,
        [1, 100, 5],
        FakeTokenizer(),
        terminal_token_ids=[99, 100],
    )
    assert not midstream["terminal_eos_ok"]
    assert not midstream["strict_accepts"]


def test_configured_terminal_ids_include_all_model_native_aliases() -> None:
    config = SimpleNamespace(
        eos_token_id=[99, 101],
        text_config=SimpleNamespace(eos_token_id=102),
    )
    generation_config = SimpleNamespace(eos_token_id=[99, 103])

    class SizedTokenizer(FakeTokenizer):
        eos_token_ids = [99, 100]

        def __len__(self) -> int:
            return 128

    assert _configured_terminal_token_ids(
        SizedTokenizer(),
        config,
        generation_config,
    ) == [99, 100, 101, 102, 103]


def test_first_sample_uses_authenticated_job_seed() -> None:
    example = row()
    example["seed"] = 123456

    assert _sample_seed(999, example, 0) == 123456
    assert _sample_seed(999, example, 1) != 123456


def test_terminal_boundary_probe_preserves_eos_and_discards_nonterminal() -> None:
    terminal, discarded = _apply_terminal_boundary_probe(
        [1, 2, 99],
        [99, 100],
        3,
    )
    assert terminal == [1, 2, 99]
    assert discarded is None

    capped, discarded = _apply_terminal_boundary_probe(
        [1, 2, 3],
        [99, 100],
        3,
    )
    assert capped == [1, 2]
    assert discarded == 3


@pytest.mark.parametrize(
    ("generated", "terminal", "length", "budget", "valid"),
    (
        ([1], False, False, False, True),
        ([1, 98, 99], True, True, True, False),
        ([99], True, False, False, True),
        ([1, 99, 5], False, False, True, True),
    ),
)
def test_strict_terminal_length_and_special_token_checks(
    generated: List[int],
    terminal: bool,
    length: bool,
    budget: bool,
    valid: bool,
) -> None:
    result = evaluate_generated_ids(row(patterns=[[1]]), generated, FakeTokenizer())
    assert result["terminal_eos_ok"] is terminal
    assert result["length_ok"] is length
    assert result["raw_token_budget_ok"] is budget
    assert result["valid_content_tokens"] is valid
    assert not result["strict_accepts"]


def test_instruction_rule_is_consumed_from_artifact() -> None:
    rule = "Use the exact rule supplied by the workload, including Ω."
    example = row("unrecognized_family_name", rule=rule)
    example["prompt_text"] = "Continue this exact source once."
    example["task_prompt_text"] = example["prompt_text"]
    instruction = render_constraint_instruction(example)
    content = compose_user_content(example)
    assert rule in instruction
    assert "1 and 64 content model tokens" in instruction
    assert content.count(example["prompt_text"]) == 1
    assert content.count("Output constraint:") == 1
    assert "Return only the requested generated text" not in content


def test_prompt_composition_requires_authenticated_task_prompt() -> None:
    example = row()
    example.pop("task_prompt_text")
    with pytest.raises(ValueError, match="task_prompt_text"):
        compose_user_content(example)

    example = row()
    example["prompt_text"] = "A different prompt."
    with pytest.raises(ValueError, match="must equal"):
        compose_user_content(example)


def test_separator_rule_is_clarified_from_verified_runtime_tokenizer() -> None:
    example = row(
        "future_side_constraint",
        rule="Using the first separator as the split, put alpha on one side.",
    )
    identity_before = sha256_value(example)

    with pytest.raises(ValueError, match="runtime tokenizer"):
        render_constraint_instruction(example)

    instruction = render_constraint_instruction(example, FakeTokenizer())
    content = compose_user_content(example, FakeTokenizer())
    assert (
        '"separator" means the exact model-token phrase \'and\'.' in instruction
    )
    assert instruction in content
    assert sha256_value(example) == identity_before

    incompatible = copy.deepcopy(example)
    incompatible["separator_text"] = " or"
    with pytest.raises(ValueError, match="separator text differs"):
        render_constraint_instruction(incompatible, FakeTokenizer())


def test_successful_record_has_normalized_method_and_structure_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = row()
    example["unrolled_states"] = 250
    task = {
        "row": example,
        "sample_index": 0,
        "sample_seed": 17,
    }
    generated = {
        "prompt_token_count": 2,
        "prompt_token_ids_sha256": "a" * 64,
        "new_token_ids": [1, 99],
        "generated_text": "alpha",
        "generation_runtime_s": 0.1,
        "cuda_peak_allocated_gib": 1.0,
        "evaluation": {
            "strict_accepts": False,
            "semantic_constraint_accepts": False,
            "terminal_eos_ok": True,
            "length_ok": True,
            "raw_token_budget_ok": True,
        },
    }
    monkeypatch.setattr(
        baseline_runner,
        "_generate",
        lambda *args, **kwargs: generated,
    )

    record = baseline_runner._build_record(
        object(),
        FakeTokenizer(),
        task,
        SimpleNamespace(),
    )
    assert record["schema_version"] == RESULT_SCHEMA_VERSION
    assert record["method"] == BASELINE_METHOD
    assert record["status"] == "success"
    assert record["evaluation"]["strict_accepts"] is False
    assert (
        record["compiled_constraint_sha256"]
        == example["compiled_constraint"]["sha256"]
    )
    assert record["design_index"] == example["design_index"]
    assert record["nfa_states"] == example["nfa_states"]
    assert record["unrolled_states"] == 250
    assert record["precompute_runtime_s"] is None


def test_compiled_artifact_digest_and_structure_are_checked() -> None:
    artifact = row()["compiled_constraint"]
    validate_compiled_constraint_artifact(artifact)

    damaged = copy.deepcopy(artifact)
    damaged["nfa"]["finals"] = [0]
    with pytest.raises(ValueError, match="digest"):
        validate_compiled_constraint_artifact(damaged)

    damaged = copy.deepcopy(artifact)
    damaged["token_classes"][1]["symbol_id"] = 0
    body = {key: value for key, value in damaged.items() if key != "sha256"}
    damaged["sha256"] = sha256_value(body)
    with pytest.raises(ValueError, match="token class"):
        validate_compiled_constraint_artifact(damaged)


def test_tokenizer_fingerprint_compatibility_and_no_retokenization() -> None:
    source = row()
    original = copy.deepcopy(source)
    actual = tokenizer_fingerprint()
    actual["path"] = "/another/mount/of/the/same/model"
    require_tokenizer_compatibility(
        source["compiled_constraint"]["tokenizer_fingerprint"],
        actual,
    )
    prepared, metadata = _prepare_scoring_rows([source], actual)
    assert prepared == [source]
    assert source == original
    assert metadata["policy"].endswith("without retokenization")

    incompatible = copy.deepcopy(actual)
    incompatible["vocab_size"] += 1
    with pytest.raises(ValueError, match="vocab_size"):
        require_tokenizer_compatibility(
            source["compiled_constraint"]["tokenizer_fingerprint"],
            incompatible,
        )


def test_compiled_artifact_rejects_empty_tokenizer_file_fingerprint() -> None:
    artifact = copy.deepcopy(row()["compiled_constraint"])
    artifact["tokenizer_fingerprint"]["files"] = {}
    artifact["tokenizer_fingerprint"]["combined_sha256"] = sha256_value({})
    body = {key: value for key, value in artifact.items() if key != "sha256"}
    artifact["sha256"] = sha256_value(body)
    with pytest.raises(ValueError, match="files"):
        validate_compiled_constraint_artifact(artifact)


def test_schema_v4_workload_hash_and_dynamic_family_order(
    tmp_path: Path,
) -> None:
    rows = []
    for family in ("family_beta", "family_alpha"):
        for family_index in range(3):
            item = row(family)
            item["job_id"] = f"{len(rows):03d}_{family}_{family_index}"
            item["design_index"] = len(rows)
            item["family_index"] = family_index
            rows.append(item)
    payload = workload(rows)
    path = tmp_path / "workload.json"
    write_workload(path, payload)

    loaded = load_workload(path, expected_dataset="test")
    assert loaded["_family_order"] == ["family_beta", "family_alpha"]
    assert loaded["_execution_jobs"] == loaded["jobs"]
    selected = select_jobs(loaded["_execution_jobs"], "midpoint_per_family", [])
    assert [item["constraint"] for item in selected] == [
        "family_beta",
        "family_alpha",
    ]
    assert all(item["family_index"] == 1 for item in selected)

    payload["jobs_sha256"] = "0" * 64
    write_workload(path, payload)
    with pytest.raises(ValueError, match="jobs_sha256"):
        load_workload(path, expected_dataset="test")


def test_authenticated_execution_selection_limits_baseline_jobs(
    tmp_path: Path,
) -> None:
    rows = []
    for index in range(4):
        item = row("family_alpha")
        item["job_id"] = f"{index:03d}_family_alpha"
        item["design_index"] = index
        item["replicate_index"] = index % 2
        rows.append(item)
    payload = workload(rows)
    selected = [rows[0], rows[2]]
    selected_ids = [item["job_id"] for item in selected]
    selection = {
        "schema_version": 1,
        "selection_order": "canonical_workload_order",
        "operation_order": ["replicate_filter", "max_jobs_prefix"],
        "replicate_indices": [0],
        "max_jobs": None,
        "source_instances": 4,
        "instances_after_replicate_filter": 2,
        "selected_instances": 2,
        "selected_family_counts": {"family_alpha": 2},
        "selected_job_ids": selected_ids,
        "selected_job_ids_sha256": sha256_value(selected_ids),
    }
    payload["execution_selection"] = {
        **selection,
        "sha256": sha256_value(selection),
    }
    path = tmp_path / "selected.json"
    write_workload(path, payload)

    loaded = load_workload(path)
    assert [item["job_id"] for item in loaded["_execution_jobs"]] == selected_ids

    payload["execution_selection"]["selected_instances"] = 4
    selection_body = dict(payload["execution_selection"])
    selection_body.pop("sha256")
    payload["execution_selection"]["sha256"] = sha256_value(selection_body)
    write_workload(path, payload)
    with pytest.raises(ValueError, match="differs from its jobs"):
        load_workload(path)


def test_legacy_or_missing_compiled_workload_fails_clearly(
    tmp_path: Path,
) -> None:
    payload = workload([row()])
    payload["schema_version"] = 3
    path = tmp_path / "legacy.json"
    write_workload(path, payload)
    with pytest.raises(ValueError, match="schema_version 4"):
        load_workload(path)

    payload["schema_version"] = 4
    payload["jobs"][0].pop("compiled_constraint")
    payload["jobs_sha256"] = sha256_value(payload["jobs"])
    write_workload(path, payload)
    with pytest.raises(ValueError, match="compiled_constraint"):
        load_workload(path)


def test_workload_dataset_identity_is_checked(tmp_path: Path) -> None:
    payload = workload([row()])
    payload["dataset"] = "common_gen"
    payload["jobs"][0]["dataset"] = "common_gen"
    payload["jobs_sha256"] = sha256_value(payload["jobs"])
    path = tmp_path / "workload.json"
    write_workload(path, payload)
    with pytest.raises(ValueError, match="does not match"):
        load_workload(path, expected_dataset="coauthor")


def test_executable_baseline_has_no_compiler_or_family_dispatch() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            Path(__file__).with_name("constraint_baseline.py"),
            Path(__file__).with_name("run_baseline_llm.py"),
        )
    )
    for forbidden in (
        "if family ==",
        "FAMILIES =",
        "src/",
    ):
        assert forbidden not in source


def test_distributed_partition_is_balanced_and_complete() -> None:
    tasks = [{"ordinal": ordinal} for ordinal in range(300)]
    assignments = _balanced_assignments(tasks, 16)
    assert [len(shard) for shard in assignments] == [19] * 12 + [18] * 4
    flattened = [int(task["ordinal"]) for shard in assignments for task in shard]
    assert sorted(flattened) == list(range(300))
    assert len(flattened) == len(set(flattened))


def test_node_major_worker_slots_and_gpu_validation() -> None:
    gpu_ids = _parse_gpu_ids("0,1,2,3")
    slots = _worker_slots(["node-a", "node-b", "node-c", "node-d"], gpu_ids)
    assert len(slots) == 16
    assert slots[0] == {
        "slot_index": 0,
        "node": "node-a",
        "physical_gpu_id": 0,
    }
    assert slots[-1] == {
        "slot_index": 15,
        "node": "node-d",
        "physical_gpu_id": 3,
    }
    with pytest.raises(ValueError, match="duplicate"):
        _parse_gpu_ids("0,1,1")


def test_model_slug_is_path_derived() -> None:
    assert _model_slug(Path("/models/gemma-4-E2B-it")) == "gemma_4_e2b_it"


def test_slurm_worker_masks_physical_gpu_but_uses_logical_cuda_zero(
    tmp_path: Path,
) -> None:
    worker_dir = tmp_path / "slot"
    command = _slurm_command(
        {"slot_index": 6, "node": "node-b", "physical_gpu_id": 2},
        worker_dir,
        worker_dir / "plan.json",
        8,
    )
    assert "--overlap" in command
    assert "--exact" in command
    assert "--exclusive" not in command
    assert "--nodelist=node-b" in command
    assert "/usr/bin/env" in command
    assert "CUDA_VISIBLE_DEVICES=2" in command
    assert "OMP_NUM_THREADS=8" in command


def _distributed_record(task: Dict[str, Any], run_signature: str) -> Dict[str, Any]:
    example = task["row"]
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "method": BASELINE_METHOD,
        "status": "success",
        "dataset": example["dataset"],
        "job_id": example["job_id"],
        "design_index": example["design_index"],
        "base_instance_id": example["base_instance_id"],
        "constraint": example["constraint"],
        "k": example.get("k"),
        "t": example["t"],
        "n_low": example["n_low"],
        "n": example["n"],
        "compiled_constraint_sha256": example["compiled_constraint"]["sha256"],
        "nfa_states": example["nfa_states"],
        "unrolled_states": example.get("unrolled_states"),
        "precompute_runtime_s": None,
        "sample_index": task["sample_index"],
        "sample_seed": task["sample_seed"],
        "new_token_ids": [1, 99],
        "generated_token_ids": [1, 99],
        "generated_content_token_ids": [1],
        "generated_total_len": 2,
        "generated_content_len": 1,
        "terminal_eos_token_id": 99,
        "terminal_eos_count": 1,
        "terminated_with_eos": True,
        "hit_content_token_cap": False,
        "valid_generation": True,
        "keyword_occurrence_counts": [1],
        "raw_generated_token_count": 2,
        "discarded_nonterminal_probe": False,
        "discarded_probe_token_id": None,
        "evaluation": {
            "strict_accepts": True,
            "semantic_constraint_accepts": True,
            "terminal_eos_ok": True,
            "length_ok": True,
            "raw_token_budget_ok": True,
            "actual_terminal_token_id": 99,
            "generated_total_token_count": 2,
            "generated_content_token_count": 1,
            "content_token_ids": [1],
            "occurrence_counts": [1],
        },
        "generation_runtime_s": 0.1,
        "execution": {
            "run_signature_sha256": run_signature,
            "task_ordinal": task["ordinal"],
        },
    }


def test_distributed_collection_is_canonical_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    rows = [row("family_alpha"), row("family_beta")]
    rows[0]["dataset"] = "common_gen"
    rows[1]["dataset"] = "coauthor"
    rows[0]["job_id"] = "job-z"
    rows[1]["job_id"] = "job-a"
    tasks = _planned_tasks(rows, 1, 7)
    signature = "abc123"
    first = tmp_path / "dispatches/dispatch_000/slot_001/records/000001.json"
    second = tmp_path / "dispatches/dispatch_000/slot_000/records/000000.json"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text(json.dumps(_distributed_record(tasks[1], signature)))
    second.write_text(json.dumps(_distributed_record(tasks[0], signature)))
    ordered, by_key = _collect_samples(tmp_path, tasks, signature)
    assert len(by_key) == 2
    assert [sample["job_id"] for sample in ordered] == ["job-z", "job-a"]

    duplicate = tmp_path / "dispatches/dispatch_001/slot_000/records/000000.json"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_text(json.dumps(_distributed_record(tasks[0], signature)))
    with pytest.raises(ValueError, match="duplicate sample"):
        _collect_samples(tmp_path, tasks, signature)


def test_finalization_emits_canonical_results_manifest_and_legacy_aliases(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "result"
    output_dir.mkdir()
    config_path = output_dir / "config.json"
    signature = "abc123"
    config_path.write_text(
        json.dumps(
            {
                "run_signature_sha256": signature,
                "datasets": ["test"],
                "workloads": {"test": {"jobs_sha256": "b" * 64}},
            }
        ),
        encoding="utf-8",
    )
    task = _planned_tasks([row()], 1, 7)[0]
    record = _distributed_record(task, signature)

    summary = _finalize_results(
        output_dir,
        Path("/models/test"),
        config_path,
        [record],
    )

    assert (output_dir / "results.jsonl").read_bytes() == (
        output_dir / "samples.jsonl"
    ).read_bytes()
    assert (output_dir / "results.csv").read_bytes() == (
        output_dir / "samples.csv"
    ).read_bytes()
    result = json.loads(
        (output_dir / "results.jsonl").read_text(encoding="utf-8")
    )
    assert result["status"] == "success"
    assert result["evaluation"]["strict_accepts"] is True
    assert result["precompute_runtime_s"] is None

    manifest = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["kind"] == MANIFEST_KIND
    assert manifest["method"] == BASELINE_METHOD
    assert manifest["status"] == "complete"
    assert manifest["records"] == 1
    assert manifest["precompute_runtime_s"] is None
    assert set(manifest["artifacts"]) == {
        "config",
        "results_jsonl",
        "results_csv",
        "samples_jsonl",
        "samples_csv",
        "summary",
    }
    assert summary["results_jsonl"].endswith("results.jsonl")
    assert summary["precompute_runtime_s"] is None


def test_missing_workload_message_names_current_schema() -> None:
    source = Path(baseline_runner.__file__).read_text(encoding="utf-8")
    assert "current schema-v4 compiled workloads are required" in source
    assert "current schema-v3 compiled workloads are required" not in source
