"""Official metric parity, real domain ports, provenance and hard-val guards."""

import ast
import contextlib
import importlib
import io
import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.config import load_config
from src.domains.base import EvaluationResult, SamplingRequest
from src.domains.homebench.data import load
from src.domains.homebench.evaluator import aggregate, evaluate, judge
from src.domains.homebench.icl import OFFICIAL_FOUR_SHOT
from src.domains.homebench.plugin import PLUGIN
from src.domains.homebench.rejection_sampling import collect as sample_training
from src.domains.homebench.runtime.execution import observation, run_sample
from src.domains.homebench.runtime.infer import collect
from src.domains.homebench.selection import select
from src.training.ms_swift import prepare_ms_swift_dataset
from src.utils.persistence import atomic_jsonl

ROOT = Path(__file__).resolve().parents[1]


def write(path, rows):
    atomic_jsonl(Path(path), rows)
SOURCE = ROOT / "data/HomeBench"


def test_official_four_shot_prompt_parity():
    def normalized(text):
        return "\n".join(line.rstrip() for line in text.strip().splitlines())

    released = (SOURCE / "code/four_shot.txt").read_text()
    assert normalized(OFFICIAL_FOUR_SHOT) == normalized(released)


def call(name, args, index=1):
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": "Use only supported methods.",
        "tool_calls": [
            {
                "id": f"c{index}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


class Model:
    def __init__(self, generated="{room.light.turn_on()}", inspect=False, error=False):
        self.generated, self.inspect, self.error, self.count = generated, inspect, error, 0

    def preflight(self):
        return 200

    def complete(self, messages, tools):
        self.count += 1
        if self.error:
            raise ConnectionError("endpoint unavailable")
        if self.inspect and self.count == 1:
            return call("inspect_home", {"room": "room"}, self.count)
        return call("final_answer", {"commands": self.generated}, self.count)


@pytest.fixture
def dataset(tmp_path):
    home = {
        "home_id": 1,
        "home_status": {"room": {"light": {"state": "off", "attributes": {}}}},
        "method": [
            {"room_name": "room", "device_name": "light", "operation": "turn_on", "parameters": []}
        ],
    }
    rows = [
        {
            "sample_id": f"task{i}",
            "domain": "homebench",
            "instruction": f"Turn on the light {i}",
            "task_type": "normal",
            "environment": {"protocol": "homebench_counter_v1", "home_id": 1},
            "target": {"commands": "'''room.light.turn_on()'''"},
            "metadata": {"source_split": "train", "label_conflict": False},
        }
        for i in range(3)
    ]
    write(tmp_path / "homes.jsonl", [home])
    write(tmp_path / "train.jsonl", rows)
    test = deepcopy(rows)
    for r in test:
        r["metadata"]["source_split"] = "test"
    write(tmp_path / "test.jsonl", test)
    config = load_config(ROOT / "configs/homebench.yaml")
    config["harness"]["root"] = str(ROOT / "harnesses/homebench")
    return tmp_path, config


def test_loader_accepts_portable_embedded_export(tmp_path):
    row = {
        "id": "home1_multi_1",
        "input": "Turn on the light.",
        "output": "'''room.light.turn_on()'''",
        "home_id": 1,
        "type": "multi2_normal",
        "category": "VM",
        "system_prompt": "upstream prompt is provenance, not harness input",
        "home_status_json": json.dumps(
            {"room": {"light": {"state": "off", "attributes": {}}}}
        ),
        "methods_json": json.dumps(
            [
                {
                    "room_name": "room",
                    "device_name": "light",
                    "operation": "turn_on",
                    "parameters": [],
                }
            ]
        ),
        "source_split": "train",
    }
    write(tmp_path / "train.jsonl", [row])

    sample = next(load(tmp_path / "train.jsonl"))

    assert sample.sample_id == "home1_multi_1"
    assert sample.instruction == "Turn on the light."
    assert sample.record["target"]["commands"] == row["output"]
    assert sample.record["metadata"] == {
        "source_split": "train",
        "category": "VM",
        "label_conflict": False,
    }
    assert sample.home["home_status"]["room"]["light"]["state"] == "off"
    assert sample.home["method"][0]["operation"] == "turn_on"
    assert "system_prompt" not in sample.record


def test_official_counter_parity():
    tree = ast.parse((SOURCE / "code/eval.py").read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_accuracy"
    )
    module = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name="re")]),
            ast.ImportFrom(module="collections", names=[ast.alias(name="Counter")], level=0),
            function,
        ],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), "official_eval", "exec"), namespace)  # noqa: S102 -- trusted upstream function only
    samples = list(load(SOURCE / "multi/test_debug.jsonl"))
    gold = [s.record["target"]["commands"] for s in samples]
    generated = ["{" + text.replace("'''", "") + "}" for text in gold]
    generated[0] = "{error_input,error_input}"
    generated[1] = "unbraced output"
    generated[2] = "{a(1,2,3)}{error_input}"
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        namespace["compute_accuracy"](generated, gold)
    official = {}
    for line in output.getvalue().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            official[k.lower()] = float(v.strip())
    adapted = aggregate(judge(p, g) for p, g in zip(generated, gold))
    assert adapted == pytest.approx(official)


@pytest.mark.parametrize(
    "pred,gold,passed",
    [
        ("{a(),b()}", "b(),a()", True),
        ("{error_input}", "error_input,error_input", False),
        ("error_input", "error_input", False),
        ("{a()}\n{b()}", "a(),b(),", True),
        ("{}", "a()", False),
    ],
)
def test_official_edge_cases(pred, gold, passed):
    assert judge(pred, gold)["passed"] is passed


def test_public_tools_and_trusted_rollout(dataset):
    root, config = dataset
    sample = next(load(root / "train.jsonl"))
    before = deepcopy(sample.home)
    obs = observation(sample.home, "inspect_home", {})
    obs.clear()
    assert sample.home == before
    result = run_sample(sample, config, "initial_harness", Model(inspect=True), max_turns=3)
    assert result["finished"] and result["tool_calls"] == 2
    assert len(result["interface_trace"]["requests"]) == 2
    assert result["interface_trace"]["termination"] == "submitted"
    assert result["requests"][1]["messages"][-1]["role"] == "tool"
    assert "<example1>" in result["requests"][0]["messages"][1]["content"]
    assert "<example4>" in result["requests"][0]["messages"][1]["content"]
    assert "target" not in result["requests"][0]["messages"][1]["content"]
    assert evaluate([result], samples=[sample]).metrics["em"] == 1
    result["task_digest"] = "bad"
    with pytest.raises(ValueError, match="mismatch"):
        evaluate([result], samples=[sample])


def test_current_harness_preflight_checks_slots_schema_bounds_and_relative_math():
    module = importlib.import_module(
        "harnesses.homebench.generated.validation"
    )
    home = {
        "home_status": {
            "room": {
                "light": {
                    "state": "on",
                    "attributes": {
                        "brightness": {"value": 40, "lowest": 0, "highest": 100}
                    },
                }
            }
        },
        "method": [
            {
                "room_name": "room",
                "device_name": "light",
                "operation": "set_brightness",
                "parameters": [{"name": "brightness", "type": "int"}],
            }
        ],
    }
    validator = module.SubmissionValidator(
        instruction="Increase the light brightness in the room by 20, and repeat the invalid request.",
        home=home,
    )
    validator.set_task_type("multi2_mix")
    assert validator.validate("{room.light.set_brightness(60),error_input}")["valid"]
    wrong_math = validator.validate("{room.light.set_brightness(20),error_input}")
    assert not wrong_math["valid"] and "should be 60" in " ".join(wrong_math["errors"])
    assert not validator.validate("{room.light.set_brightness(101),error_input}")["valid"]
    assert not validator.validate("{room.light.set_brightness('60'),error_input}")["valid"]
    assert not validator.validate("{room.light.set_brightness(60)}")["valid"]


def test_current_harness_retries_a_preflight_rejection(dataset):
    root, config = dataset
    sample = next(load(root / "train.jsonl"))

    class CorrectingModel:
        def __init__(self):
            self.count = 0

        def complete(self, messages, tools):
            self.count += 1
            if self.count == 1:
                return call("final_answer", {"commands": "{room.light.missing()}"}, 1)
            assert json.loads(messages[-1]["content"])["status"] == "rejected_by_preflight"
            return call("final_answer", {"commands": "{room.light.turn_on()}"}, 2)

    row = run_sample(
        sample,
        config,
        "generated",
        CorrectingModel(),
        max_turns=2,
    )
    assert row["finished"] and row["generated"] == "{room.light.turn_on()}"
    assert row["tool_calls"] == 2


def test_budget_and_infrastructure_failures_are_distinct(dataset):
    root, config = dataset
    sample = next(load(root / "test.jsonl"))
    row = run_sample(sample, config, "initial_harness", Model(inspect=True), max_turns=1)
    assert not row["finished"] and row["error"] is None
    assert row["interface_trace"]["termination"] == "budget_exhausted"
    row = run_sample(sample, config, "initial_harness", Model(error=True), max_turns=1)
    assert not row["finished"] and row["error"]
    assert row["interface_trace"]["termination"] == "provider_error"


@pytest.mark.parametrize("mode", ["icl", "zero_shot"])
def test_actual_prompt_contains_household_context_and_configured_examples(dataset, mode):
    root, config = dataset
    config["harness"]["prompt_mode"] = mode
    sample = next(load(root / "train.jsonl"))
    sample.record["target"]["commands"] = "PRIVATE_REFERENCE_DO_NOT_SEND"
    row = run_sample(sample, config, "initial_harness", Model(), max_turns=1)
    prompt = row["requests"][0]["messages"][1]["content"]
    body = prompt.split("<household_index>\n", 1)[1].split("</household_index>", 1)[0]
    household_index = json.loads(body.split("\n", 1)[1])
    expected_index = {
        room: [device for device in state if device != "room_name"]
        for room, state in sample.home["home_status"].items()
    }
    assert household_index == expected_index
    assert "<home_state>" not in prompt
    assert "<device_methods>" not in prompt
    assert json.dumps(sample.home["home_status"], ensure_ascii=False) not in prompt
    assert json.dumps(sample.home["method"], ensure_ascii=False) not in prompt
    assert prompt.endswith(sample.instruction)
    examples = (ROOT / "harnesses/homebench/initial_harness/icl_examples.txt").read_text()
    assert (examples in prompt) == (mode == "icl")
    assert ("<icl_examples>" in prompt) == (mode == "icl")
    assert "PRIVATE_REFERENCE_DO_NOT_SEND" not in json.dumps(row["requests"])


def test_default_homebench_prompt_uses_icl_and_mode_change_invalidates_resume(dataset):
    root, config = dataset
    assert config["harness"]["prompt_mode"] == "icl"
    args = {
        "config": config,
        "data": root / "test.jsonl",
        "output_dir": root / "prompt_run",
        "model": "mock",
        "base_url": "http://fixture",
        "harness": "initial_harness",
        "max_turns": 1,
        "client_factory": lambda sample: Model(),
    }
    collect(**args)
    config["harness"]["prompt_mode"] = "zero_shot"
    with pytest.raises(ValueError, match="settings differ"):
        collect(**args, resume=True)


def test_resume_and_real_metric_artifact(dataset):
    root, config = dataset
    args = {
        "config": config,
        "data": root / "test.jsonl",
        "output_dir": root / "run",
        "model": "mock",
        "base_url": "http://fixture",
        "harness": "initial_harness",
        "max_turns": 2,
        "concurrency": 2,
    }
    result = collect(**args, client_factory=lambda s: Model())
    assert result["em"] == 1
    adapter = PLUGIN.round_adapter()
    artifact = adapter.external_artifact(root / "run", "test")
    assert adapter.artifact_complete(artifact, 3)
    assert adapter.validate_reuse(
        artifact,
        3,
        config=config,
        data=root / "test.jsonl",
        harness="initial_harness",
        model="mock",
        base_url="http://fixture",
        max_turns=2,
    )[0]

    def no_calls(sample):
        raise AssertionError("resume should make no model calls")

    collect(**args, resume=True, client_factory=no_calls)
    with pytest.raises(ValueError, match="resume refused"):
        collect(**{**args, "model": "changed"}, resume=True, client_factory=no_calls)


def test_strict_sft_and_shared_swift_adapter(dataset):
    root, config = dataset
    import yaml

    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    request = SamplingRequest(
        config_path=path,
        data_path=root / "train.jsonl",
        output_dir=root / "sample",
        policy_model="mock",
        base_url="http://fixture",
        harness_name="initial_harness",
        attempts_per_item=2,
        accepted_per_item=1,
        concurrency=2,
        max_turns=3,
        quality_tier="strict",
    )
    stats = sample_training(request, client_factory=lambda s: Model(inspect=True))
    assert stats.accepted == stats.strict_accepted == 3
    prepare_ms_swift_dataset(root / "sample/accepted_rollouts.jsonl", root / "swift.jsonl")
    assert len((root / "swift.jsonl").read_text().splitlines()) == 3
    from dataclasses import replace

    with pytest.raises(ValueError, match="clean train"):
        sample_training(replace(request, data_path=root / "test.jsonl"))
    stats = sample_training(
        replace(request, output_dir=root / "bad"), client_factory=lambda s: Model("{error_input}")
    )
    assert stats.accepted == 0 and stats.attempts == 6
    stats = sample_training(
        replace(request, resume=True), client_factory=lambda s: pytest.fail("should skip accepted")
    )
    assert stats.skipped_existing == 3


def test_selection_identity_and_f1_gate():
    rows = [{"sample_id": "a", "task_digest": "d"}]
    parent = EvaluationResult({"em": 0.5, "f1": 0.8}, rows)
    child = EvaluationResult({"em": 0.6, "f1": 0.8}, rows)
    assert select(parent, child, {}).accepted
    assert not select(parent, EvaluationResult({"em": 0.9, "f1": 0.7}, rows), {}).accepted
    assert not select(
        parent,
        EvaluationResult({"em": 0.9, "f1": 0.9}, [{"sample_id": "b", "task_digest": "d"}]),
        {},
    ).accepted
    assert not select(
        parent, EvaluationResult({"em": 0.9, "f1": 0.9}, rows, {"infrastructure_errors": 1}), {}
    ).accepted


def test_smoke_evidence_and_registration(dataset):
    from src.domains.registry import get_domain_plugin

    root, config = dataset
    assert get_domain_plugin("HomeBench").name == "homebench"
    assert PLUGIN.smoke_test(
        "initial_harness", package="harnesses.homebench", root=ROOT / "harnesses/homebench"
    ).passed
    collect(
        config=config,
        data=root / "test.jsonl",
        output_dir=root / "run",
        model="mock",
        base_url="http://fixture",
        harness="initial_harness",
        max_turns=2,
        concurrency=1,
        client_factory=lambda s: Model(),
    )
    PLUGIN.build_evidence(root / "run", root / "evidence.jsonl")
    text = (root / "evidence.jsonl").read_text()
    assert '"target"' not in text and '"reference"' not in text


def test_runtime_rejects_forged_model_response(dataset, monkeypatch):
    root, config = dataset
    import harnesses.homebench.initial_harness as harness

    original = harness.build_agent

    def build(settings, *, client):
        agent = original(settings, client=client)
        step = agent.step

        def forged():
            step()
            return call("final_answer", {"commands": "{room.light.turn_on()}"}, 99)

        agent.step = forged
        return agent

    monkeypatch.setattr(harness, "build_agent", build)
    sample = next(load(root / "test.jsonl"))
    row = run_sample(sample, config, "initial_harness", Model("{wrong()}"))
    assert row["error"] == "agent_modified_model_response"
    assert not row["finished"]


@pytest.mark.parametrize(
    "commands,expected",
    [
        ("room.light.turn_on()", "{room.light.turn_on()}"),
        ("{room.light.turn_on()}", "{room.light.turn_on()}"),
        ("error_input,error_input", "{error_input,error_input}"),
        ("{broken", "{broken"),
        ("", ""),
    ],
)
def test_submission_transport_preserves_commands(commands, expected):
    from src.domains.homebench.submission import compact_record, official_prediction

    record = {
        "finished": True,
        "error": None,
        "generated": commands,
        "messages": [call("final_answer", {"commands": commands})],
    }
    before = deepcopy(record)
    assert official_prediction(record) == expected
    assert official_prediction(compact_record(record)) == expected
    assert record == before


def test_unsubmitted_text_and_mismatched_tool_payload_are_not_repaired():
    from src.domains.homebench.submission import official_prediction

    row = {"finished": False, "generated": "error_input", "messages": []}
    assert official_prediction(row) == "error_input"
    row["finished"] = True
    with pytest.raises(ValueError, match="lacks its final tool"):
        official_prediction(row)
    row["messages"] = [call("final_answer", {"commands": "wrong()"})]
    with pytest.raises(ValueError, match="does not match"):
        official_prediction(row)


def test_bare_tool_commands_pass_eval_and_sft(dataset):
    root, config = dataset
    sample = next(load(root / "test.jsonl"))
    row = run_sample(sample, config, "initial_harness", Model("room.light.turn_on()"))
    assert row["generated"] == "room.light.turn_on()"
    assert evaluate([row], samples=[sample]).metrics["em"] == 1
    import yaml

    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    request = SamplingRequest(
        config_path=path,
        data_path=root / "train.jsonl",
        output_dir=root / "sft",
        policy_model="mock",
        base_url="http://fixture",
        harness_name="initial_harness",
        attempts_per_item=1,
        accepted_per_item=1,
        concurrency=1,
        max_turns=2,
        quality_tier="strict",
    )
    assert (
        sample_training(request, client_factory=lambda s: Model("room.light.turn_on()")).accepted
        == 3
    )


def test_runtime_provenance_still_covers_domain_dependencies_after_move(
    dataset, tmp_path, monkeypatch
):
    from src.domains.homebench.runtime import infer

    domain = tmp_path / "domain_source"
    (domain / "runtime").mkdir(parents=True)
    runtime_source = domain / "runtime/infer.py"
    runtime_source.write_text("# runtime\n")
    dependency = domain / "client.py"
    dependency.write_text("GENERATION_DEFAULT = 1\n")
    monkeypatch.setattr(infer, "__file__", str(runtime_source))
    config = load_config(ROOT / "configs/homebench.yaml")
    first = infer.identity(
        [], config, "initial_harness", model="policy", base_url="local", max_turns=8
    )
    dependency.write_text("GENERATION_DEFAULT = 2\n")
    second = infer.identity(
        [], config, "initial_harness", model="policy", base_url="local", max_turns=8
    )
    assert first["runtime_digest"] != second["runtime_digest"]
