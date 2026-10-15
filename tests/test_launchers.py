"""Exercise real shell launchers with simulated round outputs, without GPU/API work."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(tmp_path):
    shutil.copytree(ROOT / "scripts", tmp_path / "scripts")
    (tmp_path / "src").symlink_to(ROOT / "src", target_is_directory=True)
    (tmp_path / "configs").mkdir()
    data = tmp_path / "train.jsonl"
    data.write_text('{"sample_id":"train-1"}\n')
    for domain, context, data_parallel, training_processes in (
        ("who_when", 40960, 2, 2),
        ("rbac", 8192, 2, 2),
        ("homebench", 65536, 2, 2),
    ):
        config = {
            "project": {"domain": domain},
            "model": {"base": "yaml-model"},
            "data": {"splits": {"train": str(data)}},
            "training": {
                "num_processes": training_processes,
                "sequence_parallel_size": 2 if domain == "homebench" else 1,
                "max_length": context,
            },
            "vllm": {
                "max_model_len": context,
                "gpu_memory_utilization": 0.9,
                "data_parallel_size": data_parallel,
            },
            "evolution": {"max_rounds": 3},
            "paths": {"output": str(tmp_path / "output" / domain)},
        }
        (tmp_path / "configs" / f"{domain}.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / "scripts/run_one_round.py").write_text("""
import argparse, json, os
from pathlib import Path
from src.config import load_config
p=argparse.ArgumentParser()
p.add_argument('--config', type=Path)
p.add_argument('--candidate-name')
p.add_argument('--round-number', type=int)
p.add_argument('--model-path')
p.add_argument('--dry-run', action='store_true')
a, rest=p.parse_known_args()
print('ROUND_ARGS='+json.dumps(dict(args=vars(a), rest=rest, cuda=os.environ.get('CUDA_VISIBLE_DEVICES')), default=str))
if not a.dry_run:
    c=load_config(a.config)
    out=Path(c['paths']['output'])/'rounds'/a.candidate_name
    out.mkdir(parents=True, exist_ok=True)
    if os.environ.get('FAIL_FIRST_ROUND'):
        marker=out/'failed-once'
        if not marker.exists():
            (out/'snapshot.json').write_text('{}')
            (out/'journal.json').write_text('{}')
            marker.write_text('1')
            raise SystemExit(9)
    (out/'round_summary.json').write_text(json.dumps({
        'deployment': {'model': 'next-policy', 'model_path': 'checkpoint-'+str(a.round_number)},
        'policy_updated': True,
    }))
""")
    backend = tmp_path / "backend"
    backend.write_text("#!/usr/bin/env bash\nexit 0\n")
    backend.chmod(0o755)
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "PYTHONPATH": str(tmp_path),
        "PYTHON": sys.executable,
        "CUDA_VISIBLE_DEVICES": "0,1",
        "ROUNDS": "1",
        "BACKEND": str(backend),
    }

    def run(*args, script="run.sh", **overrides):
        return subprocess.run(
            ["bash", str(tmp_path / "scripts" / script), *args],
            cwd=tmp_path,
            env=env | overrides,
            capture_output=True,
            text=True,
            check=False,
        )

    return tmp_path, run


def calls(process):
    assert process.returncode == 0, process.stdout + process.stderr
    return [
        json.loads(line.removeprefix("ROUND_ARGS="))
        for line in process.stdout.splitlines()
        if line.startswith("ROUND_ARGS=")
    ]


def test_ten_candidates_and_experiment_suffix(launcher):
    _, run = launcher
    rows = calls(run("--dry-run", MAX_HARNESS_ATTEMPTS="10",
                     COEVOX_RUN_SUFFIX="sensitivity_n10"))
    assert rows[0]["args"]["candidate_name"].endswith("_sensitivity_n10")
    rest = rows[0]["rest"]
    assert rest[rest.index("--max-harness-attempts") + 1] == "10"


def test_default_domain_dotenv_precedence_and_gpu_propagation(launcher):
    root, run = launcher
    (root / ".env").write_text(
        "PYTHON=/missing/python\nMODEL_PATH=dotenv-model\nCUDA_VISIBLE_DEVICES=7\n"
    )
    process = run("--dry-run", MODEL_PATH="caller-model")
    (call,) = calls(process)
    assert call["args"]["config"] == "configs/who_when.yaml"
    assert call["args"]["model_path"] == "caller-model"
    assert call["cuda"] == "0,1"


@pytest.mark.parametrize(
    "script,domain,context,data_parallel,training_processes",
    [
        ("run_homebench.sh", "homebench", "65536", "2", "2"),
        ("run_rbac.sh", "rbac", "8192", "2", "2"),
        ("run_whowhen.sh", "who_when", "65536", "2", "2"),
    ],
)
def test_domain_wrappers_override_dotenv_domain_config_and_use_yaml(
    launcher, script, domain, context, data_parallel, training_processes
):
    root, run = launcher
    (root / ".env").write_text("DOMAIN=who_when\nCONFIG=configs/who_when.yaml\n")
    (call,) = calls(run("--dry-run", script=script))
    assert call["args"]["config"] == f"configs/{domain}.yaml"
    assert call["rest"][call["rest"].index("--max-model-len") + 1] == context
    assert call["rest"][call["rest"].index("--training-max-length") + 1] == context
    assert call["rest"][call["rest"].index("--data-parallel-size") + 1] == data_parallel
    assert (
        call["rest"][call["rest"].index("--training-num-processes") + 1]
        == training_processes
    )
    expected_sequence_parallel = "2" if domain == "homebench" else "1"
    assert (
        call["rest"][call["rest"].index("--training-sequence-parallel-size") + 1]
        == expected_sequence_parallel
    )


@pytest.mark.parametrize(
    "script,label",
    [
        ("run_homebench.sh", "HomeBench"),
        ("run_rbac.sh", "RBAC"),
        ("run_whowhen.sh", "WhoWhen"),
    ],
)
def test_domain_wrappers_reject_project_venv_for_orchestrator(launcher, script, label):
    root, _ = launcher
    venv_bin = root / ".venvs/stale-runtime/bin"
    venv_bin.mkdir(parents=True)
    project_python = venv_bin / "python"
    project_python.symlink_to(sys.executable)

    process = subprocess.run(
        [
            "bash",
            "-c",
            (
                'PROJECT_ROOT="$1"; source "$1/scripts/lib/environment.sh"; '
                'coevox_global_runtime "$2"'
            ),
            script,
            str(root),
            label,
        ],
        cwd=root,
        env={
            **os.environ,
            "VIRTUAL_ENV": str(venv_bin.parent),
            "PYTHON": str(project_python),
            "PATH": f"{venv_bin}:{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode == 0, process.stdout + process.stderr
    runtime_line = next(
        line for line in process.stdout.splitlines() if line.startswith(f"{label} runtime:")
    )
    assert str(root / ".venvs") not in runtime_line


def test_domain_alias_and_explicit_config_precedence(launcher):
    _, run = launcher
    assert calls(run("whowhen", "--dry-run"))[0]["args"]["config"] == "configs/who_when.yaml"
    assert (
        calls(run("--config", "configs/homebench.yaml", "--dry-run"))[0]["args"]["config"]
        == "configs/homebench.yaml"
    )
    for args in (
        ("--domain", "rbac", "--config", "configs/homebench.yaml"),
        ("--config", "configs/homebench.yaml", "--domain", "rbac"),
    ):
        result = run(*args, "--dry-run")
        assert result.returncode == 2 and "conflicts" in result.stderr


@pytest.mark.parametrize(
    "args,env,message",
    [
        (("--continue-candidate", "child"), {}, "retired"),
        (("--resume", "round_01_missing"), {}, "Missing frozen resume artifact"),
        (("--resume", "../bad"), {}, "Cannot determine round"),
        (("--dry-run",), {"DATA_PARALLEL_SIZE": "3"}, "TP * DP exceeds"),
        (("--domain", "databench", "--dry-run"), {}, "Domain config not found"),
    ],
)
def test_invalid_launch_fails_before_running_rounds(launcher, args, env, message):
    _, run = launcher
    result = run(*args, **env)
    assert result.returncode == 2
    assert message in result.stderr
    assert "ROUND_ARGS=" not in result.stdout


def test_round_loop_propagates_selected_checkpoint(launcher):
    _, run = launcher
    first, second = calls(run(ROUNDS="2"))
    assert first["args"]["model_path"] == "yaml-model"
    assert second["args"]["model_path"] == "checkpoint-1"
    assert "--stop-after-round" in first["rest"]
    assert "--stop-after-round" not in second["rest"]


def test_resume_starts_from_frozen_round_without_manual_training_bypass(launcher):
    root, run = launcher
    name = "round_02_existing"
    directory = root / "output/who_when/rounds" / name
    directory.mkdir(parents=True)
    for artifact in ("journal.json", "snapshot.json"):
        (directory / artifact).write_text("{}")
    (call,) = calls(run("--resume", name, ROUNDS="2"))
    assert call["args"]["round_number"] == 2
    assert call["args"]["candidate_name"] == name
    assert "--resume-parent-artifacts" in call["rest"]


def test_bare_resume_selects_latest_run_then_round_in_config_output(launcher):
    root, run = launcher
    names = ['round_03_20261001T000000Z', 'round_01_20261002T000000Z',
             'round_02_20261002T000000Z']
    for name in names:
        directory = root / 'output/rbac/rounds' / name
        directory.mkdir(parents=True)
        for artifact in ('snapshot.json', 'journal.json'):
            (directory / artifact).write_text('{}')
    os.utime(root / 'output/rbac/rounds' / names[0], (2000000000, 2000000000))
    rows = calls(run('--resume', '--config', 'configs/rbac.yaml', '--dry-run', ROUNDS='2'))
    assert len(rows) == 1
    assert rows[0]['args']['candidate_name'] == names[-1]
    assert '--resume-parent-artifacts' in rows[0]['rest']


@pytest.mark.parametrize('latest_state', ['missing', 'incomplete', 'completed'])
def test_bare_resume_does_not_silently_start_or_use_stale_run(launcher, latest_state):
    root, run = launcher
    if latest_state != 'missing':
        for stamp in ('20261001T000000Z', '20261002T000000Z'):
            directory = root / 'output/who_when/rounds' / f'round_01_{stamp}'
            directory.mkdir(parents=True)
            (directory / 'snapshot.json').write_text('{}')
            (directory / 'journal.json').write_text('{}')
        if latest_state == 'completed':
            (directory / 'round_summary.json').write_text('{}')
        else:
            (directory / 'journal.json').unlink()
    result = run('--dry-run', '--resume')
    assert result.returncode == 2
    assert 'ROUND_ARGS=' not in result.stdout


def test_normal_launch_starts_fresh_despite_durable_incomplete_round(launcher):
    root, run = launcher
    name = "round_02_incomplete"
    directory = root / "output/who_when/rounds" / name
    directory.mkdir(parents=True)
    (directory / "journal.json").write_text(
        json.dumps({"incumbent_eval": {"id": "completed-parent-evaluation"}})
    )
    (directory / "snapshot.json").write_text("{}")

    first, second = calls(run(ROUNDS="2"))

    assert first["args"]["round_number"] == 1
    assert first["args"]["candidate_name"] != name
    assert second["args"]["round_number"] == 2
    assert "--resume-parent-artifacts" not in first["rest"]
    assert "--resume-parent-artifacts" not in second["rest"]


def test_normal_launch_does_not_resume_round_with_no_durable_stage(launcher):
    root, run = launcher
    empty = "round_01_empty"
    directory = root / "output/who_when/rounds" / empty
    directory.mkdir(parents=True)
    (directory / "snapshot.json").write_text("{}")
    (directory / "journal.json").write_text(
        json.dumps(
            {
                "identity": {"round_id": empty},
                "batches": [],
                "evaluations": {},
                "training": {},
                "costs": [{"stage": "freeze_and_identify"}],
                "model_exclusions": [],
            }
        )
    )

    process = run()
    (call,) = calls(process)

    assert call["args"]["candidate_name"] != empty
    assert "--resume-parent-artifacts" not in call["rest"]
    assert "Auto-resume" not in process.stdout


def test_failed_round_stops_until_explicit_resume(launcher):
    _, run = launcher
    failed = run(FAIL_FIRST_ROUND="1")
    assert failed.returncode == 9
    first = [
        json.loads(line.removeprefix("ROUND_ARGS="))
        for line in failed.stdout.splitlines()
        if line.startswith("ROUND_ARGS=")
    ]
    assert len(first) == 1
    name = first[0]["args"]["candidate_name"]
    assert "--resume-parent-artifacts" not in first[0]["rest"]
    assert f"rerun with --resume {name}" in failed.stdout

    (resumed,) = calls(run("--resume", name, FAIL_FIRST_ROUND="1"))
    assert resumed["args"]["candidate_name"] == name
    assert "--resume-parent-artifacts" in resumed["rest"]


def test_help_does_not_require_environment(launcher):
    _, run = launcher
    result = run("--help", PYTHON="/missing/python")
    assert result.returncode == 0 and "Default domain: who_when" in result.stdout


def test_empty_dotenv_values_preserve_config_defaults(tmp_path, monkeypatch):
    from src.config import load_config

    monkeypatch.delenv("EVO_TEST_MODEL", raising=False)
    (tmp_path / "configs").mkdir()
    (tmp_path / ".env").write_text("EVO_TEST_MODEL=\n")
    path = tmp_path / "configs/test.yaml"
    path.write_text("model: ${oc.env:EVO_TEST_MODEL,default-model}\n")
    assert load_config(path)["model"] == "default-model"


def test_fresh_example_dotenv_is_sourceable_and_uses_yaml_defaults(launcher):
    root, run = launcher
    shutil.copyfile(ROOT / ".env.example", root / ".env")
    (call,) = calls(run("--dry-run"))
    assert call["args"]["model_path"] == "yaml-model"
    assert call["cuda"] == "0,1"
