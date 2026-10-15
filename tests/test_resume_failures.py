"""Recovery regressions that do not require GPU or model dependencies."""
import copy
import subprocess
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('recover', [True, False])
def test_meta_timeout_preserves_output_and_retries(tmp_path, monkeypatch, recover):
    from src.meta_evolver import context

    monkeypatch.setattr(context, '_require_command', lambda command: None)
    calls = []

    def run(command, **kwargs):
        result = kwargs['cwd'] / 'final_message.md'
        assert not result.exists()
        calls.append(command)
        if recover and len(calls) == 2:
            result.write_text('recovered')
            return subprocess.CompletedProcess(command, 0, '', '')
        result.write_text('partial result')
        raise subprocess.TimeoutExpired(command, 1800, output=b'partial events\xff', stderr=b'error')

    monkeypatch.setattr(context.subprocess, 'run', run)
    output = tmp_path / 'stage'
    kwargs = dict(config={'meta_evolver': {'backend': 'codex', 'codex': {'command': 'fake'},
                                         'text_output_attempts': 2}},
                  project_root=tmp_path, output_dir=output, inputs={}, prompt='Return result')
    if recover:
        assert context.run_meta_text(**kwargs).read_text() == 'recovered'
    else:
        with pytest.raises(subprocess.TimeoutExpired):
            context.run_meta_text(**kwargs)
        assert not (output / 'complete.json').exists()
        assert not (output / 'workspace' / 'final_message.md').exists()
    assert len(calls) == 2
    assert (output / 'attempt_01.stdout.log').read_text() == 'partial events\ufffd'
    assert (output / 'attempt_01.stderr.log').read_text() == 'error'


@pytest.mark.parametrize('changed', [None, 'effective_weights_hash', 'architecture_hash',
                                    'tokenizer_hash', 'layout_hash', 'precision'])
def test_model_metadata_refresh_requires_identical_content(tmp_path, monkeypatch, changed):
    from src.meta_evolver import round_runtime as module

    checkpoint = tmp_path / 'model.bin'
    checkpoint.write_bytes(b'unchanged')
    st = checkpoint.stat()
    model = dict(id='model', path=str(tmp_path), effective_weights_hash='weights',
                 architecture_hash='architecture', tokenizer_hash='tokenizer',
                 layout_hash='layout', precision=['torch.float32'],
                 file_signature={str(checkpoint): [st.st_size, st.st_mtime_ns, st.st_ctime_ns - 1]})
    current = copy.deepcopy(model)
    current['file_signature'][str(checkpoint)][2] = st.st_ctime_ns
    if changed:
        current[changed] = 'different'
    verified = []

    def describe(*args, **kwargs):
        verified.append(args)
        return current

    monkeypatch.setattr(module, 'describe_model', describe)
    monkeypatch.setattr(module, 'build_vllm_command', lambda **kwargs: ['server'])
    loads = []
    runtime = SimpleNamespace(loaded_model=None, root=tmp_path, weights_root=tmp_path,
                              stop=lambda: None, vllm_settings={}, round_dir=tmp_path,
                              base_url='unused', cost=lambda *a, **k: None,
                              services=SimpleNamespace(start_vllm=lambda *a, **k: loads.append(k)))
    original = copy.deepcopy(model['file_signature'])
    if changed:
        with pytest.raises(ValueError, match='model content changed'):
            module.Runtime.ensure_model(runtime, model)
        assert not loads
        assert model['file_signature'] == original
    else:
        module.Runtime.ensure_model(runtime, model)
        assert len(loads) == 1
        assert model['file_signature'] == current['file_signature']
    assert len(verified) == 1
