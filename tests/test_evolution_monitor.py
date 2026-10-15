from scripts.monitor_evolution import deterministic_failure_classification


def test_monitor_stops_on_gpu_memory_failure():
    assert (
        deterministic_failure_classification(
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB"
        )
        == "memory"
    )


def test_monitor_stops_on_environment_failure():
    assert (
        deterministic_failure_classification(
            "RuntimeError: CUDA driver version is insufficient for CUDA runtime"
        )
        == "environment"
    )
    assert (
        deterministic_failure_classification("vLLM executable not found: vllm")
        == "environment"
    )


def test_monitor_stops_on_insufficient_free_gpu_memory():
    assert (
        deterministic_failure_classification(
            "Insufficient free GPU memory on physical GPU 4: stop without retry"
        )
        == "memory"
    )


def test_monitor_leaves_repository_traceback_for_codex_diagnosis():
    assert (
        deterministic_failure_classification(
            "Traceback: File 'src/runner/infer.py', line 42, in main\n"
            "TypeError: unexpected argument"
        )
        is None
    )
