from src.runner.model_client import OpenAICompatibleClient


def build_client(config, *, model, base_url, sampling=False):
    opts = dict(config.get("rollout") or {})
    opts.update(
        {
            k: v
            for k, v in (config.get("model") or {}).items()
            if k in {"max_tokens", "temperature", "top_p", "enable_thinking", "reasoning_effort"}
        }
    )
    if sampling:
        opts["temperature"] = (config.get("rejection_sampling") or {}).get("temperature", 0.7)
    return OpenAICompatibleClient(
        seed=opts.get("seed"),
        model=model,
        base_url=base_url,
        api_key=(config.get("model") or {}).get("api_key"),
        temperature=opts.get("temperature", 0.0),
        top_p=opts.get("top_p", 1.0),
        top_k=opts.get("top_k"),
        max_tokens=opts.get("max_tokens", 4096),
        reasoning_effort=opts.get("reasoning_effort", "medium"),
        enable_thinking=opts.get("enable_thinking", True),
        timeout=opts.get("timeout_seconds", 300),
        tool_choice=str(opts.get("tool_choice", "required")),
    )
