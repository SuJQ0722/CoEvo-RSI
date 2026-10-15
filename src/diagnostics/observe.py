"""Capture model interfaces before a harness repairs or mutates responses.

Keep this module standard-library-only: domain execution environments need not
install the controller's offline validation dependencies.
"""

from copy import deepcopy

TRACE_VERSION = 1
_OBSERVED_RESPONSE_FIELDS = {"role", "content", "tool_calls"}


def _interface_response(response):
    return deepcopy(
        {k: v for k, v in response.items() if k in _OBSERVED_RESPONSE_FIELDS}
        if isinstance(response, dict)
        else response
    )


class ObservedModel:
    def __init__(self, model):
        self._model = model
        self.requests = []

    def __getattr__(self, name):
        return getattr(self._model, name)

    def complete(self, messages, tools, **kwargs):
        event = {"tools": deepcopy(tools)}
        self.requests.append(event)

        def capture(response):
            # Reasoning and prompts are not needed to validate the interface.
            event["response"] = _interface_response(response)

        try:
            observed_complete = getattr(self._model, "complete_observed", None)
            if callable(observed_complete):
                response = observed_complete(messages, tools, observe_response=capture, **kwargs)
                event["response_stage"] = "provider"
            else:
                response = self._model.complete(messages, tools, **kwargs)
                capture(response)
                event["response_stage"] = "client"
        except Exception as exc:
            event["error_type"] = type(exc).__name__
            raise
        raw = event.get("response")
        effective = _interface_response(response)
        event["transport_repaired"] = repaired = (
            isinstance(raw, dict)
            and isinstance(effective, dict)
            and raw.get("tool_calls") != effective.get("tool_calls")
        )
        if repaired:
            # This is the interface the harness actually consumes. Persist it
            # only when it differs from the provider shape; reasoning text is
            # intentionally excluded from both copies.
            event["effective_response"] = effective
        return response

    def trace(self, termination=None):
        return {
            "version": TRACE_VERSION,
            "requests": deepcopy(self.requests),
            "termination": termination,
        }
