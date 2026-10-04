"""Offline processor routing and optional deterministic GR00T inference."""

import os
import random

import numpy as np
import torch
from transformers import Qwen3VLProcessor


_original_from_pretrained = Qwen3VLProcessor.from_pretrained


@classmethod
def _from_pretrained(cls, model_name, *args, **kwargs):
    if model_name == "nvidia/Cosmos-Reason2-2B":
        local_path = os.environ.get("GROOT_COSMOS_PROCESSOR_PATH")
        if not local_path:
            raise RuntimeError(
                "GROOT_COSMOS_PROCESSOR_PATH is required for offline Cosmos loading"
            )
        model_name = local_path
    return _original_from_pretrained(model_name, *args, **kwargs)


Qwen3VLProcessor.from_pretrained = _from_pretrained


inference_seed_text = os.environ.get("GROOT_POLICY_INFERENCE_SEED")
if inference_seed_text:
    try:
        inference_seed = int(inference_seed_text)
    except ValueError as exc:
        raise RuntimeError("GROOT_POLICY_INFERENCE_SEED must be an integer") from exc

    from gr00t.policy.policy import BasePolicy

    _original_get_action = BasePolicy.get_action

    def _deterministic_get_action(self, observation, options=None):
        # GR00T N1.7 begins each flow-matching sample with torch.randn. Reset
        # every request so identical closed-loop observations receive identical
        # action noise even when one server evaluates multiple trials.
        random.seed(inference_seed)
        np.random.seed(inference_seed)
        torch.manual_seed(inference_seed)
        torch.cuda.manual_seed_all(inference_seed)
        return _original_get_action(self, observation, options)

    BasePolicy.get_action = _deterministic_get_action
    print(f"Deterministic GR00T inference seed enabled: {inference_seed}", flush=True)
