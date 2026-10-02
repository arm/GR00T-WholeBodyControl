"""Resolve the gated Cosmos processor from the pinned local Spark cache."""

import os

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
