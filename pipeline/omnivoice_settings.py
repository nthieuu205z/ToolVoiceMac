"""Immutable per-job scalars and lightweight installed-API discovery."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class OmniVoiceSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)

    speed: float = Field(default=1.0, ge=0.5, le=1.5)
    # Manual targets stay within upstream's default short-audio threshold.
    # Blank remains automatic; this is not a cap on automatic long-text jobs.
    duration: float | None = Field(default=None, gt=0, le=30.0)
    num_step: int = Field(default=32, ge=4, le=64)
    guidance_scale: float = Field(default=2.0, ge=0, le=4)
    denoise: bool = True
    preprocess_prompt: bool = True
    postprocess_output: bool = True

    def generation_kwargs(self) -> dict:
        return self.model_dump(exclude={"speed", "duration"})


def installed_capabilities() -> dict:
    """Read source, not import torch/transformers or initialize any model.

    Conservatively disable the panel for packages whose API cannot be verified.
    Legacy synthesis remains available with its existing defaults.
    """
    try:
        spec = importlib.util.find_spec("omnivoice")
        if spec is None or not spec.origin:
            return {"supported": False, "defaults": {}}
        source = Path(spec.origin).parent / "models" / "omnivoice.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        config = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OmniVoiceGenerationConfig")
        defaults = {
            node.target.id: ast.literal_eval(node.value)
            for node in config.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
            and node.target.id in OmniVoiceSettings.model_fields and node.value is not None
        }
        generate = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "generate")
        args = {arg.arg for arg in generate.args.args + generate.args.kwonlyargs}
        required = set(OmniVoiceSettings.model_fields) - {"speed", "duration"}
        if not required <= defaults.keys() or not {"speed", "duration", "generation_config"} <= args:
            return {"supported": False, "defaults": {}}
        return {"supported": True, "defaults": OmniVoiceSettings(**defaults).model_dump()}
    except (OSError, SyntaxError, ValueError, TypeError, StopIteration, ImportError):
        return {"supported": False, "defaults": {}}
