# verl/tools/analyze_text_tool.py
# Copyright ...
from __future__ import annotations

import json
import logging
from typing import Any, Optional
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)

try:
    # Optional dependency; we only import if a tokenizer name is provided
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore


class AnalyzeTextTool(BaseTool):
    """
    VERL tool: analyzeText

    Purpose
    -------
    - Analyze an attached document and report basic stats.
    - Matches orchestrator_es.ToolLibrary.analyzeText() keys:
        { "file_name": ..., "total_tokens": ... }
      plus a few non-breaking extras (chars/lines/words).

    Create-time kwargs (preferred)
    ------------------------------
    - document: str            # REQUIRED (the full text to analyze)
    - file_name: Optional[str] # optional filename for reporting
    - tokenizer_name: Optional[str]
        e.g., "Qwen/Qwen2.5-7B-Instruct" or any HF tokenizer id
        If absent or HF not installed, we use a simple whitespace tokenizer.

    Execute-time parameters (optional)
    ----------------------------------
    - detail: "basic" | "full" (default: "basic")
        "basic": return file_name & total_tokens only (compat with your current agent)
        "full":  return extra fields (char_count, line_count, word_count)

    Reward
    ------
    - This tool is informational; step reward is 0.0.
    """

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        # Build a default schema if one is not injected
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema.model_validate(
                {
                    "type": "function",
                    "function": {
                        "name": "analyzeText",
                        "description": "Analyze the attached document and return token count and basic statistics.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "detail": {
                                    "type": "string",
                                    "enum": ["basic", "full"],
                                    "description": "Level of detail to return. Defaults to 'basic'.",
                                }
                            },
                            "required": [],
                        },
                    },
                }
            )
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict[str, Any]] = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        document: Optional[str] = None,
        file_name: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        """
        Store the document and (optionally) construct a tokenizer for accurate token counts.
        Also supports pulling values from kwargs["create_kwargs"] for compatibility with VERL runners.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        if document is None:
            document = kwargs.get("create_kwargs", {}).get("document")
        if file_name is None:
            file_name = kwargs.get("create_kwargs", {}).get("file_name", "attached_document.txt")
        if tokenizer_name is None:
            tokenizer_name = kwargs.get("create_kwargs", {}).get("tokenizer_name")

        if not isinstance(document, str) or len(document) == 0:
            msg = "analyzeText.create: 'document' (non-empty string) is required."
            logger.warning(msg)
            self._instance_dict[instance_id] = {"error": msg}
            return instance_id, ToolResponse(text=msg)

        # Try to prepare tokenizer if a name is provided and transformers is available
        tokenizer = None
        if tokenizer_name and AutoTokenizer is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            except Exception as e:  # degrade gracefully
                logger.warning(f"analyzeText.create: failed to load tokenizer '{tokenizer_name}': {e}")

        self._instance_dict[instance_id] = {
            "document": document,
            "file_name": file_name or "attached_document.txt",
            "tokenizer_name": tokenizer_name,
            "tokenizer": tokenizer,  # may be None
            "last_result": None,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """
        Compute stats and return them. Zero step reward.
        """
        st = self._instance_dict.get(instance_id)
        if st is None or "document" not in st:
            msg = "analyzeText.execute: invalid instance or document missing. Did you call create() with a document?"
            return ToolResponse(text=msg), 0.0, {"ok": False}

        detail = parameters.get("detail", "basic")
        if detail not in ("basic", "full"):
            detail = "basic"

        doc: str = st["document"]
        file_name: str = st["file_name"]
        tokenizer = st.get("tokenizer")

        # Token count mirroring your orchestrator: prefer HF tokenizer if available,
        # add_special_tokens=False to match causal input length semantics.
        if tokenizer is not None:
            try:
                enc = tokenizer(doc, return_offsets_mapping=False, add_special_tokens=False)
                total_tokens = len(enc.get("input_ids", []))
            except Exception as e:
                logger.warning(f"analyzeText.execute: tokenizer failed, falling back to whitespace: {e}")
                total_tokens = len(doc.split())
        else:
            # Simple fallback (not byte- or BPE-accurate, but deterministic)
            total_tokens = len(doc.split())

        result_basic = {
            "file_name": file_name,
            "total_tokens": int(total_tokens),
        }

        if detail == "basic":
            payload = result_basic
        else:
            payload = {
                **result_basic,
                "char_count": len(doc),
                "line_count": doc.count("\n") + 1 if doc else 0,
                "word_count": len(doc.split()),
            }

        st["last_result"] = payload
        text = json.dumps(payload, ensure_ascii=False)
        metrics = {"ok": True, "detail": detail}
        return ToolResponse(text=text), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # No environment reward from this tool; keep it zero.
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
