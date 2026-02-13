# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import copy
import json
import logging
import os
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    MultiTrajectoryAgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

import numpy as np

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# Add a console handler if not already present to ensure logs are visible
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    _handler.flush = lambda: None  # Enable immediate flushing
    logger.addHandler(_handler)
    logger.propagate = True  # Ensure logs propagate to root logger


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"

def strip_the_last_thinking_tag(response_ids: list[int]) -> list[int]:
    """
    Strip the last <think> ... </think> tag from the response_ids.
    
    ['<|im_start|>', 'assistant', 'Ċ', '<think>', 'ĊĊ', '</think>', 'ĊĊ']
    --> [151644, 77091, 198, 151667, 271, 151668, 271]
    
    So each time we find the last pattern appeared in the prompt_ids
    and remove the thinking tags: [151667, 271, 151668, 271]
    """
    


def render_context(conv_history: list[dict[str, Any]], memory_notes: dict[str, dict], deleted_msg_ids: set[int]) -> list[dict[str, Any]]:
    """
    Build a rendered view of the message history by taking into account editions invoked by StateLM.
    We also need to set the response_mask and response_logprobs to empty lists for the current turn.
    """
    conv_history_cp = copy.deepcopy(conv_history)
    memory_notes_cp = copy.deepcopy(memory_notes)
    deleted_msg_ids_cp = copy.deepcopy(deleted_msg_ids)
    
    stub_message = "Content has been deleted to save space."
    rendered_context: list[dict[str, Any]] = []

    if memory_notes_cp:
        notes_summary = (
            f"\n\n<external_memory>\n## Available Notes\n"
            f"{"\n".join([f"- **{key}**: {data['summary']}" for key, data in memory_notes_cp.items()])}\n</external_memory>"
        )
    else:
        notes_summary = (
            f"\n\n<external_memory>\n## Available Notes\n"
            "No notes recorded.\n</external_memory>"
        )

    first_user_msg_seen = False
    for idx, msg in enumerate(conv_history_cp):
        role = msg.get("role")

        if role == "system":
            # Pass through system messages unchanged
            rendered_context.append({"role": "system", "content": msg["content"]})

        elif role == "user":
            # Add notes_summary only to the first user message
            if not first_user_msg_seen:
                text = msg["content"] + notes_summary
                first_user_msg_seen = True
            else:
                text = msg["content"]
            rendered_context.append({"role": "user", "content": text})

        elif role == "assistant":
            msg_id = msg["msg_id"]
            tool_calls = msg.get("tool_calls", [])

            if msg_id in deleted_msg_ids_cp:
                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    stub_tool_calls = []
                    for tc in tool_calls:
                        fn = tc.get("function") or {}
                        name = fn.get("name") or ""

                        stub_tool_calls.append({
                            "id": tc.get("id"),
                            "type": "function",
                            "function": {
                                "name": name,
                                # arguments MUST be a JSON string
                                "arguments": json.dumps(
                                    {"message": stub_message},
                                ),
                            },
                        })
                    rendered_context.append({
                        "role": "assistant",
                        "content": stub_message,
                        "tool_calls": stub_tool_calls,
                    })
                else:
                    rendered_context.append({
                        "role": "assistant",
                        "content": stub_message,
                    })
            # Undeleted messages
            else:
                assistant_content = msg["content"]
                assert len(assistant_content) == 1, "Expected single content block in assistant message."
                raw_text = assistant_content[0]["text"]
                cleaned_text = raw_text.strip()
                assistant_msg = {
                    "role": "assistant",
                    "content": (cleaned_text if cleaned_text else ""),
                }
                # If this assistant turn contained tool calls, forward them verbatim
                if tool_calls:
                    assistant_msg["tool_calls"] = msg["tool_calls"]
                rendered_context.append(assistant_msg)

        elif role == "tool":
            msg_id = msg["msg_id"]
            msg_id_ia = msg["msg_id(invoking_assistant)"]
            
            # Handle both string and dict content types
            original_content = msg["content"]
            if isinstance(original_content, str):
                # Try to parse as JSON, or wrap as message
                try:
                    tool_result_content_cp = json.loads(original_content)
                except (json.JSONDecodeError, TypeError):
                    tool_result_content_cp = {"message": original_content}
            else:
                tool_result_content_cp = copy.deepcopy(original_content)
            
            tool_result_content_cp["msg_id"] = msg_id
            tool_result_content_cp["msg_id(invoking_assistant)"] = msg_id_ia
            
            if msg_id in deleted_msg_ids_cp:
                tool_name = msg.get("tool_name", "unknown")
                tool_result_content_cp = {
                    "msg_id": msg_id,
                    "msg_id(invoking_assistant)": msg_id_ia,
                    "status": "success",
                    "message": stub_message,
                    "original_tool": tool_name
                }
            
            rendered_context.append(
                {
                    "role": "tool",
                    "content": json.dumps(tool_result_content_cp, ensure_ascii=False),
                }
            )
    return rendered_context

def strip_the_last_think_tags(response_ids: list[int]) -> list[int]:
    """
    Strip the last <think> ... </think> tags from the response_ids.
    
    Target Logic:
    1. Find the last occurrence of the header: [151644, 77091, 198, 151667, 271]
       (<|im_start|>assistant\n<think>\n\n)
    2. Find the first occurrence of the footer AFTER that header: [151668, 271]
       (</think>\n\n)
    3. Remove everything from <think> (inclusive) to the footer (inclusive).
    """
    
    # Hardcoded patterns based on the description
    # Header: <|im_start|>, assistant, \n, <think>, \n\n
    HEADER_IDS = [151644, 77091, 198, 151667, 271]
    # Footer: </think>, \n\n
    FOOTER_IDS = [151668, 271]
    
    n = len(response_ids)
    h_len = len(HEADER_IDS)
    f_len = len(FOOTER_IDS)
    
    if n < h_len + f_len:
        return response_ids

    # 1. Search BACKWARDS for the start of the Header pattern
    # We look for the last occurrence.
    start_idx = -1
    for i in range(n - h_len, -1, -1):
        # Direct comparison is faster than slicing [i:i+5]
        if (response_ids[i] == HEADER_IDS[0] and 
            response_ids[i+1] == HEADER_IDS[1] and 
            response_ids[i+2] == HEADER_IDS[2] and 
            response_ids[i+3] == HEADER_IDS[3] and 
            response_ids[i+4] == HEADER_IDS[4]):
            start_idx = i
            break
            
    if start_idx == -1:
        return response_ids

    # 2. Search FORWARD for the Footer pattern
    # The search must start after the full header to ensure we capture the content in between.
    search_pos = start_idx + h_len
    end_idx = -1
    
    for i in range(search_pos, n - f_len + 1):
        if (response_ids[i] == FOOTER_IDS[0] and 
            response_ids[i+1] == FOOTER_IDS[1]):
            end_idx = i
            break

    # 3. Strip if both parts are found
    if end_idx != -1:
        # We want to keep:
        # prefix: up to start_idx + 3. 
        # (HEADER_IDS indices: 0, 1, 2 are kept. 3 is <think>, which starts removal)
        keep_until = start_idx + 3
        
        # We want to resume after the footer
        resume_at = end_idx + f_len
        
        return response_ids[:keep_until] + response_ids[resume_at:]

    return response_ids

class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: Any,
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
        document_content: Optional[str] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}
        self.document_content = document_content

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        # ===========================
        # StateLM Customizations
        # ===========================
        # document_content is already set above from parameter
        
        # NEW: Editable state tracking
        # full_history will be populated in the run() method with proper msg_id tracking
        self.full_history: list[dict] = []
        self.deleted_msg_ids: set[int] = set()  # Track deleted message IDs
        self.msg_id_counter: int = 0  # Assign unique IDs to each message
        
        # NEW: View snapshots for causal log-prob
        self.emission_views: list[list[int]] = []  # prompt_ids when each assistant turn started
        self.assistant_turn_boundaries: list[tuple[int, int]] = []  # (start_idx, end_idx) in response_ids
        
        # Track snapshots for deleteContext operations
        self.trajectory_snapshots: list[dict[str, Any]] = []  # Store trajectory state before each delete
        
        # Track whether we had a deleteContext operation
        self.had_delete_operation: bool = False

        self.notes: dict[str, dict] = {}
        
        # StateLM State Manager for document operations
        self.doc_state_manager = None


@register("statelm_tool_agent")
class StatelmToolAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level StatelmToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.min_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.get("min_assistant_turns", None)
        cls.below_min_turns_reward = config.actor_rollout_ref.rollout.multi_turn.get("below_min_turns_reward", -1.0)
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        print(f"Initialized tools: {cls.tools}")

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )
        # Initialize interactions from config file
        cls.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if cls.interaction_config_file:
            cls.interaction_map: dict[str, BaseInteraction] = cls._initialize_interactions(cls.interaction_config_file)
        
        # Max model context length for protection
        cls.max_model_length = config.actor_rollout_ref.rollout.get("max_model_len", 8192)
        cls.max_response_length = config.actor_rollout_ref.rollout.multi_turn.get("single_turn_max_tokens", 2048)
        cls.exceed_length_penalty = config.actor_rollout_ref.rollout.multi_turn.get("exceed_length_penalty", -1.0)
        cls.context_len_or_turn_exceeded = False
        cls.model_type = config.actor_rollout_ref.rollout.multi_turn.get("model_type") # Be aware of the thinking tag problem <think> ... </think> 

        # Trajectory dumping configuration
        cls.dump_trajectories_enabled = config.actor_rollout_ref.rollout.multi_turn.get("dump_trajectories_enabled", False)
        cls.dump_trajectories_dir = config.actor_rollout_ref.rollout.multi_turn.get("dump_trajectories_dir", "trajectories")
        cls.dump_trajectories_freq = config.actor_rollout_ref.rollout.multi_turn.get("dump_trajectories_freq", 1)
        cls._trajectory_dump_counter = 0
        if cls.dump_trajectories_enabled:
            os.makedirs(cls.dump_trajectories_dir, exist_ok=True)
            print(f"Trajectory dumping enabled. Saving to: {cls.dump_trajectories_dir}")
        
        logger.info(f"[StatelmToolAgentLoop] ======== Initialized StateLM Agent Loop ======== ")

    def _dump_trajectory(
        self,
        agent_data: AgentData,
        trajectories: list[AgentLoopOutput],
        request_id: str,
        **kwargs,
    ) -> None:
        """Dump trajectory data to a JSON file for debugging and analysis.
        
        Args:
            agent_data: The agent data containing full conversation history
            trajectories: List of trajectory outputs
            request_id: Unique request identifier
            **kwargs: Additional context from the rollout
        """
        if not self.dump_trajectories_enabled:
            return
        
        self.__class__._trajectory_dump_counter += 1
        if self.__class__._trajectory_dump_counter % self.dump_trajectories_freq != 0:
            return
        
        try:
            # Prepare trajectory data
            trajectory_data = {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "user_turns": agent_data.user_turns,
                "assistant_turns": agent_data.assistant_turns,
                "had_delete_operation": agent_data.had_delete_operation,
                "deleted_msg_ids": list(agent_data.deleted_msg_ids),
            }
            
            # Add conversation history (messages)
            messages_for_dump = []
            for msg in agent_data.full_history:
                msg_copy = dict(msg)
                # Convert content to string if needed for JSON serialization
                if isinstance(msg_copy.get("content"), list):
                    # Extract text from content blocks
                    text_parts = []
                    for block in msg_copy["content"]:
                        if isinstance(block, dict) and "text" in block:
                            text_parts.append(block["text"])
                        elif isinstance(block, str):
                            text_parts.append(block)
                    msg_copy["content"] = "\n".join(text_parts)
                messages_for_dump.append(msg_copy)
            trajectory_data["messages"] = messages_for_dump
            
            # Add trajectory outputs (decoded if tokenizer available)
            trajectory_outputs = []
            for i, traj in enumerate(trajectories):
                traj_info = {
                    "index": i,
                    "is_snapshot": traj.extra_fields.get("is_snapshot", False) if traj.extra_fields else False,
                    "num_turns": traj.num_turns,
                    "prompt_length": len(traj.prompt_ids) if traj.prompt_ids else 0,
                    "response_length": len(traj.response_ids) if traj.response_ids else 0,
                    "response_mask_sum": sum(traj.response_mask) if traj.response_mask else 0,
                }
                # Decode tokens to text if tokenizer available
                if self.tokenizer:
                    try:
                        traj_info["prompt_text"] = self.tokenizer.decode(traj.prompt_ids, skip_special_tokens=False)
                        traj_info["response_text"] = self.tokenizer.decode(traj.response_ids, skip_special_tokens=False)
                    except Exception as e:
                        logger.warning(f"Failed to decode trajectory tokens: {e}")
                trajectory_outputs.append(traj_info)
            trajectory_data["trajectory_outputs"] = trajectory_outputs
            
            # Add extra info from kwargs if available
            if "extra_info" in kwargs:
                extra_info = kwargs["extra_info"]
                if isinstance(extra_info, dict):
                    # Filter to JSON-serializable fields
                    safe_extra = {}
                    for k, v in extra_info.items():
                        if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                            safe_extra[k] = v
                    trajectory_data["extra_info"] = safe_extra
            
            # Write to file
            filename = f"trajectory_{request_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.dump_trajectories_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(trajectory_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"[StatelmToolAgentLoop] Dumped trajectory to: {filepath}")
        except Exception as e:
            logger.warning(f"[StatelmToolAgentLoop] Failed to dump trajectory: {e}")

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> MultiTrajectoryAgentLoopOutput:
        logger.info(f"[StatelmToolAgentLoop] Starting run()")
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        # NEW: Extract document content from kwargs (not in messages)
        document_content = kwargs.get("document_content", "")
        logger.info(f"[StatelmToolAgentLoop] document_content length: {len(document_content) if document_content else 0}")
        
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            document_content=document_content,  # NEW
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
        )

        # NEW: Initialize full_history with all initial messages (system + user)
        agent_data.msg_id_counter = 0
        for msg in messages:
            agent_data.full_history.append({
                "role": msg.get("role"),
                "content": msg.get("content"),
                "msg_id": agent_data.msg_id_counter
            })
            agent_data.msg_id_counter += 1
        
        # Initialize StateLM state manager if document_content is provided
        if document_content:
            from verl.tools.statelm_tools import DocStateManager
            agent_data.doc_state_manager = DocStateManager(self.tokenizer, document_content)
        else:
            logger.warning(f"[StatelmToolAgentLoop] document_content is not provided, using empty document content")
            agent_data.doc_state_manager = DocStateManager(self.tokenizer, " ")

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                state = AgentState.TERMINATED

        # Handle TERMINATED state
        # Finalize output - handle multiple trajectories for StateLM
        # suppose prompt_ids is [1,2,(p) 3,4,5,(r1) 6,7,8,(tool) 9,10,11,12 (r2)], response_mask is [1,1,1,0,0,0,1,1,1,1], response_ids is [9,10,11,12]
        # then trajectory_response_ids is [3,4,5,6,7,8,9,10,11,12] and trajectory_prompt_ids is [1,2]
        trajectories = []
        
        # First, add all snapshot trajectories (for StateLM deleteContext)
        if agent_data.trajectory_snapshots:
            for idx, snapshot in enumerate(agent_data.trajectory_snapshots):
                snapshot_output = self._create_trajectory_output(
                    prompt_ids=snapshot["prompt_ids"],
                    response_ids=snapshot["response_ids"],
                    response_mask=snapshot["response_mask"],
                    response_logprobs=snapshot["response_logprobs"],
                    num_turns=snapshot["num_turns"],
                    is_snapshot=True,
                    metrics=agent_data.metrics,
                )
                if snapshot_output is not None:
                    trajectories.append(snapshot_output)
        
        # Add the final trajectory
        last_prompt_ids = copy.deepcopy(agent_data.prompt_ids)
        last_response_ids = copy.deepcopy(agent_data.response_ids)
        last_response_mask = copy.deepcopy(agent_data.response_mask)
        last_response_logprobs = copy.deepcopy(agent_data.response_logprobs)
        
        # Calling deleteContext and the tool result overflows the model context length?
        if len(last_response_mask) == 0:
            logger.warning(f"[StatelmToolAgentLoop] response_mask is empty, but have {len(trajectories)} snapshots and {agent_data.assistant_turns} turns.")
            if len(trajectories) > 0:
                logger.info(f"[StatelmToolAgentLoop] Returning {len(trajectories)} valid snapshot trajectories despite empty final response_mask.")
                return MultiTrajectoryAgentLoopOutput(trajectories=trajectories)
            else:
                logger.warning(f"[StatelmToolAgentLoop][WARNING] Return empty MultiTrajectoryAgentLoopOutput because no valid trajectories exist.")
                return MultiTrajectoryAgentLoopOutput(trajectories=[])

        # Add the final trajectory
        if self.context_len_or_turn_exceeded:
            reward_score = self.exceed_length_penalty
        elif (self.min_assistant_turns is not None) and (agent_data.assistant_turns < self.min_assistant_turns):
            reward_score = self.below_min_turns_reward
        else:
            reward_score = None
        
        last_traj_output = self._create_trajectory_output(
            prompt_ids=last_prompt_ids,
            response_ids=last_response_ids,
            response_mask=last_response_mask,
            response_logprobs=last_response_logprobs,
            num_turns=agent_data.assistant_turns,
            is_snapshot=False,
            reward_score=reward_score,
            metrics=agent_data.metrics,
            enforce_output=True,
        )
        trajectories.append(last_traj_output)
        
        # Cleanup state manager if needed
        if agent_data.doc_state_manager:
            try:
                # Run blocking Elasticsearch call in executor to avoid blocking event loop
                # Add timeout to prevent hanging forever if ES is slow/unresponsive
                await asyncio.wait_for(
                    self.loop.run_in_executor(
                        None,
                        lambda: agent_data.doc_state_manager.clear_current_document()
                    ),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.warning("Elasticsearch cleanup timed out after 30 seconds")
            except Exception as e:
                logger.warning(f"Error clearing state manager: {e}")
        
        # Dump trajectories if enabled
        if self.dump_trajectories_enabled:
            self._dump_trajectory(agent_data, trajectories, request_id, **kwargs)
        
        return MultiTrajectoryAgentLoopOutput(trajectories=trajectories)

    def _create_trajectory_output(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float],
        num_turns: int,
        reward_score: float = None,
        is_snapshot: bool = False,
        metrics: Optional[dict[str, Any]] = None,
        enforce_output: bool = False,
    ) -> Optional[AgentLoopOutput]:
        """
        Create an AgentLoopOutput from trajectory data.
        
        Args:
            prompt_ids: Full prompt token ids (includes response tokens)
            response_ids: Response token ids for the last turn (should be a suffix of prompt_ids)
            response_mask: Mask indicating which tokens are response tokens (should be a suffix of prompt_ids)
            response_logprobs: Log probabilities for response tokens (same length as response_ids)
            num_turns: Number of assistant turns in this trajectory
            reward_score: Reward score for this trajectory
            is_snapshot: Whether this is a snapshot trajectory
            metrics: Metrics for this trajectory
            enforce_output: Whether to enforce the output to be non-empty
        Returns:
            AgentLoopOutput for this trajectory, None if the trajectory is not valid
        """
        # Debugging: Assert that response_ids is a suffix of prompt_ids
        if len(response_ids) > 0:
            expected_suffix = prompt_ids[-len(response_ids):]
            assert expected_suffix == response_ids, (
                f"response_ids must be a suffix of prompt_ids. "
                f"Got prompt_ids={prompt_ids}, response_ids={response_ids}."
            )

        assert len(prompt_ids) <= self.max_model_length, (
            f"[StatelmToolAgentLoop][BUG] prompt_ids length {len(prompt_ids)} exceeds max_model_length {self.max_model_length}. This should not happen, check the safety mechanism."
        )
        
        trajectory_response_ids = prompt_ids[-len(response_mask):]
        trajectory_prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

        if (len(trajectory_response_ids) > self.response_length) or (len(trajectory_prompt_ids) > self.prompt_length):
            if enforce_output:
                return AgentLoopOutput(
                    prompt_ids=trajectory_prompt_ids[:self.prompt_length],
                    response_ids=trajectory_response_ids[:self.response_length],
                    response_mask=response_mask[:self.response_length],
                    response_logprobs=response_logprobs[:self.response_length] if response_logprobs else [],
                    num_turns=num_turns,
                    reward_score=reward_score,
                    metrics=metrics if metrics is not None else {},
                    extra_fields={
                        "is_snapshot": is_snapshot,
                    },
                )
            else:
                logger.warning(
                    f"[StatelmToolAgentLoop] trajectory_response_ids length {len(trajectory_response_ids)} > response_length {self.response_length} or trajectory_prompt_ids length {len(trajectory_prompt_ids)} > prompt_length {self.prompt_length}. Discarding the trajectory."
                )
                return None
            
        return AgentLoopOutput(
            prompt_ids=trajectory_prompt_ids,
            response_ids=trajectory_response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs if response_logprobs else [],
            num_turns=num_turns,
            reward_score=reward_score,
            metrics=metrics if metrics is not None else {},
            extra_fields={
                "is_snapshot": is_snapshot,
            },
        )

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""

        messages = render_context(agent_data.full_history, agent_data.notes, agent_data.deleted_msg_ids)

        agent_data.prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages,
                tools=self.tool_schemas,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )
        
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
            )

        agent_data.assistant_turns += 1
        # [WARNING] response_ids only stores the response tokens generated by the model at this turn
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        # Extract tool calls
        text_response, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        # Append assistant message to full_history; convert FunctionCall to proper OpenAI tool_call format
        tool_calls_formatted = []
        current_msg_id = agent_data.msg_id_counter
        for i, tc in enumerate(agent_data.tool_calls):
            tool_calls_formatted.append({
                "id": f"tool_call_{current_msg_id}",
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
            })
        agent_data.full_history.append({
            "role": "assistant",
            "content": [{"text": text_response}],  # Content should be a list with text block
            "tool_calls": tool_calls_formatted,
            "msg_id": current_msg_id,
        })
        agent_data.msg_id_counter += 1

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        else:
            logger.info(f"[StatelmToolAgentLoop] No tool calls, terminating the agent.")
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []

        tool_responses = []
        tool_calls_to_process = agent_data.tool_calls[: self.max_parallel_calls]
        with simple_timer("tool_calls", agent_data.metrics):
            for tool_call in tool_calls_to_process:
                response = self._call_tool(tool_call, agent_data.tools_kwargs, agent_data)
                tool_responses.append(response)


        finish_tool_call = False
        editor_tool_call = False
        delete_msg_ids = []  # Collect message IDs to delete
        tool_result_list = []
        assistant_msg_id = agent_data.msg_id_counter - 1
        for tool_response, tool_reward, tool_result_dict, tool_name in tool_responses:
            message = {"role": "tool", "content": tool_response.text or ""}

            # Handle special tool types
            if tool_name == 'finish':
                finish_tool_call = True
            elif tool_name == 'deleteContext':
                editor_tool_call = True
                agent_data.had_delete_operation = True
                # Extract message IDs to delete from tool result
                if tool_result_dict and "deleted_msg_ids" in tool_result_dict:
                    delete_msg_ids.extend(tool_result_dict["deleted_msg_ids"])

            add_messages.append(message)
            
            # Collect tool results
            current_msg_id = agent_data.msg_id_counter
            tool_result_list.append({
                "role": "tool",
                "content": message["content"],
                "tool_name": tool_name,
                "msg_id": current_msg_id,
                "msg_id(invoking_assistant)": assistant_msg_id
            })
            agent_data.msg_id_counter += 1

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        if finish_tool_call:
            return AgentState.TERMINATED
        
        # Handle deleteContext operation: snapshot, mask, and delete
        if editor_tool_call or (self.model_type == "qwen3"):
            # Step 1: Create snapshot of current trajectory state (run in executor to avoid blocking)
            def _create_snapshot():
                return {
                    "prompt_ids": copy.deepcopy(agent_data.prompt_ids), # up to the last assistant message, no tool response tokens
                    "response_ids": copy.deepcopy(agent_data.response_ids), # assistant response tokens generated by the model at this turn
                    "response_mask": copy.deepcopy(agent_data.response_mask),
                    "response_logprobs": copy.deepcopy(agent_data.response_logprobs),
                    "num_turns": agent_data.assistant_turns,
                }
            snapshot = await self.loop.run_in_executor(None, _create_snapshot)
            agent_data.trajectory_snapshots.append(snapshot)
            
            # Step 2: Clear response_mask, response_ids and response_logprobs
            agent_data.response_mask = []
            agent_data.response_ids = []
            agent_data.response_logprobs = []
            
            # Step 3: Mark messages as deleted
            for msg_id in delete_msg_ids:
                agent_data.deleted_msg_ids.add(msg_id)
            
            if editor_tool_call: # full history unchanged yet
                rebuilt_message = render_context(agent_data.full_history, agent_data.notes, agent_data.deleted_msg_ids)
                agent_data.prompt_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        rebuilt_message,
                        tools=self.tool_schemas,
                        add_generation_prompt=False,
                        tokenize=True,
                        **self.apply_chat_template_kwargs,
                    ),
                )
    
        # Update prompt with tool responses
        tool_result_token_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                add_messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
            ),
        )
        # Debugging: print the tool result tokens
        tool_result_tokens = self.tokenizer.decode(tool_result_token_ids, skip_special_tokens=False)
        # logger.info(f"[PROCESSING_TOOLS] Tool result tokens: {tool_result_tokens[:100]}...")

        # ===================================== CONTEXT CHECK =========================================
        # (1) If the total length of prompt_ids and tool_result_token_ids exceeds max_model_length, 
        # terminate without appending the tool result
        # 
        # (2) If assistant turns exceeds max_assistant_turns, terminate without appending the tool result
        # Notice that this implementation ensures that the prompt_ids and response_mask are set up to the 
        # last assistant message, so we do not need to strip the tool result tokens in TERMINATED state.
        # =============================================================================================
        
        # Also need to budget the next round of generation
        if len(agent_data.prompt_ids) + len(tool_result_token_ids) + self.max_response_length > self.max_model_length:
            logger.warning(
                f"[PROCESSING_TOOLS] prompt_ids (len={len(agent_data.prompt_ids)}) + tool_result_token_ids (len={len(tool_result_token_ids)}) + max_response_length ({self.max_response_length}) exceeds max_model_length ({self.max_model_length}). Terminating without appending the tool result."
            )
            self.context_len_or_turn_exceeded = True
            return AgentState.TERMINATED # last turn, no need to strip the thinking tag, never append the tool result
        
        # Not a finishing tool call, so equals to the number of assistant turns also means exceeding
        if agent_data.assistant_turns >= self.max_assistant_turns:
            logger.warning(
                f"[PROCESSING_TOOLS] assistant_turns (={agent_data.assistant_turns}) exceeds(>=) max_assistant_turns (={self.max_assistant_turns}), terminating without appending the tool result."
            )
            self.context_len_or_turn_exceeded = True
            return AgentState.TERMINATED # last turn, no need to strip the thinking tag, never append the tool result

        # Append the tool results to the full history
        for tool_result in tool_result_list:
            agent_data.full_history.append(tool_result)
        
        # Handling the thinking tag problem for Qwen3
        if self.model_type == "qwen3":
            agent_data.prompt_ids = strip_the_last_think_tags(agent_data.prompt_ids)

        agent_data.prompt_ids += tool_result_token_ids

        # For non-state-editing tool call, we can just append the tool response to prompt directly
        if not(editor_tool_call or (self.model_type == "qwen3")):
            agent_data.response_mask += [0] * len(tool_result_token_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(tool_result_token_ids)
        return AgentState.GENERATING


    def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict, str]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        tool_name = ""
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            
            # Handle regular tools (deleteContext is now a proper tool class)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            
            # Prepare execution kwargs for StateLM tools
            exec_kwargs = {}
            statelm_tool_names = {
                'analyzeText', 'loadDocument', 'buildIndex', 'readChunk',
                'searchEngine', 'note', 'readNote', 'updateNote', 'mergeNotes',
                'checkBudget', 'getContextStats', 'deleteContext', 'finish'
            }
            if tool_name in statelm_tool_names:
                exec_kwargs['agent_data'] = agent_data
                exec_kwargs['doc_state_manager'] = agent_data.doc_state_manager
                exec_kwargs['tokenizer'] = self.tokenizer
                exec_kwargs['tool_schemas'] = self.tool_schemas
                exec_kwargs['max_context_exp'] = getattr(self, 'max_model_length', 32000)
                exec_kwargs['max_output_tokens'] = self.response_length
                exec_kwargs['max_turns'] = self.max_assistant_turns
            tool_execution_response, tool_reward, res = tool.execute(instance_id, tool_args, **exec_kwargs)
        
        except json.JSONDecodeError as e:
            return (
                ToolResponse(
                    text=f"Error: Invalid JSON in tool arguments: {e}",
                ),
                0.0,
                {},
                tool_name,
            )
        except Exception as e:
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
                tool_name,
            )
        finally:
            if tool and instance_id:
                tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        return ToolResponse(text=tool_response_text), tool_reward, res if res else {}, tool_name