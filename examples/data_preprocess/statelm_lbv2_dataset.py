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
"""
Preprocess a long-context multiple-choice QA dataset for the StateLM tool agent loop.

Each input example is expected to have the structure:

{
    "_id": "66fa69a4bb02136c067c6b75",
    "domain": "Code Repository Understanding",
    "sub_domain": "Code repo QA",
    "difficulty": "hard",
    "length": "long",
    "question": "What contributions does this code base provide?",
    "choice_A": "...",
    "choice_B": "...",
    "choice_C": "...",
    "choice_D": "...",
    "answer": "D",
    "context": "Full reference document content..."
}

The `context` field contains the long document that will be exposed to the agent via StateLM tools.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import datasets

from verl.utils.hdfs_io import copy, makedirs


SYSTEM_PROMPT = """You are an AI assistant for long-context processing with tools. Produce factually correct answers grounded in any attached text while conserving the context window by deleting unnecessary messages and taking notes. Describe your processing plan first, then proceed with the tools."""


def _extract_choices(example: Dict[str, Any]) -> Dict[str, str]:
    choices: Dict[str, str] = {}
    for key in ["A", "B", "C", "D"]:
        value = example.get(f"choice_{key}")
        if value is not None:
            choices[key] = value.strip()
    return choices


def build_mcq_prompt_and_answer(example: Dict[str, Any]) -> str:
    question = example['question'].strip()
    choices = [example[f'choice_{opt}'].strip() for opt in ['A', 'B', 'C', 'D']]
    question_str = f"{question}\n"
    for i, choice in enumerate(choices):
        question_str += f"{chr(65 + i)}. {choice}\n"
    question_str += "Select the best option from the choices above."
    return question_str, example["answer"].strip()

def build_sa_prompt_and_answer(example: Dict[str, Any]) -> str:
    question_str = example['question'].strip()
    text_answer = example[f'choice_{example["answer"]}']
    return question_str, text_answer.strip()


def preprocess_split(dataset_split, split_name: str, data_source: str, agent_name: str):
    mc_to_sa_count = 0
    mc_count = 0
    def map_fn(example: dict, idx: int):
        nonlocal mc_to_sa_count, mc_count
        answer_letter = example.get("answer", "").strip()
        choices = _extract_choices(example)
        document_content = example.get("context", "")

        if example.get("mc_to_sa") == 'true':
            user_prompt, answer_text = build_sa_prompt_and_answer(example)
            mc_to_sa_count += 1
        else:
            user_prompt, answer_text = build_mcq_prompt_and_answer(example)
            mc_count += 1
        record = {
            "data_source": data_source,
            "agent_name": agent_name,
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "ability": "long_context_reasoning",
            "reward_model": {"style": "rule", "ground_truth": answer_text},
            "document_content": document_content,
            "extra_info": {
                "split": split_name,
                "index": idx,
                "example_id": example.get("_id"),
                "domain": example.get("domain"),
                "sub_domain": example.get("sub_domain"),
                "difficulty": example.get("difficulty"),
                "length": example.get("length"),
                "choices": choices,
                "answer_text": answer_letter,
                "need_tools_kwargs": False,
            },
        }
        return record

    remove_cols = [c for c in dataset_split.column_names if c not in {"prompt", "document_content"}]
    mapped_dataset = dataset_split.map(map_fn, with_indices=True, remove_columns=remove_cols)
    print(f"[{split_name}] Converted to short answer questions: {mc_to_sa_count}")
    print(f"[{split_name}] MC questions: {mc_count}")
    return mapped_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare StateLM Book QA dataset.")
    parser.add_argument("--dataset_name", default="your_org/book_qa", help="HF dataset identifier.")
    parser.add_argument("--dataset_config", default=None, help="Optional dataset config name.")
    parser.add_argument("--local_dataset_path", default=None, help="Optional local dataset path.")
    parser.add_argument("--splits", default="train,validation", help="Comma-separated splits to export.")
    parser.add_argument("--local_save_dir", default="~/data/statelm_book_qa", help="Output directory for parquet files.")
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS destination.")
    parser.add_argument("--data_source_tag", default="long_context_mcq", help="Value stored in data_source field.")
    parser.add_argument("--agent_name", default="statelm_tool_agent", help="Name of the agent.")
    args = parser.parse_args()

    dataset_name = args.local_dataset_path or args.dataset_name
    load_kwargs = {}
    if args.dataset_config:
        load_kwargs["name"] = args.dataset_config

    dataset_dict = datasets.load_dataset(dataset_name, **load_kwargs)
    split_names: List[str] = [s.strip() for s in args.splits.split(",") if s.strip()]

    target_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(target_dir, exist_ok=True)

    for split in split_names:
        if split not in dataset_dict:
            raise ValueError(f"Split '{split}' not available in dataset.")
        processed = preprocess_split(dataset_dict[split], split, args.data_source_tag, args.agent_name)
        save_path = os.path.join(target_dir, f"{split}.parquet")
        processed.to_parquet(save_path)
        print(f"Saved {split} with {len(processed)} examples to {save_path}")

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=target_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()