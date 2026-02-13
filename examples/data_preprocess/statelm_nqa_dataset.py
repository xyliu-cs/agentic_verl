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
  'book_id': 'B00',
  'question_id': 'Q0762',
  'question': "Eustace Woodville is used to be negative and finally becomes a postive one? Tell in one sentence that which episode marks this charact-er's change.",
  'answer': "Valeria Brinton marries Eustace Woodville despite objections from Woodville's family; this decision worries Valeria's family and friends.",
  'question_type': 'open_end',
  'context': ...
  'meta_info': {'Answer': "Valeria Brinton marries Eustace Woodville despite objections from Woodville's family; this decision worries Valeria's family and friends.",
  'Aspect': 'character',
  'Complexity': 'mh',
  'Gold': 'D',
  'Options': {'A': 'Eustace Woodville reveals his true identity as Eustace Macallan, which causes a scandal and estrangement from Valeria',
   'B': "Valeria Brinton discovers Eustace Woodville's secret past, leading to a confrontation and his eventual redemption",
   'C': "Eustace Woodville's gambling debts are exposed, forcing him to reconcile with his family and Valeria's support.",
   'D': "Valeria Brinton marries Eustace Woodville despite objections from Woodville's family; this decision worries Valeria's family and friends."},
  'Question': "Eustace Woodville is used to be negative and finally becomes a postive one? Tell in one sentence that which episode marks this charact-er's change.",
  'book_author': 'Wilkie Collins',
  'book_name': 'The Law and the Lady',
  'book_tokenlen': 181760}
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



def preprocess_split(dataset_split, split_name: str, data_source: str, agent_name: str):
    def map_fn(example: dict, idx: int):
        record = {
            "data_source": data_source,
            "agent_name": agent_name,
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example.get("question")},
            ],
            "ability": "long_context_reading",
            "reward_model": {"style": "rule", "ground_truth": example.get("answer")},
            "document_content": example.get("context"),
            "extra_info": {
                "split": split_name,
                "book_id": example.get("book_id"),
                "question_id": example.get("question_id"),
                "question_type": example.get("question_type"),
                "need_tools_kwargs": False,
            },
        }
        return record

    return dataset_split.map(map_fn, with_indices=True, remove_columns=dataset_split.column_names)


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