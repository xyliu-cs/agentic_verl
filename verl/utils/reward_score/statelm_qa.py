"""
Reward function for QA problems for StateLM Training.

- answer is correct with good format: +1
- answer is of good format but incorrect: -0.5
- answer is of bad format: -1

definition of good format:
- tool call is used to finish the question
- for MCQ, the answer is just the option letter or start with the option letter
- for Open-Ended, the answer is judged correct by the LLM-Judge, and is within 1 sentence.
"""

import re
import json
import os
import random
from openai import OpenAI
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

LLM_JUDGE_PROMPT_TEMPLATE = """
Given a problem, its correct answer, and a student's answer below, your task is to review the student's answer and determine if it is correct by comparing it to the correct answer. If the student's answer is incomplete or ambiguous, assume it is incorrect.

### Problem
{problem}

### Answer
{answer}

### Student Answer
{model_ans}

Please put your final answer (True or False) in \\boxed{{}}. Specifically, if the student's answer is correct, the final answer should be \\boxed{{True}}; otherwise, the final answer should be \\boxed{{False}}.
""".strip()


def has_valid_tool_call(solution_str: str) -> bool:
    """Check if the solution string contains a proper tool call."""
    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    match = re.search(tool_call_pattern, solution_str, re.DOTALL)
    if match:
        try:
            tool_call = json.loads(match.group(1))
            return tool_call.get("name") == "finish" and "answer" in tool_call.get("arguments", {})
        except (json.JSONDecodeError, AttributeError):
            return False
    return False


def extract_tool_call_answer(solution_str: str) -> tuple[str, bool]:
    """Extract the answer from the tool call in the model output.
    A decoded solution string may look like this:

    assistant
    I find out who married Rosamund. Let me finish this question.
    <tool_call>
    {"name": "finish", "arguments": {"answer": "A"}}
    </tool_call>

    Returns:
        tuple: (extracted_answer, has_valid_tool_call)
    """
    # Remove everything before the last "assistant"
    last_assistant_idx = solution_str.rfind("assistant")
    if last_assistant_idx != -1:
        solution_str = solution_str[last_assistant_idx + len("assistant"):]
    
    # Try to parse from the last tool call (JSON format)
    if has_valid_tool_call(solution_str):
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        match = re.search(tool_call_pattern, solution_str, re.DOTALL)
        if match:
            try:
                tool_call = json.loads(match.group(1))
                final_answer = tool_call.get("arguments").get("answer")
                return final_answer, True
            except (json.JSONDecodeError, AttributeError):
                return "", False
    return "", False


def is_mcq(answer_str: str) -> bool:
    """Determine if the answer of MCQ format (i.e., ABCD)."""
    ans = answer_str.strip()
    # If ground truth is a single letter A-G, it's MCQ problem
    return len(ans) == 1 and ans in "ABCD"


def count_sentences(text: str) -> int:
    """Count the number of sentences in text."""
    # Simple sentence counter: count periods, exclamation marks, question marks
    text = text.strip()
    if not text:
        return 0
    # Split by sentence delimiters
    sentences = re.split(r'[.!?]+|\n+', text)
    # Filter out empty strings
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)


def llm_judge_answer(prediction: str, ground_truth: str, question: str = None) -> bool:
    """Use OpenAI API to judge if the prediction is correct.
    
    Args:
        prediction: The predicted answer
        ground_truth: The correct answer
        question: Optional question text for context
        
    Returns:
        True if LLM judges the answer as correct, False otherwise
    """
    import time
    from openai import RateLimitError, APIConnectionError

    MAX_RETRIES = 3
    BASE_DELAY = 2
    
    # Initialize client once outside the loop
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"), 
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    model_name = os.environ.get("OPENAI_MODEL")
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator."},
                    {"role": "user", "content": LLM_JUDGE_PROMPT_TEMPLATE.format(
                        problem=question, 
                        answer=ground_truth, 
                        model_ans=prediction
                    )}
                ],
                temperature=0.0,
                max_tokens=1024
            )
            result = last_boxed_only_string(response.choices[0].message.content)
            if result is not None:
                correctness = remove_boxed(result).strip().lower() == "true"
                return correctness
            
            # No boxed result found - retry if attempts remain
            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2 ** attempt)
                print(f"LLM Judge: No boxed result found, retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise ValueError(f"No boxed result found after {MAX_RETRIES} attempts: {response.choices[0].message.content}")
                
        except (RateLimitError, APIConnectionError) as e:
            # Transient errors - retry with exponential backoff
            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2 ** attempt)
                print(f"LLM Judge API error: {e}, retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"LLM Judge failed after {MAX_RETRIES} attempts: {e}")
                # Fallback to simple string matching
                pred_normalized = prediction.lower().strip()
                gt_normalized = ground_truth.lower().strip()
                return pred_normalized == gt_normalized
                
        except Exception as e:
            # Other errors (auth, parsing, etc.) - fail to fallback immediately
            print(f"LLM Judge error: {e}")
            pred_normalized = prediction.lower().strip()
            gt_normalized = ground_truth.lower().strip()
            return pred_normalized == gt_normalized


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> float:
    """Compute the score for QA problems following the StateLM grading scheme.
    
    Scoring:
    - answer is correct with good format: +1
    - answer is of good format but incorrect: -0.5
    - answer is of bad format: -1
    
    Good format definition:
    - tool call is used to finish the question
    - for MCQ, the answer is just the option letter or start with the option letter
    - for Open-Ended, the answer is judged correct by LLM-Judge and is within 1 sentence
    
    Args:
        solution_str: The model's output string
        ground_truth: The correct answer (letter for MCQ, or text for Open-Ended)
        **kwargs: Additional keyword arguments:
            - question: Optional question text for LLM judge context
    
    Returns:
        float: +1.0, -0.5, or -1.0
    """
    MAX_SENTENCES_FOR_GOOD_FORMAT = 3
    # Extract answer and check if tool call was used
    predicted, has_valid_tool_call = extract_tool_call_answer(solution_str)
    
    # Ensure predicted is a string (handle cases where it might be a dict or other type)
    if not isinstance(predicted, str):
        predicted = str(predicted) if predicted else ""
    
    messages = kwargs["extra_info"]["raw_prompt"]
    for msg in messages:
        if msg.get("role") == "user":
            question = msg.get("content")
            break
    else:
        question = ""
        print(f"[Statelm QA Reward] No user prompt found, using empty question.")
    
    # Determine question type
    is_mcq_question = is_mcq(ground_truth)
    
    if is_mcq_question:
        # MCQ grading
        has_good_format = has_valid_tool_call
        is_correct = (predicted.strip() == ground_truth) or (predicted.strip().startswith(f"{ground_truth}."))
        
        if is_correct and has_good_format:
            score = 1.0  # Correct with good format
        elif has_good_format and not is_correct:
            score = -0.5  # Good format but incorrect
        else:
            score = -1.0  # Bad format
    else:
        # Open-Ended grading
        num_sentences = count_sentences(predicted)
        if num_sentences > MAX_SENTENCES_FOR_GOOD_FORMAT:
            print(f"[Statelm QA Reward] Open-Ended overlong answer: {predicted}, score: -1.0")
            return -1.0

        is_correct = llm_judge_answer(predicted, ground_truth, question)
        has_good_format = has_valid_tool_call and num_sentences <= MAX_SENTENCES_FOR_GOOD_FORMAT
        
        if is_correct and has_good_format:
            score = 1.0  # Correct with good format
        elif has_good_format and not is_correct:
            score = -0.5  # Good format but incorrect
        else:
            score = -1.0  # Bad format

    # Give 1/5 chance of printing the question, answer, model answer, and score
    if random.random() < 0.2:
        print(f"[Statelm QA Reward] Question: {question}")
        print(f"[Statelm QA Reward] Correct Answer: {ground_truth}")
        # print(f"[Statelm QA Reward] Model Solution: {solution_str}")
        print(f"[Statelm QA Reward] Extracted Answer: {predicted}")
        print(f"[Statelm QA Reward] Score: {score}")
        
    return score