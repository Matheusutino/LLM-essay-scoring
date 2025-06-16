import json
import time
import random
from typing import Dict, Any, List, Optional
from src.core.utils import extract_title_and_text, extract_evaluation_info
from configs.config import predicted_cols, valid_labels

def llm_generator(
    prompt_name: str,
    user_prompt: str,
    system_prompt: str,
    llm: Any,
    temperature: float,
    max_output_tokens: int,
    keys: List[str],
    max_retries: int = 10
) -> Dict[str, Any]:
    """
    Generates content using a Large Language Model (LLM) and processes its output based on the prompt type.

    Args:
        prompt_name (str): Type of prompt being used (e.g., "evaluator_zero_shot", "evaluator_few_shot").
        user_prompt (str): Prompt provided by the user.
        system_prompt (str): System-level instruction for the LLM.
        llm (Any): The LLM instance that has a `predict` method.
        temperature (float): Sampling temperature for the LLM.
        max_output_tokens (int): Maximum number of output tokens allowed in the response.
        keys (List[str]): Expected keys in the result dictionary.
        max_retries (int, optional): Maximum number of retry attempts if the output is invalid. Defaults to 10.

    Returns:
        Dict[str, Any]: A dictionary with parsed response data and original response. Returns empty dict if all retries fail.
    """
    current_temperature = temperature
    result = None

    for attempt in range(max_retries):
        try:
            # Send request to LLM
            response = llm.predict(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=current_temperature,
                max_output_tokens=max_output_tokens
            )

            # Case: evaluation prompts
            if prompt_name in {"evaluator_zero_shot", "evaluator_few_shot"}:
                result = extract_evaluation_info(response)

                # Strip whitespace and prepare result
                result_data = {
                    key: value.strip() if isinstance(value, str) else value
                    for key, value in result.items()
                }
                result_data['original_response'] = response

                # Validate predicted labels and keys
                if all(result.get(key) in valid_labels for key in predicted_cols) and set(result.keys()) == set(keys):
                    return result_data
                else:
                    invalid_fields = [
                        key for key in predicted_cols
                        if key not in result or result[key] not in valid_labels or set(result.keys()) != set(keys)
                    ]
                    print(f"[Retry {attempt + 1}] Invalid label(s) in: {', '.join(invalid_fields)}")

            # Case: content generation prompts (e.g., title and text)
            else:
                result = extract_title_and_text(response)
                result['original_response'] = response

                # Validate result
                if not result:
                    print(f"[Retry {attempt + 1}] No valid title and text found in response.")
                    continue
                
                return result

        except Exception as e:
            print(f"[Retry {attempt + 1}] Error: {e}")
            print(result)
            # On error, randomize temperature for next retry
            current_temperature = random.uniform(0, 2)
            print(f"Setting temperature to random value: {current_temperature:.2f} for next attempt.")
        
        # Sleep to avoid overwhelming the model or rate limits
        time.sleep(1)

    # If all attempts fail, return empty result
    return {}
