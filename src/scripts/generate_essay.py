import ast
import argparse
import pandas as pd
from tqdm import tqdm
from src.core.llm_predictor.llm_factory import LLMFactory
from src.core.utils import check_directory_exists, create_directory, get_prompts, save_json
from configs.config import all_keys_writer
from src.core.llm_classification import llm_generator


def generate_essay(
    prompts_path: str,
    model_name: str,
    prompt_name: str,
    llm_provider: str,
    temperature: float,
    max_output_tokens: int,
    output_path: str,
):
    """
    Generates essays from motivational texts using a Large Language Model (LLM).

    Args:
        prompts_path (str): Path to the CSV file containing motivational prompts.
        model_name (str): Name or path of the LLM model to use.
        prompt_name (str): Name of the prompt template to use for generation.
        llm_provider (str): Name of the provider (e.g., 'openai', 'nvidia').
        temperature (float): Temperature setting for the LLM (controls randomness).
        max_output_tokens (int): Maximum number of tokens to generate.
        output_path (str): Directory path where the output will be saved.

    Outputs:
        - A CSV file with generated essays.
        - A JSON config file logging the run parameters.
    """
    # Extract model ID from model path (e.g., "org/model" -> "model")
    model_id = model_name.split("/")[-1] if "/" in model_name else model_name
    path_to_save = f"{output_path}/{model_id}/{prompt_name}"

    # Ensure the directory exists
    check_directory_exists(path_to_save)

    # Load the dataset containing motivational texts
    df_prompts = pd.read_csv(prompts_path)

    # Load the user and system prompt templates
    user_prompt, system_prompt = get_prompts(prompt_name)

    # Instantiate the LLM predictor
    llm = LLMFactory.get_predictor(llm_provider, model_name)

    # Initialize a dictionary to store the results
    llm_results = {}

    # Loop through each row in the dataset
    for idx, row in tqdm(df_prompts.iterrows(), total=len(df_prompts), desc="Classifying with LLM"):
        try:
            # Try to parse the string as a list (if it's a list of sentences)
            val = ast.literal_eval(row['description'])
            motivational_texts = '\n\n'.join(val) if isinstance(val, list) else val
        except Exception as e:
            # Fallback if parsing fails
            motivational_texts = row['description']

        # Format the user prompt using the current motivational text(s)
        user_prompt_formatted = user_prompt.format(
            motivational_texts=motivational_texts,
        )

        # Generate essay using the LLM
        generated_essay = llm_generator(
            prompt_name=prompt_name,
            user_prompt=user_prompt_formatted,
            system_prompt=system_prompt,
            llm=llm,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            expected_keys=all_keys_writer,
            max_retries=50
        )

        # Save each key (e.g., "essay", "summary", etc.) from the generated result
        for key in all_keys_writer:
            if key not in llm_results:
                llm_results[key] = []
            llm_results[key].append(generated_essay.get(key))

    # Add the generated columns to the original DataFrame
    for key, value in llm_results.items():
        df_prompts[key] = value

    # Create directory to save the results (if it doesn't exist)
    create_directory(path_to_save)

    # Save the DataFrame with the generated essays
    df_prompts.to_csv(f"{path_to_save}/essay_generated.csv", index=False)

    # Save the configuration used for this run
    config_to_save = {
        "prompts_path": prompts_path,
        "model_name": model_name,
        "prompt_name": prompt_name,
        "llm_provider": llm_provider,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens
    }
    save_json(config_to_save, f"{path_to_save}/config.json")


if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Run few-shot evaluator on essay dataset using LLM.")
    parser.add_argument("--prompts_path", type=str, default="dataset/prompts.csv", help="Path to the prompts CSV.")
    parser.add_argument("--model_name", type=str, default="mistralai/mistral-medium-3-instruct", help="Model name or path.")
    parser.add_argument("--prompt_name", type=str, default="writer_zero_shot", help="Name of the prompt template.")
    parser.add_argument("--llm_provider", type=str, default="nvidia", help="LLM provider name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature.")
    parser.add_argument("--max_output_tokens", type=int, default=8000, help="Max number of output tokens.")
    parser.add_argument("--output_path", type=str, default="results/generated", help="Base path to save results.")

    # Parse arguments from CLI
    args = parser.parse_args()

    # Run the essay generation process
    generate_essay(
        prompts_path=args.prompts_path,
        model_name=args.model_name,
        prompt_name=args.prompt_name,
        llm_provider=args.llm_provider,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        output_path=args.output_path
    )
