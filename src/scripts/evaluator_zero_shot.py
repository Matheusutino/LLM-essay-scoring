import ast
import argparse
import pandas as pd
from tqdm import tqdm
from src.core.llm_predictor.llm_factory import LLMFactory
from src.core.utils import check_directory_exists, create_directory, save_json, get_prompts
from configs.config import competence_cols, all_keys_evaluator
from src.core.llm_classification import llm_generator


def evaluator_zero_shot(
    essay_path: str,
    prompts_path: str,
    model_name: str,
    prompt_name: str,
    llm_provider: str,
    temperature: float,
    max_output_tokens: int,
    output_path: str,
):
    """Evaluates essays using a zero-shot LLM-based approach.

    This function reads an essay dataset and a prompt template,
    formats the input for the LLM, obtains predictions, and saves the results.

    Args:
        essay_path (str): Path to the input CSV file containing essays and scores.
        prompts_path (str): Path to the CSV file containing prompt descriptions.
        model_name (str): Name or ID of the LLM model to use.
        prompt_name (str): Name of the prompt template to load.
        llm_provider (str): Provider of the LLM (e.g., "openai", "nvidia").
        temperature (float): Temperature setting for the LLM.
        max_output_tokens (int): Maximum number of tokens the LLM can generate.
        output_path (str): Base path where results will be saved.
    """
    model_id = model_name.split("/")[-1] if "/" in model_name else model_name
    path_to_save = f"{output_path}/zero_shot/{model_id}/{prompt_name}"
    check_directory_exists(path_to_save)

    # Load and preprocess the essay data
    df_essay = pd.read_csv(essay_path)
    df_essay["competence"] = df_essay["competence"].apply(ast.literal_eval)
    df_essay["essay"] = df_essay["essay"].apply(ast.literal_eval)
    df_essay["essay"] = df_essay["essay"].apply(lambda x: "\n\n".join(x))
    df_essay[competence_cols] = pd.DataFrame(df_essay["competence"].tolist(), index=df_essay.index)

    # Load prompts and LLM
    df_prompts = pd.read_csv(prompts_path)
    user_prompt, system_prompt = get_prompts(prompt_name)
    llm = LLMFactory.get_predictor(llm_provider, model_name)

    llm_results = {}

    # Iterate through each essay
    for idx, row in tqdm(df_essay.iterrows(), total=len(df_essay), desc="Classifying with LLM"):
        line = df_prompts[df_prompts['id'] == row['prompt']]
        motivational_texts = line['description'].values[0]

        title = row['title']
        essay = row['essay']

        # Format the user prompt
        user_prompt_formatted = user_prompt.format(
            motivational_texts=motivational_texts,
            title=title,
            essay=essay
        )

        # Get LLM output
        result_data = llm_generator(
            prompt_name,
            user_prompt_formatted,
            system_prompt,
            llm,
            temperature,
            max_output_tokens,
            all_keys_evaluator,
            max_retries=50
        )

        # Store each expected key from the result
        for key in all_keys_evaluator:
            if key not in llm_results:
                llm_results[key] = []
            llm_results[key].append(result_data.get(key))

    # Merge LLM results into the DataFrame
    for key, value in llm_results.items():
        df_essay[key] = value

    # Save the evaluation results
    create_directory(path_to_save)
    df_essay.to_csv(f"{path_to_save}/essay_results.csv", index=False)

    # Save the configuration used
    config_to_save = {
        "essay_path": essay_path,
        "prompts_path": prompts_path,
        "model_name": model_name,
        "prompt_name": prompt_name,
        "llm_provider": llm_provider,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens
    }
    save_json(config_to_save, f"{path_to_save}/config.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run zero-shot evaluator on essay dataset using LLM.")
    parser.add_argument("--essay_path", type=str, default="dataset/testing.csv", help="Path to the essay CSV.")
    parser.add_argument("--prompts_path", type=str, default="dataset/prompts.csv", help="Path to the prompts CSV.")
    parser.add_argument("--model_name", type=str, default="mistralai/mistral-medium-3-instruct", help="Model name or path.")
    parser.add_argument("--prompt_name", type=str, default="evaluator_zero_shot", help="Name of the prompt template.")
    parser.add_argument("--llm_provider", type=str, default="nvidia", help="LLM provider name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature.")
    parser.add_argument("--max_output_tokens", type=int, default=8000, help="Max number of output tokens.")
    parser.add_argument("--output_path", type=str, default="results/essay", help="Base path to save results.")

    args = parser.parse_args()

    evaluator_zero_shot(
        essay_path=args.essay_path,
        prompts_path=args.prompts_path,
        model_name=args.model_name,
        prompt_name=args.prompt_name,
        llm_provider=args.llm_provider,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        output_path=args.output_path
    )
