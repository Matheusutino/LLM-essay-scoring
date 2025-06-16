import ast
import argparse
import pandas as pd
from tqdm import tqdm
from typing import Any

from src.core.llm_predictor.llm_factory import LLMFactory
from src.core.utils import check_directory_exists, create_directory, save_json, get_prompts
from configs.config import competence_cols, all_keys_evaluator
from src.core.llm_classification import llm_generator


def evaluator_few_shot(
    all_essay_path: str,
    essay_path: str,
    prompts_path: str,
    model_name: str,
    prompt_name: str,
    llm_provider: str,
    temperature: float,
    max_output_tokens: int,
    output_path: str,
) -> None:
    """
    Evaluates essays using a few-shot prompt strategy with a Language Model.

    This function loads a dataset of essays and evaluates them by comparing each essay
    with a top-scoring example from the same prompt. It then uses a language model to 
    generate evaluation scores for each competence.

    Args:
        all_essay_path (str): Path to the complete essay dataset (for selecting examples).
        essay_path (str): Path to the subset of essays to be evaluated.
        prompts_path (str): Path to the CSV containing the prompt descriptions.
        model_name (str): Name or ID of the model to use.
        prompt_name (str): Name of the prompt template to use.
        llm_provider (str): Name of the provider used for the LLM.
        temperature (float): Temperature parameter for the LLM generation.
        max_output_tokens (int): Maximum number of tokens the LLM can generate.
        output_path (str): Directory where results will be saved.
    """
    # Prepare output directory
    model_id = model_name.split("/")[-1] if "/" in model_name else model_name
    path_to_save = f"{output_path}/few_shot/{model_id}/{prompt_name}"
    check_directory_exists(path_to_save)

    # Load and prepare all essay data (used to select few-shot examples)
    df_all_essay = pd.read_csv(all_essay_path)
    df_all_essay["competence"] = df_all_essay["competence"].apply(ast.literal_eval)
    df_all_essay["essay"] = df_all_essay["essay"].apply(ast.literal_eval)
    df_all_essay["essay"] = df_all_essay["essay"].apply(lambda x: "\n\n".join(x))
    df_all_essay[competence_cols] = pd.DataFrame(df_all_essay["competence"].tolist(), index=df_all_essay.index)

    # Load and prepare target essay data (to be evaluated)
    df_essay = pd.read_csv(essay_path)
    df_essay["competence"] = df_essay["competence"].apply(ast.literal_eval)
    df_essay["essay"] = df_essay["essay"].apply(ast.literal_eval)
    df_essay["essay"] = df_essay["essay"].apply(lambda x: "\n\n".join(x))
    df_essay[competence_cols] = pd.DataFrame(df_essay["competence"].tolist(), index=df_essay.index)

    # Load prompts
    df_prompts = pd.read_csv(prompts_path)
    user_prompt, system_prompt = get_prompts(prompt_name)

    # Load LLM
    llm = LLMFactory.get_predictor(llm_provider, model_name)

    llm_results = {}

    # Iterate through each essay and evaluate
    for idx, row in tqdm(df_essay.iterrows(), total=len(df_essay), desc="Classifying with LLM"):
        line = df_prompts[df_prompts['id'] == row['prompt']]
        motivational_texts = line['description'].values[0]
        title = row['title']
        essay = row['essay']

        # Find high-scoring essays with the same prompt
        same_prompt_essays = df_all_essay[df_all_essay['prompt'] == row['prompt']].copy()
        same_prompt_essays_sorted = same_prompt_essays.sort_values(by="score", ascending=False).reset_index(drop=True)

        # Remove the current essay from the list (by content)
        other_essays = same_prompt_essays_sorted[same_prompt_essays_sorted['essay'] != row['essay']]

        # Select the top high-score essay as few-shot example
        top_other_essay = other_essays.iloc[0]

        # Format the user prompt with both current and example essays
        user_prompt_formatted = user_prompt.format(
            motivational_texts=motivational_texts,
            title=title,
            essay=essay,
            total_score=row['score'],
            score_I=top_other_essay['nota_competencia_I_real'],
            score_II=top_other_essay['nota_competencia_II_real'],
            score_III=top_other_essay['nota_competencia_III_real'],
            score_IV=top_other_essay['nota_competencia_IV_real'],
            score_V=top_other_essay['nota_competencia_V_real'],
            title_example=top_other_essay['title'],
            essay_example=top_other_essay['essay']
        )

        # Call the LLM to get predicted scores
        result_data = llm_generator(
            prompt_name=prompt_name,
            user_prompt=user_prompt_formatted,
            system_prompt=system_prompt,
            llm=llm,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            keys=all_keys_evaluator,
            max_retries=50
        )

        # Append result data to output dictionary
        for key in all_keys_evaluator:
            if key not in llm_results:
                llm_results[key] = []
            llm_results[key].append(result_data.get(key))

    # Save results into dataframe
    for key, value in llm_results.items():
        df_essay[key] = value

    # Save results to CSV
    create_directory(path_to_save)
    df_essay.to_csv(f"{path_to_save}/essay_results.csv", index=False)

    # Save config file
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
    parser = argparse.ArgumentParser(description="Run few-shot evaluator on essay dataset using LLM.")
    parser.add_argument("--all_essay_path", type=str, default="dataset/essay-br.csv", help="Path to the essay CSV.")
    parser.add_argument("--essay_path", type=str, default="dataset/testing.csv", help="Path to the essay CSV.")
    parser.add_argument("--prompts_path", type=str, default="dataset/prompts.csv", help="Path to the prompts CSV.")
    parser.add_argument("--model_name", type=str, default="mistralai/mistral-medium-3-instruct", help="Model name or path.")
    parser.add_argument("--prompt_name", type=str, default="evaluator_few_shot", help="Name of the prompt template.")
    parser.add_argument("--llm_provider", type=str, default="nvidia", help="LLM provider name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature.")
    parser.add_argument("--max_output_tokens", type=int, default=8000, help="Max number of output tokens.")
    parser.add_argument("--output_path", type=str, default="results/essay", help="Base path to save results.")

    args = parser.parse_args()

    evaluator_few_shot(
        all_essay_path=args.all_essay_path,
        essay_path=args.essay_path,
        prompts_path=args.prompts_path,
        model_name=args.model_name,
        prompt_name=args.prompt_name,
        llm_provider=args.llm_provider,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        output_path=args.output_path
    )
