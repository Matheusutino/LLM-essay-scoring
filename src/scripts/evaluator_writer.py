import ast
import argparse
import pandas as pd
from tqdm import tqdm
from src.core.llm_predictor.llm_factory import LLMFactory
from src.core.utils import check_directory_exists, create_directory, save_json, get_prompts
from configs.config import competence_cols, all_keys_evaluator
from src.core.llm_classification import llm_generator
from src.scripts.generate_essay import generate_essay


def evaluator_writer(
    essay_path: str,
    prompts_path: str,
    model_name: str,
    prompt_name_evaluator: str,
    prompt_name_writer: str,
    llm_provider: str,
    temperature: float,
    max_output_tokens: int,
    output_path: str,
):
    """Evaluates essays using an LLM-based essay.

    Args:
        essay_path (str): Path to the CSV file containing essays and their metadata.
        prompts_path (str): Path to the CSV file containing prompt definitions.
        model_name (str): Name or path of the model to be used.
        prompt_name_evaluator (str): Name of the evaluation prompt template.
        prompt_name_writer (str): Name of the writing prompt template.
        llm_provider (str): Name of the LLM provider (e.g., openai, anthropic, nvidia).
        temperature (float): Sampling temperature for the LLM.
        max_output_tokens (int): Maximum number of tokens to generate per output.
        output_path (str): Path where the output results will be saved.
    """
    # Get model identifier
    model_id = model_name.split("/")[-1] if "/" in model_name else model_name
    path_to_save = f"{output_path}/writer/{model_id}/{prompt_name_writer}"
    check_directory_exists(path_to_save)

    # Load and preprocess essay dataset
    df_essay = pd.read_csv(essay_path)
    df_essay["competence"] = df_essay["competence"].apply(ast.literal_eval)
    df_essay["essay"] = df_essay["essay"].apply(ast.literal_eval)
    df_essay["essay"] = df_essay["essay"].apply(lambda x: "\n\n".join(x))
    df_essay[competence_cols] = pd.DataFrame(df_essay["competence"].tolist(), index=df_essay.index)

    # Load prompts and evaluator template
    df_prompts = pd.read_csv(prompts_path)
    user_prompt_evaluator, system_prompt_evaluator = get_prompts(prompt_name_evaluator)

    # Instantiate LLM
    llm = LLMFactory.get_predictor(llm_provider, model_name)
    llm_results = {}

    # Iterate through each essay for evaluation
    for idx, row in tqdm(df_essay.iterrows(), total=len(df_essay), desc="Classifying with LLM"):
        line = df_prompts[df_prompts['id'] == row['prompt']]
        motivational_texts = line['description'].values[0]

        title = row['title']
        essay = row['essay']

        # Generate a reference essay to be used as example
        try:
            generate_essay(
                prompts_path=prompts_path,
                model_name=model_name,
                prompt_name=prompt_name_writer,
                llm_provider=llm_provider,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                output_path="results/generated",
            )
        except Exception as e:
            # Continue even if generation fails
            pass

        # Load the generated reference essay
        generated_essay = pd.read_csv(f"results/generated/{model_id}/{prompt_name_writer}/essay_generated.csv")

        # Format the prompt for evaluation
        user_prompt_evaluator_formatted = user_prompt_evaluator.format(
            motivational_texts=motivational_texts,
            title=title,
            essay=essay,
            total_score=1000,
            score_I=200,
            score_II=200,
            score_III=200,
            score_IV=200,
            score_V=200,
            title_example=generated_essay['title_generated'],
            essay_example=generated_essay['essay_generated']
        )

        # Call LLM to generate evaluation
        result_data = llm_generator(
            prompt_name_evaluator,
            user_prompt_evaluator_formatted,
            system_prompt_evaluator,
            llm,
            temperature,
            max_output_tokens,
            all_keys_evaluator,
            max_retries=50
        )

        # Store results for each key
        for key in all_keys_evaluator:
            if key not in llm_results:
                llm_results[key] = []
            llm_results[key].append(result_data.get(key))

    # Append results to original DataFrame
    for key, value in llm_results.items():
        df_essay[key] = value

    # Save results and config
    create_directory(path_to_save)
    df_essay.to_csv(f"{path_to_save}/essay_results.csv", index=False)

    config_to_save = {
        "essay_path": essay_path,
        "prompts_path": prompts_path,
        "model_name": model_name,
        "prompt_name_evaluator": prompt_name_evaluator,
        "prompt_name_writer": prompt_name_writer,
        "llm_provider": llm_provider,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens
    }

    save_json(config_to_save, f"{path_to_save}/config.json")


if __name__ == "__main__":
    # Command-line interface for running the evaluator
    parser = argparse.ArgumentParser(description="Run few-shot evaluator on essay dataset using LLM.")
    parser.add_argument("--essay_path", type=str, default="dataset/testing.csv", help="Path to the essay CSV.")
    parser.add_argument("--prompts_path", type=str, default="dataset/prompts.csv", help="Path to the prompts CSV.")
    parser.add_argument("--model_name", type=str, default="mistralai/mistral-medium-3-instruct", help="Model name or path.")
    parser.add_argument("--prompt_name_evaluator", type=str, default="evaluator_few_shot", help="Name of the prompt template.")
    parser.add_argument("--prompt_name_writer", type=str, default="writer_zero_shot", help="Name of the prompt template.")
    parser.add_argument("--llm_provider", type=str, default="nvidia", help="LLM provider name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature.")
    parser.add_argument("--max_output_tokens", type=int, default=8000, help="Max number of output tokens.")
    parser.add_argument("--output_path", type=str, default="results/essay", help="Base path to save results.")

    args = parser.parse_args()

    evaluator_writer(
        essay_path=args.essay_path,
        prompts_path=args.prompts_path,
        model_name=args.model_name,
        prompt_name_evaluator=args.prompt_name_evaluator,
        prompt_name_writer=args.prompt_name_writer,
        llm_provider=args.llm_provider,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        output_path=args.output_path
    )
