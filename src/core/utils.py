import re
import os
import json
import yaml
from typing import Optional, Dict, Union

def check_directory_exists(directory_path: str) -> None:
    """
    Checks if a directory exists and raises an error if it does.

    Args:
        directory_path (str): The path to the directory to check.

    Raises:
        FileExistsError: If the specified directory already exists.
    """
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        raise FileExistsError(f"The directory '{directory_path}' already exists.")

def create_directory(directory_path: str) -> bool:
    """Create a directory if it does not exist. 

    Args:
        directory_path (str): The path of the directory to create.

    Returns:
        bool: True if the directory was successfully created or already exists, False if there was an error.
    
    Raises:
        OSError: If an error occurs while creating the directory.
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except OSError as e:
        print(f"Error creating directory: {e}")
        return False

def save_json(data, file_path):
    """
    Saves the given dictionary to a JSON file.

    Args:
        data (dict): The dictionary containing data to be saved.
        file_path (str): The path where the JSON file should be saved.

    Returns:
        bool: True if the file was saved successfully, False otherwise.
    
    Example:
        data = {"name": "John", "age": 30}
        file_path = "output.json"
        success = save_json(data, file_path)
    """
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        return True
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        return False


def read_json(file_path: str) -> dict:
    """Reads a JSON file and returns its content as a dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The content of the JSON file.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def extract_json_from_text(text):
    """
    Extract the JSON content between ```json and ```
    """
    match = re.search(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def get_prompts(prompt_name: str, path_prompts: str = "configs/prompts.yaml"):
    """
    Retrieves the user and system prompts from a YAML configuration file.

    Args:
        prompt_name (str): The name of the prompt section to retrieve from the YAML file.
        path_prompts (str, optional): The path to the YAML file containing the prompts.
            Defaults to "configs/prompts.yaml".

    Returns:
        tuple[str, str]: A tuple containing the user prompt and the system prompt 
            if the prompt_name is found in the YAML file.

        str: An error message if the prompt_name is not found in the YAML file.
    """

    with open(path_prompts, 'r') as file:
        prompts = yaml.safe_load(file)

    if prompt_name in prompts:  
        user_prompt = prompts[prompt_name].get("user_prompt", "User prompt not found.")
        system_prompt = prompts[prompt_name].get("system_prompt", "System prompt not found.")
        return user_prompt, system_prompt
    else:
        return f"Prompt '{prompt_name}' not found."

def extract_evaluation_info(xml_string: str) -> Optional[Dict[str, Union[int, str]]]:
    """
    Extracts evaluation-related fields from a given XML-like string.

    Args:
        xml_string (str): A string containing the evaluation tags.

    Returns:
        Optional[Dict[str, Union[int, str]]]: A dictionary with keys like 'nota_competencia_I',
            'explicacao_competencia_I', etc., containing the extracted content.
            'nota_*' fields are converted to int. Returns None if no tags are found.
    """
    tags = [
        "nota_competencia_I",
        "explicacao_competencia_I",
        "nota_competencia_II",
        "explicacao_competencia_II",
        "nota_competencia_III",
        "explicacao_competencia_III",
        "nota_competencia_IV",
        "explicacao_competencia_IV",
        "nota_competencia_V",
        "explicacao_competencia_V",
        "comentarios_gerais",
        "sugestoes_melhoria"
    ]

    result = {}
    for tag in tags:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, xml_string, re.DOTALL)
        if match:
            value = match.group(1).strip()
            if tag.startswith("nota_"):
                try:
                    value = int(value)
                except ValueError:
                    pass  # mantém como string se não puder converter, mas pode levantar erro se preferir
            result[tag] = value

    return result if result else None



def extract_title_and_text(xml_string: str) -> Optional[Dict[str, str]]:
    """
    Extracts the content of <titulo>, <título>, <title> and <texto>, <text> tags from a given XML-like string.

    Args:
        xml_string (str): A string containing title and text tags.

    Returns:
        Optional[Dict[str, str]]: A dictionary with 'title_generated' and 'essay_generated' keys containing
            the extracted content. Returns None if the pattern is not found.
    """
    pattern = r"<(?:titulo|título|title)>(.*?)</(?:titulo|título|title)>\s*<(?:texto|text)>(.*?)</(?:texto|text)>"
    match = re.search(pattern, xml_string, re.DOTALL | re.IGNORECASE)
    if match:
        return {
            'title_generated': match.group(1).strip(),
            'essay_generated': match.group(2).strip()
        }
    return None




