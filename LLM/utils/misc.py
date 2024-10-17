'''
Interacting with LLM includes the following steps:
1. Generating a prompt to give to the model 
2. Generating a response from the model
3. Post-processing the response
4. Returning the response


Possible modes of this interface include: 
1. Generating plan from current state and other additional information
2. Generating the next action from the current state (SayCan, Text2Motion like interface)
2. Generating a LMP Qfunction and Value function. (May need debugging)
3. Generating the 
'''
import os 
from pathlib import Path 
from termcolor import colored
import json 
import yaml
from copy import deepcopy
from collections import namedtuple
import itertools
from openai import OpenAI
import tiktoken


def find_project_base_dir():
    file_path = os.path.abspath("")
    project_base_dir = Path(file_path)
    
    while 'stalm' not in project_base_dir.name:
        project_base_dir = project_base_dir.parent
    return project_base_dir


def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }
    
    for message in messages:
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and message.get("function_call"):
            print(colored(f"assistant: {message['function_call']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "function":
            print(colored(f"function ({message['name']}): {message['content']}\n", role_to_color[message["role"]]))
        else: 
            raise ValueError(f"Invalid role: {message['role']}")



def read_yaml(file_path: str)-> dict:
    r"""
    Read a yaml file and return the data as a dictionary.
    args:
    - file_path: str, path to the yaml file.
    return:
    - data: dict, data from the yaml file."""

    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def read_json(file_path: str)-> dict:
    r"""
    Read a json file and return the data as a dictionary.
    args:
    - file_path: str, path to the json file.
    return:
    - data: dict, data from the json file."""

    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def read_txt(file_path: str) -> str: 
    r"""
    Read a text file and return the data as a string.
    args:
    - file_path: str, path to the text file.
    return:
    - data: str, data from the text file."""

    with open(file_path, 'r') as f:
        data = f.read()
    return data

def write_yaml(file_path: str, data: dict):
    r"""
    Write a dictionary to a yaml file.
    args:
    - file_path: str, path to the yaml file.
    - data: dict, data to be written to the yaml file.
    return:
    - None."""

    with open(file_path, 'w') as f:
        yaml.dump(data, f)

def write_json(file_path: str, data: dict):
    r"""
    Write a dictionary to a json file.
    args:
    - file_path: str, path to the json file.
    - data: dict, data to be written to the json file.
    return:
    - None."""

    with open(file_path, 'w') as f:
        json.dump(data, f)

def write_txt(file_path: str, data: str):
    with open(file_path, 'w') as f:
        f.write(data)

def num_tokens_from_string(string: str, encoding_name: str="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4"
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens



