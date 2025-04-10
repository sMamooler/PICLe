import logging
import os
import sys
import time

import nltk
import openai
import tiktoken
from transformers import AutoTokenizer
from vllm import SamplingParams

nltk.download("punkt")
from utils.parsers import parse_llm_output

openai.api_key = os.environ["OPENAI_API_KEY"]
sys.stdout.reconfigure(encoding="utf-8")

log = logging.getLogger(__name__)
# TODO(@smamooler): clean this script up and add docstrings

COSTS = {
    "gpt-3.5-turbo": {"input": 0.0000015, "output": 0.000002},
    "gpt-3.5-turbo-16k": {"input": 0.000003, "output": 0.000004},
    "gpt-4": {"input": 0.00003, "output": 0.00006},
    "text-davinci-003": {"input": 0.00002, "output": 0.00002},
}

ENCODINGS = {
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    "gpt-4": "cl100k_base",
    "text-davinci-003": "p50k_base",
}


def _num_tokens_from_string(sentence: str, model: str):
    """Computes the number of tokens in a string using the encoding of the model."""
    encoding_name = ENCODINGS[model]
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(sentence))
    return num_tokens


def _compute_cost(input_: str, output: str, model: str):
    """Computes the cost of a prompt based on the number of tokens in the input and output."""
    input_len = _num_tokens_from_string(input_, model)
    output_len = _num_tokens_from_string(output, model)
    return input_len * COSTS[model]["input"] + output_len * COSTS[model]["output"]


def call_gpt(
    model: str,
    batch_messages: list,
    prompt_strategy: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
) -> tuple[list[str], list[str], float]:
    """Call the OpenAI API to generate responses for a batch of messages.

    Args:
        model (str): The GPT model to use.
        batch_messages (list): The list of messages to generate responses for.
        prompt_strategy (str): The prompt strategy used to generate the output.
        temperature (float, optional): Decoding temperature. Defaults to 0.0.
        top_p (float, optional): Decoding top-p. Defaults to 1.0.
        max_tokens (int, optional): Maximum number of tokens to generarte. Defaults to 1024.

    Returns:
        tuple[list[str], list[str], float]: The list of generated responses, the list of predicted entities, and the cost.
    """
    batch_gpt_output = []
    batch_pred_entities = []
    costs = 0

    for messages in batch_messages:
        nb_trial = 0
        stop = False
        while not stop:
            try:
                nb_trial += 1
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )
                output = response["choices"][0]["message"]["content"]
                cost = _compute_cost(
                    "".join([m["content"] for m in messages]), output, model
                )
                parsed_output = parse_llm_output(output, prompt_strategy)
                stop = True
                batch_pred_entities.append(parsed_output)
                batch_gpt_output.append(output)
                costs += cost

            except openai.error.RateLimitError as e:
                log.info("Rate limit error, waiting 10 seconds")
                time.sleep(10)
            except openai.error.Timeout as e:
                log.info("Timeout error, waiting 10 seconds")
                time.sleep(10)
            except IndexError as e:
                if nb_trial > 5:
                    log.info(
                        f"Index error {e}.\nModel Output:\n",
                        output,
                        "Too many trials! Moving on...",
                    )
                    stop = True
                    parsed_output = []
                else:
                    log.info(f"Index error {e}. Trying again.\nModel Output:\n", output)
            except ValueError as e:
                if nb_trial > 5:
                    log.info(
                        f"Value error {e}.\nModel Output:\n",
                        output,
                        "Too many trials! Moving on...",
                    )
                    stop = True
                    parsed_output = []
                else:
                    temperature += 0.1
                    log.info(f"Value error {e}.\nModel Output:\n", output)

    return batch_gpt_output, batch_pred_entities, cost


def call_hf(
    langauge_model,
    tokenizer: AutoTokenizer,
    batch_messages: list,
    prompt_strategy: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    seed: int = 12345,
) -> tuple[list[str], list[str], float]:
    """Call the Hugging Face API to generate responses for a batch of messages.

    Args:
        langauge_model: vLLM language model
        tokenizer (AutoTokenizer): LLM tokenizer
        batch_messages (list): The list of messages to generate responses for.
        prompt_strategy (str): The prompt strategy used to generate the output.
        temperature (float, optional): Decoding temperature. Defaults to 0.0.
        top_p (float, optional): Decoding top-p. Defaults to 1.0.
        max_tokens (int, optional): Maximum number of tokens to generarte. Defaults to 512.
        seed (int, optional): Generation seed. Defaults to 12345.

    Returns:
        tuple[list[str], list[str], float]: The list of generated responses, the list of predicted entities, and the cost
    """

    formatted_messages = list(
        map(
            lambda x: tokenizer.apply_chat_template(
                x, return_tensors="pt", add_generation_prompt=True, tokenize=False
            ),
            batch_messages,
        )
    )
    # torch.manual_seed(seed)
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens
    )
    llm_outputs = langauge_model.generate(formatted_messages, sampling_params)

    outputs = []
    parsed_outputs = []

    for i in range(len(llm_outputs)):
        output = llm_outputs[i].outputs[0].text
        outputs.append(output)
        parsed = parse_llm_output(output, prompt_strategy)
        parsed_outputs.append(parsed)

        log.debug(f"output:\n{output}")
        log.debug(f"parsed output:\n{parsed}")

    cost = 0

    return outputs, parsed_outputs, cost
