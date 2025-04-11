import logging

import regex as re

log = logging.getLogger(__name__)


def _parse_bio_ner_output(llm_output: str) -> list[str]:
    # TODO(@smamooler): Is this function needed?

    # Split the text by '||' to separate each word and label pair
    word_label_pairs = llm_output.split("||")

    current_entity = ""
    entities = []
    # Iterate through each word and label pair
    for pair in word_label_pairs:
        # Split the pair by '|' to separate the word and label
        word, label = pair.split("|")
        word = word.strip()
        label = label.strip()

        # Check if the label indicates the beginning of a disease entity
        if label.startswith("B-"):
            # If there was a previous disease entity, store it
            if current_entity:
                entities.append(current_entity.strip())
            # Start a new disease entity with the current word
            current_entity = word
        # Check if the label indicates a word within a disease entity
        elif label.startswith("I-"):
            # Append the current word to the existing disease entity
            current_entity += " " + word
        # Check if the label indicates the end of a disease entity
        elif label == "O":
            # If there was a disease entity being constructed, store it
            if current_entity:
                entities.append(current_entity.strip())
                # Reset current_ entity for the next entity
                current_entity = ""
        else:
            raise ValueError(f"Label {label} is not supported.")
    return entities


def _parse_standard_output(llm_output: str) -> list[str]:
    """Parse the output of the LLM in list format to extract the entities."""

    llm_output = llm_output.replace("'", "")
    # find all the entities between the first square brackets mentioned in the model output
    if ("[" in llm_output) and ("]" in llm_output):
        pattern = re.compile(r"\[(.*?)\]")
        try:
            output = ""
            for e in re.findall(pattern, llm_output):
                output += "," + e

        except IndexError as e:
            if "\n" in llm_output:
                pattern = re.compile(r"\[(?:.|\n)*?\]")
                try:
                    output = re.findall(pattern, llm_output)[0]
                except IndexError as e:
                    output = ""
                    log.info(f"Line {llm_output} is not formatted correctly.")
            else:
                output = ""
                log.info(f"Line {llm_output} is not formatted correctly.")

        output = output.replace("'", "").split(",")
    else:
        if ": " in llm_output:
            llm_output = llm_output.split(": ")[1].strip()

        output = llm_output.split("\n")[0].strip()
        output = output.replace("'", "")
        if output.endswith("."):
            output = output[:-1]
        if output.startswith("[") and output.endswith("]"):
            output = output[1:-1]
        elif output.startswith("["):
            output = output[1:]
        elif output.endswith("]"):
            output = output[:-1]

        output = output.split(",")

    return output


def _parse_gpt_ner_output(llm_output: str) -> list[str]:
    """Parse the output of the LLM in GPT-NER format to extract the entities."""

    pattern = re.compile(r"@@(.*?)##")
    output = re.findall(pattern, llm_output)
    return output


def _parse_prompt_ner_output(llm_output: str) -> list[str]:
    """Parse the output of the LLM in Prompt-NER format to extract the entities."""

    lines = llm_output.split("\n")
    lines = [line for line in lines if line]
    output = []
    for line in lines:
        if "| yes |" in line:
            output.append(line.split("| yes |")[0].strip())
        elif "| yes" in line:
            output.append(line.split("| yes")[0].strip())
        else:
            continue
    return output


def parse_llm_output(llm_output, prompt_strategy) -> list[str]:
    """Parse the output of the LLM to extract the entities.
    Args:
        llm_output (str): The output of the LLM.
        prompt_strategy (str): The prompt strategy used to generate the output.

    Returns:
        list[str]: The list of extracted entities.
    """

    if prompt_strategy == "standard":
        output = _parse_standard_output
    elif prompt_strategy == "GPT-NER":
        output = _parse_gpt_ner_output(llm_output)
    elif prompt_strategy == "Prompt-NER":
        output = _parse_prompt_ner_output(llm_output)
    elif prompt_strategy == "BIO-NER":
        output = _parse_bio_ner_output(llm_output)
    else:
        raise ValueError(f"Prompt strategy {prompt_strategy} is not supported.")

    for i, ent in enumerate(output):
        output[i] = ent.replace('"', "").strip().lower()

    return output
