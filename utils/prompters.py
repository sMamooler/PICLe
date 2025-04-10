import json
import random
from typing import Any

import CONSTANTS
import regex as re


def _get_gpt_ner_demos(text, entities):
    """
    Constructs the demonstration in GPT-NER format
    example:
        input: Degradation of MAC13243 and studies of the interaction of resulting thiourea compounds...
        output: Degradation of MAC13243 and studies of the interaction of resulting @@thiourea## compounds...
    """
    output = str(text)
    # the list of entities needs to be sorted to avoid situations like @@17β-@@estradiol####.
    entities = list(sorted([e for e in entities], key=len, reverse=True))
    for entity in set(entities):
        pattern = re.compile(rf"(?<![a-zA-Z@]){re.escape(entity)}(?![a-zA-Z#])")
        output = re.sub(pattern, f"@@{entity}##", output)

    return output


def _get_prompt_ner_demos(entities: str, entity_type: str) -> str:
    """Constructs the demonstration in Prompt-NER format.
    Example:
        input: Degradation of MAC13243 and studies of the interaction of resulting thiourea compounds...
        output: MAC13243 | yes | because it is a chemical
    """

    output = ""
    for ent in entities:
        output += f"{ent} | yes | because it is a {entity_type}\n"
    return output


def _get_entity_definition(entity_type: str, nb_entity_examples: int = 0):
    """Gets the definition of an entity type. If nb_entity_examples is provided, it also includes examples of entities."""
    # entity definition by ChatGPT unless otherwise specified
    try:
        entity_def = CONSTANTS.ENTITY2DEFINITION[entity_type]
    except KeyError:
        raise ValueError(f"Type {entity_type} is not supported.")

    if nb_entity_examples:
        try:
            entity_examples_list = json.load(
                open(CONSTANTS.ENTITY2EXAMPLESFILE[entity_type], "r")
            )
        except KeyError:
            raise ValueError(f"Type {entity_type} is not supported.")

        random.shuffle(entity_examples_list)
        entity_examples_list = random.sample(entity_examples_list, nb_entity_examples)
        entity_def += f"\nExamples of {entity_type} entities are:\n{','.join(entity_examples_list)}"

    return entity_def


def _build_demo_chat(
    demo_dataset: list[dict[str, Any]],
    prompt_strategy: str,
    entity_type: str,
) -> list[dict[str, str]]:
    """Build the demonstration chat based on the prompt strategy."""

    demonstrations = []
    for sample in demo_dataset:
        text = sample["text"]
        entities = sample["entities"]

        if prompt_strategy == "GPT-NER":
            output = _get_gpt_ner_demos(text, entities)

        elif prompt_strategy == "Prompt-NER":
            output = _get_prompt_ner_demos(entities, entity_type)

        elif prompt_strategy == "standard":
            output = entities
        else:
            raise ValueError(f"Prompt strategy {prompt_strategy} is not supported.")

        demonstrations += [
            {"role": "user", "content": f"input: {text}\n{entity_type} entities:"},
            {"role": "assistant", "content": f"{output}\n\n"},
        ]

    return demonstrations


def get_prompt(
    input: str,
    demos: list[dict[str, Any]],
    entity_type: str,
    prompt_strategy: str,
    entity_definition: bool = True,
    nb_entity_examples: int = 0,
) -> list[dict[str, str]]:
    """Constructs the prompt for the entity extraction task.

    Args:
        input (str): The input text to extract entities from.
        demos (list[dict[str, Any]]): The demonstration dataset.
        entity_type (str): The type of entity to extract.
        prompt_strategy (str): The prompt strategy to use. One of 'standard', 'GPT-NER', 'Prompt-NER'.
        entity_definition (bool): Whether to include the definition of the entity type.
        nb_entity_examples (int): The number of examples of entities to include in the entity definition.

    Returns:
        list[dict[str, str]]: The prompt messages.
    """

    if entity_definition:
        entity_def = _get_entity_definition(entity_type, nb_entity_examples)

    entity_type_formulation = {
        "chemical": "chemicals",
        "gene/protein": "genes and proteins",
        "disease/illness": "diseases",
    }
    other_entity_types = [
        et
        for et in ["chemical", "gene/protein", "disease/illness"]
        if et != entity_type
    ]
    other_entity_types = ", and ".join(
        [entity_type_formulation[et] for et in other_entity_types]
    )

    instructions = f"Your task is to extract all of the {entity_type_formulation[entity_type]} mentioned in a given abstract published in PubMed.\n"
    if entity_definition:
        instructions += f"{entity_def}\n\n"

    demonstrations = _build_demo_chat(demos, prompt_strategy, entity_type)

    instructions += f"""Please extract all of the entities corresponding to {entity_type_formulation[entity_type]} from the following paragraph, same way as they are marked in the examples.
Make sure to include all and only the {entity_type_formulation[entity_type]} mentioned in the text, but not the {other_entity_types}. If there are no {entity_type_formulation[entity_type]} entities in the text output 'None'. You will be penalized if you include an entity more or less than the number of times it appears in the text."""

    messages = [
        {"role": "user", "content": instructions},
        {"role": "assistant", "content": "understood! let's get started!"},
    ]
    if demonstrations:
        demonstrations[0]["content"] = (
            f"## Here are some examples:\n" + demonstrations[0]["content"]
        )
        messages += demonstrations

    messages += [
        {"role": "user", "content": f"input: {input}\n{entity_type} entities:"}
    ]
    return messages
