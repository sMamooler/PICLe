"""This module contains functions to perturb the annotations in a sample."""

import logging
import random
from typing import Any, Optional

import CONSTANTS as CONSTANTS
import numpy as np

log = logging.getLogger(__name__)


def _deletion_substitution(
    entity: str, perturbation_factor: float, substituted_entity: str
) -> Optional[str]:
    """Delete and/or substitute an entity based on the perturbation factor."""
    delete_entity = np.random.binomial(2, p=perturbation_factor)
    if delete_entity:
        return None

    substitute_entity = np.random.binomial(2, p=perturbation_factor)
    if substitute_entity:
        return substituted_entity
    else:
        return entity


def _addition_substitution(
    entity: str, perturbation_factor: float, added_entity: str, substituted_entity: str
) -> Optional[list[str]]:
    """Add and/or substitute an entity based on the perturbation factor."""

    perturbed_entities = []
    add_entity = np.random.binomial(2, p=perturbation_factor)
    if add_entity:
        perturbed_entities += [added_entity]

    substitute_entity = np.random.binomial(2, p=perturbation_factor)
    if substitute_entity:
        perturbed_entities += [substituted_entity]
    else:
        perturbed_entities += [entity]

    return perturbed_entities


def _perturb_entity(
    entity: str,
    entity_list: list[str],
    perturbation_type: str,
    perturbation_factor: float,
) -> Optional[list[str]]:
    """Pertrub an entity based on the perturbation type and factor."""
    perturbed_entities = []

    added_entity = random.choice(entity_list)
    substituted_entity = random.choice(entity_list)

    if perturbation_type in ["addition", "substitution", "deletion"]:
        perturb = np.random.binomial(2, p=perturbation_factor)
        if perturb:
            if perturbation_type == "addition":
                perturbed_entities += [entity, added_entity]
            elif perturbation_type == "deletion":
                return None
            elif perturbation_type == "substitution":
                perturbed_entities += [substituted_entity]
            else:
                raise ValueError(f"perturbation type {perturbation_type} not supported")
        else:
            perturbed_entities += [entity]

    elif perturbation_type == "addition-substitution":
        perturbed_entities += _addition_substitution(
            entity, perturbation_factor, added_entity, substituted_entity
        )

    elif perturbation_type == "deletion-substitution":

        perturbed_entity = _deletion_substitution(
            entity, perturbation_factor, substituted_entity
        )
        if perturbed_entity:
            perturbed_entities += [perturbed_entity]

    else:
        raise ValueError(f"perturbation type {perturbation_type} not supported")

    return perturbed_entities


def perturb_annotations(
    sample: dict[str, Any],
    entity_list: list[str],
    perturbation_type: str,
    perturbation_factor: float,
) -> tuple[dict[str, Any], float, float, int, int]:
    """Pertrub the annotations in a sample based on the perturbation type and factor.

    Args:
        sample (dict[str, Any]): The sample to perturb.
        entity_list (list[str]): The list of entities to choose from in case of addition and susbsitution.
        perturbation_type (str): The type of perturbation. One of ["addition", "substitution", "deletion", "addition-substitution", "deletion-substitution"].
        perturbation_factor (float): The perturbation factor. A value between 0 and 1.

    Returns:
        tuple[dict[str, Any], float, float, int, int]: The perturbed sample, the perturbed precision, the perturbed recall, the number of entities in the perturbed sample, and the difference in the number of entities between the perturbed and the original sample.
    """
    if len(sample["entities"]) == 0:
        return sample, 1, 1, 0, 0

    perturbed_entities = []

    for entity in sample["entities"]:

        peturbed_entity = _perturb_entity(
            entity, entity_list, perturbation_type, perturbation_factor
        )
        if peturbed_entity:
            perturbed_entities += peturbed_entity

    perturbed_sample = {"text": sample["text"], "entities": perturbed_entities}
    perturbed_sample_prc = (
        len(set(perturbed_entities).intersection(set(sample["entities"])))
        / len(perturbed_entities)
        if len(perturbed_entities) > 0
        else 0
    )
    perturbed_sample_rec = len(
        set(perturbed_entities).intersection(set(sample["entities"]))
    ) / len(sample["entities"])

    num_entities = len(perturbed_sample["entities"])
    num_entities_diff = len(perturbed_sample["entities"]) - len(sample["entities"])

    log.debug(f"Before perturbation: {sample['entities']}")
    log.debug(f"After perturbation: {perturbed_entities}")
    log.debug(f"Perturbed nb entities: {num_entities}")
    log.debug(f"Perturbed nb entities diff: {num_entities_diff}")
    log.debug(f"Perturbed PRC: {perturbed_sample_prc}")
    log.debug(f"Perturbed REC: {perturbed_sample_rec}")

    return (
        perturbed_sample,
        perturbed_sample_prc,
        perturbed_sample_rec,
        num_entities,
        num_entities_diff,
    )
