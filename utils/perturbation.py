"""This module contains functions to perturb the annotations in a sample."""

import logging
import random
import string
from typing import Any, Optional

import CONSTANTS as CONSTANTS
import nltk
import numpy as np

nltk.download("words")
nltk.download("stopwords")
import ssl

from nltk.corpus import stopwords, words

### for nltk downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("punkt")

log = logging.getLogger(__name__)


### utility functions for analysis with partially correct demonstrations


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


### utility functions for analysis with corrupted demonstrations


def _random_id_labels(
    data: list,
):
    """Replace the ground-truth labels in a sample with random words from the in-domain label space."""
    id_lable_space = []
    for sample in data:
        id_lable_space.extend(sample["entities"])
    id_lable_space = set(id_lable_space)

    for sample in data:
        correct_labels = sample["entities"]
        wrong_id_labels = random.sample(
            id_lable_space.difference(set(correct_labels)), len(correct_labels)
        )
        sample["entities"] = wrong_id_labels

    return data


def _random_ood_labels(
    data: list,
):
    """Replace the ground-truth labels in a sample with random words from the out-of-domain label space."""
    id_lable_space = []
    for sample in data:
        id_lable_space.extend(sample["entities"])
    id_lable_space = set(id_lable_space)

    ood_label_space = set(words.words()).difference(set(id_lable_space))
    for sample in data:
        correct_labels = sample["entities"]
        wrong_ood_labels = random.sample(ood_label_space, len(correct_labels))
        sample["entities"] = wrong_ood_labels

    return data


def _random_ood_labels_from_text(data: list):
    """Replace the ground-truth labels in a sample with random words from the text of that samples."""
    stop_words = set(stopwords.words("english"))
    for sample in data:
        correct_labels = sample["entities"]
        text = sample["text"]
        text_words = [
            w
            for w in text.split()
            if w.lower() not in stop_words
            and w not in correct_labels
            and w not in string.punctuation
        ]
        if len(text_words) < len(correct_labels):
            print(text, correct_labels)
            wrong_ood_labels_from_text = []
        else:
            wrong_ood_labels_from_text = random.sample(text_words, len(correct_labels))
            for l_id, wrong_label in enumerate(wrong_ood_labels_from_text):
                for p in string.punctuation:
                    wrong_ood_labels_from_text[l_id] = wrong_label.replace(p, "")

        sample["entities"] = wrong_ood_labels_from_text
    return data


def _swapped_id_labels(
    data: list,
):
    """Swap the id labels in a sample with other samples in the dataset."""
    for sample in data:
        other_samples = data.copy()
        other_samples.remove(sample)
        swapped_id_labels = random.sample(other_samples, 1)[0]["entities"]
        sample["entities"] = swapped_id_labels

    return data


def corrupt_labels(
    data: list,
    wrong_annotation_type: str,
):
    """Replaces the ground-truth labels with wrong labels.
    Args:
        data (list): The data to replace the annotations in.
        wrong_annotation_type (str): The type of wrong annotation to use. One of ["random_id_labels", "random_ood_labels", "random_ood_labels_from_text", "swapped_id_labels"].
    Returns:
        list: The data with the added wrong annotations.
    Raises:
        ValueError: If the wrong annotation type is not supported.
    """
    if wrong_annotation_type == "random_id_labels":
        data = _random_id_labels(data)
    elif wrong_annotation_type == "random_ood_labels":
        data = _random_ood_labels(data)
    elif wrong_annotation_type == "random_ood_labels_from_text":
        data = _random_ood_labels_from_text(data)
    elif wrong_annotation_type == "swapped_id_labels":
        data = _swapped_id_labels(data)
    else:
        raise ValueError(f"wrong_annotation_type {wrong_annotation_type} not supported")
    return data


def corrupt_texts(data: list, corrupt_label: bool, shuffle: bool):
    """Replace text (and labels if corrupt_label is True) in a sample with corrupted text (and labels).
    Args:
        data (list): The data to replace the text in.
        corrupt_label (bool): Whether to corrupt the labels or not.
        shuffle (bool): Whether to shuffle the text words or not.
    Returns:
        list: The data with the added wrong text (and labels if corrupt_label is True).
    """
    id_lable_space = []
    for sample in data:
        id_lable_space.extend(sample["entities"])
    id_lable_space = set(id_lable_space)

    ood_label_space = set(words.words()).difference(set(id_lable_space))
    for sample in data:
        text = sample["text"]
        random_ood_words = random.sample(ood_label_space, len(set(sample["entities"])))
        entity2random_ood_words = dict(zip(set(sample["entities"]), random_ood_words))
        ood_labels = [None for _ in range(len(sample["entities"]))]
        for i, entity in enumerate(sample["entities"]):
            ood_word = entity2random_ood_words[entity]
            text = text.replace(entity, ood_word)
            if shuffle:
                text_words = text.split()
                random.shuffle(text_words)
                text = " ".join(text_words)
            ood_labels[i] = ood_word

        sample["text"] = text
        if corrupt_label:
            sample["entities"] = ood_labels

    return data
