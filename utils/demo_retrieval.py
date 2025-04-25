"""This module contains functions to retrieve demos for a given inference sample."""

import random
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from transformers import BertModel, BertTokenizer

import CONSTANTS as CONSTANTS

# TODO(smamooler): Add the functions from demo_retrieval_utils.py to this file, and clean them.
tokenizer = BertTokenizer.from_pretrained("pritamdeka/S-PubMedBert-MS-MARCO")
model = BertModel.from_pretrained("pritamdeka/S-PubMedBert-MS-MARCO")


def embed_samples(
    samples: list,
    entity_embedding_type: Optional[str] = None,
) -> torch.Tensor:
    """Embed the samples using the model.

    Args:
        samples (list): The samples to embed.
        entity_embedding_type (Optional[str], optional): The type of entity embeddings to use. Defaults to None.
    Returns:
        torch.Tensor: The embeddings of the samples.
    """
    samples_texts = [sample["text"] for sample in samples]
    tokenized_demo_text = tokenizer(
        samples_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )
    with torch.no_grad():
        text_embeddings = model(
            **tokenized_demo_text, output_hidden_states=True, return_dict=True
        ).pooler_output
    return text_embeddings


def cluster_demos(
    demo_dataset: list[dict[str, Any]],
    demo_dict: dict[str, Any],
    demo_embeddings_dict: dict[str, Any],
    num_clusters: int,
) -> dict[str, Any]:
    """Cluster the demonstrations using KMeans and save the cluster information to a file.

    Args:
        demo_dataset (list[dict[str, Any]]): The demonstrations dataset.
        demo_dict (dict[str, Any]): The demonstrations dictionary with keys "non-null", "null", "all".
        demo_embeddings_dict (dict[str, Any]): The demonstrations embeddings dictionary with keys "non-null", "null", "all".
        num_clusters (int): The number of clusters.

    Returns:
        dict[str, Any]: The KMeans model for each of the keys in demo_dict.
    """
    kmeans_dict = {"non-null": None, "null": None, "all": None}
    for key in demo_dict:
        kmeans_dict[key] = KMeans(
            n_clusters=num_clusters if key != "null" else 1, random_state=0, n_init=10
        ).fit(demo_embeddings_dict[key])

    cluster_dict = {
        "cluster": kmeans_dict["all"].labels_,
        "text": [sample["text"] for sample in demo_dataset],
        "entities": [sample["entities"] for sample in demo_dataset],
    }
    cluster_df = pd.DataFrame(cluster_dict)
    cluster_df = cluster_df.sort_values(by=["cluster"])

    return kmeans_dict, cluster_df


def compute_cosin_sim_matrix(
    demo_dict: dict[str, Any],
    demo_embeddings_dict: dict[str, Any],
    inference_embeddings: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the cosine similarity matrix between the inference embeddings and the demonstrations embeddings.

    Args:
        demo_dict (dict[str, Any]): The demonstrations dictionary with keys "non-null", "null", "all".
        demo_embeddings_dict (dict[str, Any]): The demonstrations embeddings dictionary with keys "non-null", "null", "all".
        inference_embeddings (np.ndarray): The embedding of inference samples.

    Returns:
        dict[str, np.ndarray]: The cosine similarity matrix for each of the keys in demo_dict.
    """

    cosine_sim_matrix_dict = {"non-null": None, "null": None, "all": None}
    for key in demo_dict:
        norm_matrix = np.matmul(
            np.expand_dims(np.linalg.norm(inference_embeddings, axis=1), axis=1),
            np.expand_dims(np.linalg.norm(demo_embeddings_dict[key], axis=1), axis=1).T,
        )
        dot_product_matrix = np.matmul(
            inference_embeddings, demo_embeddings_dict[key].T
        )
        cosine_sim_matrix = dot_product_matrix / norm_matrix
        cosine_sim_matrix_dict[key] = cosine_sim_matrix
    return cosine_sim_matrix_dict


def _get_kmeans_knn_demos(
    cosine_sim_matrix_dict: dict[str, np.ndarray],
    demo_dict: dict[str, Any],
    kmeans_dict: dict[str, Any],
    num_shots: int,
    index: int,
    sample_index: int,
    discard_null_samples: bool = False,
) -> list[dict[str, Any]]:
    """Get the demonstrations obtained using the kmeans_knn method."""
    if discard_null_samples:
        non_null_cosine_sim_scores = cosine_sim_matrix_dict["non-null"][
            index + sample_index
        ]
        non_null_demos = kmeans_knn_demo_retrieval(
            num_shots - 1,
            demo_dict["non-null"],
            kmeans_dict["non-null"],
            non_null_cosine_sim_scores,
        )
        demos = non_null_demos
    else:
        all_cosine_sim_scores = cosine_sim_matrix_dict["all"][index + sample_index]
        all_demos = kmeans_knn_demo_retrieval(
            num_shots, demo_dict["all"], kmeans_dict["all"], all_cosine_sim_scores
        )
        demos = all_demos

    return demos


def _knn_demo_retrieval(nb_shots: int, demo_dataset: list, arg_sort: list) -> list:
    """KNN retrieval from the demonstration dataset for ICL."""
    shots = []
    index = 0

    while len(shots) < nb_shots:
        demo_index = arg_sort[index]
        sample = demo_dataset[demo_index]
        shots.append(sample)
        index += 1

    return shots


def _get_knn_demos(
    cosine_sim_matrix_dict: dict[str, np.ndarray],
    demo_dict: dict[str, Any],
    num_shots: int,
    index: int,
    sample_index: int,
    discard_null_samples: bool = False,
) -> list[dict[str, Any]]:
    """Get the demonstrations obtained using the knn method."""

    if discard_null_samples:
        non_null_cosine_sim_scores = cosine_sim_matrix_dict["non-null"][
            index + sample_index
        ]
        args_sorted_non_null_cosine_sim = np.argsort(-non_null_cosine_sim_scores)
        non_null_demos = _knn_demo_retrieval(
            num_shots - 1, demo_dict["non-null"], args_sorted_non_null_cosine_sim
        )
        demos = non_null_demos
    else:
        all_cosine_sim_scores = cosine_sim_matrix_dict["all"][index + sample_index]
        args_sorted_all_cosine_sim = np.argsort(-all_cosine_sim_scores)
        all_demos = _knn_demo_retrieval(
            num_shots, demo_dict["all"], args_sorted_all_cosine_sim
        )
        demos = all_demos

    return demos


def _get_random_demos(
    demo_dict: dict[str, Any],
    num_shots: int,
    discard_null_samples: bool = False,
) -> list[dict[str, Any]]:
    """Get the demonstrations obtained using the random method."""

    if discard_null_samples:
        non_null_demos = random.sample(
            demo_dict["non-null"],
            min(len(demo_dict["non-null"]), max(0, num_shots - 1)),
        )
        demos = non_null_demos
    else:
        all_demos = random.sample(
            demo_dict["all"], min(num_shots, len(demo_dict["all"]))
        )
        demos = all_demos
    return demos


def _get_kmeans_demos(
    demo_dict: dict[str, Any],
    kmeans_dict: dict[str, Any],
    num_shots: int,
    index: int,
    sample_index: int,
    discard_null_samples: bool = False,
) -> list[dict[str, Any]]:
    """Get the demonstrations obtained using the kmeans method."""

    if discard_null_samples:
        non_null_demos = kmeans_demo_retrieval(
            num_shots - 1, demo_dict["non-null"], kmeans_dict["non-null"]
        )
        demos = non_null_demos
    else:
        all_demos = kmeans_demo_retrieval(
            num_shots, demo_dict["all"], kmeans_dict["all"]
        )
        demos = all_demos
    return demos


def _get_sp_kmeans_demos(
    demo_dict: dict[str, Any],
    kmeans_dict: dict[str, Any],
    num_shots: int,
    cluster_id: int,
) -> list[dict[str, Any]]:
    """Get the demonstrations obtained using the specialized kmeans method."""

    all_demos = specialized_kmeans_demo_retrieval(
        num_shots, demo_dict["all"], kmeans_dict["all"], cluster_id
    )
    demos = all_demos
    return demos


def get_demos(
    demo_retrieval: str,
    demo_dict: dict[str, Any],
    num_shots: int,
    index: int,
    sample_index: int,
    discard_null_samples: bool = False,
    cosine_sim_matrix_dict: Optional[dict[str, np.ndarray]] = None,
    kmeans_dict: Optional[dict[str, Any]] = None,
    cluster_id: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Get the demonstrations for a given inference sample based on the demo retrieval method.

    Args:
        demo_retrieval (str): The demo retrieval method. One of ["kmeans_knn", "knn", "kmeans", "specialized_kmeans", "random"].
        demo_dict (dict[str, Any]): The demonstrations dictionary with keys "non-null", "null", "all".
        num_shots (int): The number of demonstrations to retrieve.
        index (int): The index of the first inference sample in the batch.
        sample_index (int): The index of the inference sample in the batch.
        discard_null_samples (bool, optional): If True, the null samples will be discarded from retrieval. Defaults to False.
        cosine_sim_matrix_dict (dict[str, np.ndarray]): The cosine similarity matrix for each of the keys in demo_dict.
        kmeans_dict (dict[str, Any]): The KMeans model for each of the keys in demo_dict.
        cluster_id (Optional[int], optional): The id of the cluster for specialized_kmeans retreival. Defaults to None.

    Raises:
        ValueError: If the demo retrieval method is not supported.

    Returns:
        list[dict[str, Any]]: The demonstrations for the given inference sample.
    """

    assert not (
        demo_retrieval == "specialized_kmeans" and cluster_id is None
    ), "cluster_id must be provided for specialized_kmeans retrieval"
    assert not (
        demo_retrieval == "kmeans_knn" and cosine_sim_matrix_dict is None
    ), "cosine_sim_matrix_dict must be provided for kmeans_knn retrieval"
    assert not (
        demo_retrieval in ["kmeans_knn", "kmeans"] and kmeans_dict is None
    ), "kmeans_dict must be provided for kmeans_knn and kmeans retrieval"

    if demo_retrieval == "kmeans_knn":
        demos = _get_kmeans_knn_demos(
            cosine_sim_matrix_dict,
            demo_dict,
            kmeans_dict,
            num_shots,
            index,
            sample_index,
            discard_null_samples,
        )

    elif demo_retrieval == "knn":
        demos = _get_knn_demos(
            cosine_sim_matrix_dict,
            demo_dict,
            num_shots,
            index,
            sample_index,
            discard_null_samples,
        )

    elif demo_retrieval == "kmeans":
        demos = _get_kmeans_demos(
            demo_dict, kmeans_dict, num_shots, index, sample_index, discard_null_samples
        )

    elif demo_retrieval == "specialized_kmeans":
        demos = _get_sp_kmeans_demos(
            demo_dict,
            kmeans_dict,
            num_shots,
            discard_null_samples,
            cluster_id,
        )

    elif demo_retrieval == "random":
        demos = _get_random_demos(demo_dict, num_shots, discard_null_samples)
    else:
        raise ValueError(f"demo retrieval method {demo_retrieval} not supported")
    return demos
