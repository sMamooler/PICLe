"""This module contains functions to retrieve demonstrations for a given inference sample."""

import logging
import os
import random
from typing import Any, Optional

import CONSTANTS as CONSTANTS
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("pritamdeka/S-PubMedBert-MS-MARCO")
model = BertModel.from_pretrained("pritamdeka/S-PubMedBert-MS-MARCO")

log = logging.getLogger(__name__)


def _embedd_texts(samples_texts: list[str], embedding_file: str):
    """Embed the texts using a transformer model and normalize the embeddings such that the mean and std of the embeddings are 0 and 1 respectively."""

    if os.path.exists(embedding_file):
        text_embeddings = torch.load(embedding_file)
    else:
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

        # normalize the embeddings
        text_embeddings = (
            text_embeddings - text_embeddings.mean(dim=0)
        ) / text_embeddings.std(dim=0)

        torch.save(text_embeddings, embedding_file)

    return text_embeddings


def _embedd_entities(
    samples_entities: list[str], entity_embedding_type: str, embedding_file: str
):
    """Embed the entities using a transformer model and normalize the embeddings such that the mean and std of the embeddings are 0 and 1 respectively.
    Supported types of entity embeddings:
    1. sum: sum the embeddings of the entities of the sample
    2. avg: average the embeddings of the entities of the sample
    3. combine: concatenate the entities of the sample and embed the concatenated string
    """
    for sample_index, sample_entities in enumerate(samples_entities):
        if len(sample_entities) == 0:
            samples_entities[sample_index] = [
                "There are no entities mentioned in this text."
            ]

    if entity_embedding_type in ["sum", "avg"]:
        label_space = set()
        for sample_entities in samples_entities:
            for entity in sample_entities:
                label_space.add(entity)
        label2id = {label: i for i, label in enumerate(label_space)}

        if os.path.exists(embedding_file):
            label_space_embeddings = torch.load(embedding_file)
        else:
            label_space_tokenized = tokenizer(
                list(label_space),
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            with torch.no_grad():
                label_space_embeddings = model(
                    **label_space_tokenized, output_hidden_states=True, return_dict=True
                ).pooler_output
                torch.save(label_space_embeddings, embedding_file)

        samples_entities_embeddings = torch.Tensor([])
        for sample_entities in samples_entities:
            sample_entities_embedding = torch.stack(
                [label_space_embeddings[label2id[entity]] for entity in sample_entities]
            )
            if entity_embedding_type == "sum":
                sample_entities_embedding = sample_entities_embedding.sum(dim=0)
            elif entity_embedding_type == "avg":
                sample_entities_embedding = sample_entities_embedding.mean(dim=0)
            samples_entities_embeddings = torch.cat(
                (samples_entities_embeddings, sample_entities_embedding.unsqueeze(0)),
                dim=0,
            )

        # normalize the embeddings
        samples_entities_embeddings = (
            samples_entities_embeddings - samples_entities_embeddings.mean(dim=0)
        ) / samples_entities_embeddings.std(dim=0)

    elif entity_embedding_type == "combine":
        samples_combined_entities = [
            ", ".join(sample_entities) for sample_entities in samples_entities
        ]
        samples_entities_embeddings = _embedd_texts(
            samples_combined_entities, embedding_file
        )
    else:
        raise ValueError(
            f"entity_embedding_type {entity_embedding_type} not supported. Supported types are ['sum', 'avg', 'combine']"
        )

    return samples_entities_embeddings


def get_samples_embeddings(
    cfg: DictConfig,
    samples: list,
    entity_embedding_type: Optional[str] = None,
    inference_mode: bool = False,
) -> torch.Tensor:
    """Embed the samples using the model.

    Args:
        cfg (DictConfig): The configuration object.
        samples (list): The samples to embed.
        entity_embedding_type (Optional[str], optional): The type of entity embeddings to use. Defaults to None.
        inference_mode (bool, optional): If True, the inference data is embedded otherwise the demonstration data is embedded. Defaults to False.

    Returns:
        torch.Tensor: The embeddings of the samples.
    """
    if inference_mode:
        text_embedding_file = f"{cfg.data.data_dir}/{cfg.data.dataset}/{cfg.data.inference_data_filename.replace('.json', '_text_embeddings.pt')}"
    else:
        text_embedding_file = f"{cfg.data.data_dir}/{cfg.data.dataset}/{cfg.data.demo_data_filename.replace('.json', '_text_embeddings.pt')}"

    text_embeddings = _embedd_texts(
        [sample["text"] for sample in samples], text_embedding_file
    )

    if not entity_embedding_type:
        return text_embeddings
    else:
        if entity_embedding_type in ["sum", "avg"]:
            entity_embedding_file = f"{cfg.data.data_dir}/{cfg.data.dataset}/{cfg.data.demo_data_filename.replace('.json', '_label_space_embeddings.pt')}"
        elif entity_embedding_type == "combine":
            entity_embedding_file = f"{cfg.data.data_dir}/{cfg.data.dataset}/{cfg.data.demo_data_filename.replace('.json', '_combined_entity_embeddings.pt')}"

        entity_embeddings = _embedd_entities(
            [sample["entities"] for sample in samples],
            entity_embedding_type,
            entity_embedding_file,
        )

        embeddings = torch.cat((text_embeddings, entity_embeddings), dim=1)

    return embeddings


def cluster_demos(
    demo_dataset: list[dict[str, Any]],
    demo_embeddings: np.ndarray,
    num_clusters: int,
) -> tuple[KMeans, pd.DataFrame]:
    """Cluster the demonstrations using KMeans and save the cluster information to a file.

    Args:
        demo_dataset (list[dict[str, Any]]): The demonstrations dataset.
        demo_embeddings (np.ndarray): The embeddings of the demonstrations.
        num_clusters (int): The number of clusters.

    Returns:
        tuple[KMeans, pd.DataFrame]: The KMeans model and the cluster information as a dataframe.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(
        demo_embeddings
    )

    cluster_dict = {
        "cluster": kmeans.labels_,
        "text": [sample["text"] for sample in demo_dataset],
        "entities": [sample["entities"] for sample in demo_dataset],
    }
    cluster_df = pd.DataFrame(cluster_dict)
    cluster_df = cluster_df.sort_values(by=["cluster"])

    return kmeans, cluster_df


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
    cosine_sim_matrix,
    demo_pool: list,
    kmeans,
    num_shots: int,
    index: int,
    sample_index: int,
) -> list[dict[str, Any]]:
    """Get the demonstrations obtained using the kmeans_knn method: the nearest neighbor of the inference sample, one from each cluster."""

    cosine_sim_scores = cosine_sim_matrix[index + sample_index]

    demos = []
    for i in range(num_shots):
        indices = np.where(kmeans.labels_ == i)[0]
        cluster_cosine_sim_scores = cosine_sim_scores[indices]
        argmax_cluster_cosine_sim = np.argmax(cluster_cosine_sim_scores)
        index = indices[argmax_cluster_cosine_sim]
        demos.append(demo_pool[index])

    return demos


def _get_knn_demos(
    cosine_sim_matrix,
    demo_pool: list,
    num_shots: int,
    index: int,
    sample_index: int,
) -> list[dict[str, Any]]:
    """Get the demonstrations obtained using the knn method: the k nearest neighbors of the inference sample."""

    cosine_sim_scores = cosine_sim_matrix[index + sample_index]
    args_sorted_cosine_sim = np.argsort(-cosine_sim_scores)

    demos = []
    index = 0
    while len(demos) < num_shots:
        index = args_sorted_cosine_sim[index]
        demos.append(demo_pool[index])
        index += 1

    return demos


def _get_random_demos(
    demo_pool: list,
    num_shots: int,
) -> list[dict[str, Any]]:
    """Get the demonstrations obtained using the random method."""

    demos = random.sample(demo_pool, min(num_shots, len(demo_pool)))
    return demos


def _get_kmeans_demos(
    demo_pool: list,
    kmeans,
    num_shots: int,
) -> list[dict[str, Any]]:
    """Get the demonstrations obtained using the kmeans method: one demonstration per cluster."""

    demos = []
    for i in range(num_shots):
        indices = np.where(kmeans.labels_ == i)[0]
        index = random.choice(indices)
        demos.append(demo_pool[index])

    return demos


def _get_sp_kmeans_demos(
    demo_pool: list,
    kmeans,
    num_shots: int,
    cluster_id: int,
) -> list[dict[str, Any]]:
    """Get the demonstrations obtained using the specialized kmeans method: given a cluster id, it samples at random num_shots demonstrations from that cluster."""

    indices = np.where(kmeans.labels_ == cluster_id)[0]
    if num_shots > len(indices):
        num_shots = min(num_shots, len(indices))
        log.warning(
            f"Not enough samples in cluster {cluster_id}. Reducing num_shots to {num_shots}."
        )

    sampled_indices = random.sample(list(indices), num_shots)
    demos = [demo_pool[index] for index in sampled_indices]
    # demos = []
    # for _ in range(num_shots):
    #     index = random.sample(list(indices), 1)[0]
    #     demos.append(demo_pool[index])

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
        demo_retrieval in ["kmeans_knn", "knn"] and cosine_sim_matrix_dict is None
    ), "cosine_sim_matrix_dict must be provided for kmeans_knn retrieval"
    assert not (
        demo_retrieval in ["kmeans_knn", "kmeans"] and kmeans_dict is None
    ), "kmeans_dict must be provided for kmeans_knn and kmeans retrieval"

    demo_pool = demo_dict["all"] if not discard_null_samples else demo_dict["non-null"]

    if cosine_sim_matrix_dict:
        cosine_sim_matrix = (
            cosine_sim_matrix_dict["all"]
            if not discard_null_samples
            else cosine_sim_matrix_dict["non-null"]
        )
    if kmeans_dict:
        kmeans = (
            kmeans_dict["all"] if not discard_null_samples else kmeans_dict["non-null"]
        )

    if demo_retrieval == "kmeans_knn":
        demos = _get_kmeans_knn_demos(
            cosine_sim_matrix,
            demo_pool,
            kmeans,
            num_shots,
            index,
            sample_index,
        )

    elif demo_retrieval == "knn":
        demos = _get_knn_demos(
            cosine_sim_matrix,
            demo_pool,
            num_shots,
            index,
            sample_index,
        )

    elif demo_retrieval == "kmeans":
        demos = _get_kmeans_demos(demo_pool, kmeans, num_shots)

    elif demo_retrieval == "specialized_kmeans":
        demos = _get_sp_kmeans_demos(
            demo_pool,
            kmeans,
            num_shots,
            cluster_id,
        )

    elif demo_retrieval == "random":
        demos = _get_random_demos(demo_pool, num_shots)
    else:
        raise ValueError(f"demo retrieval method {demo_retrieval} not supported")
    return demos
