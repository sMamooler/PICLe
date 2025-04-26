import os

from omegaconf import DictConfig, OmegaConf


def validate_config(cfg: DictConfig):
    if cfg.demonstration_retrieval.num_shots == 0:
        assert (
            cfg.data.get("demo_data_filename") is None
        ), "If num_shots is 0, the demonstration data filename must be None."
        assert (
            cfg.data.get("demo_data_size") is None
        ), "If num_shots is 0, the demonstration data size must be None."
        assert (
            cfg.demonstration_retrieval.get("method") is None
        ), "If num_shots is 0, the demonstration retrieval method must be 'none'."
        assert (
            cfg.demonstration_retrieval.get("cluster_id") is None
        ), "If num_shots is 0, the demonstration retrieval cluster_id must be None."
        assert (
            cfg.demonstration_retrieval.get("entity_embedding_type") is None
        ), "If num_shots is 0, the demonstration retrieval entity_embedding_type must be None."

    if (
        cfg.demonstration_retrieval.method == "specialized_kmeans"
        and cfg.demonstration_retrieval.cluster_id is None
    ):
        raise ValueError(
            "cluster id must be specified for specialized kmeans retrieval."
        )
    if cfg.demonstration_retrieval.method == "knn":
        assert (
            cfg.demonstration_retrieval.entity_embedding_type is None
        ), "entity embeddings cannot be used for knn retrieval."


def load_experiment_result_config(config_path: str, config_name: str):
    cfg = OmegaConf.load(os.path.join(config_path, config_name + ".yaml"))
    cfg_dict = OmegaConf.to_container(cfg, resolve=False)
    return cfg_dict
