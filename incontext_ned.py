"""Performs named entity detection (NED) with in-context learning."""

# TODO(smamooler): Add merge_annotations.py and clean it up.
# TODO(smamooler): Add random perturbation data prepatation and clean it up.
# TODO(smamooler): Add notebook for plotting random label experiment and clean it up.
# TODO(smamooler): Add notebook for plotting perturbation experiment and clean it up.
# TODO(smamooler): Add notebook for plotting PICLe experiment and clean it up.
# TODO(smamooler): Add notebook for plotting retrieval comparison experiment and clean it up.


import ast
import csv
import json
import logging
import os
import random
import sys

import CONSTANTS as const
import hydra
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from numpyencoder import NumpyEncoder
from omegaconf import DictConfig
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.demo_retrieval import (
    cluster_demos,
    compute_cosin_sim_matrix,
    embed_samples,
    get_demos,
)
from utils.evaluation import (
    entity_list_to_iob_format,
    ner_list_eval,
    post_process_extractions,
)
from utils.llm import call_gpt, call_hf
from utils.perturbation import perturb_annotations
from utils.prompters import get_prompt
from vllm import LLM

log = logging.getLogger(__name__)
csv.field_size_limit(sys.maxsize)


def run_incontext_ned(
    cfg: DictConfig,
    entity_type: str,
    last_index: int,
    output_file: str,
):
    """Performs named entity detection (NED) with in-context learning.

    Args:
        cfg (DictConfig): The configuration file.
        entity_type (str): The type of the entity to detect, e.g. "chemical"
        last_index (int): The index of the last sample in the predictions file.
        output_file (str): The path to the predictions file.
    """

    cost_sum = 0
    random.seed(cfg.seed)

    if "gpt" not in cfg.model:
        tokenizer = AutoTokenizer.from_pretrained(
            const.MODEL_NAME2HUGGINGFACE[cfg.model],
            token=os.getenv("HF_API_TOKEN"),
            revision=cfg.model_revision_hash,
        )
        langauge_model = LLM(
            model=const.MODEL_NAME2HUGGINGFACE[cfg.model],
            trust_remote_code=True,
            seed=cfg.seed,
            dtype="half",
            revision=cfg.model_revision_hash,
            tokenizer_revision=cfg.model_revision_hash,
        )

    # load the demonstration and inference data
    # TODO(@smamooler): if utf-8 does not work the code should handle latin-1 encoding
    inference_dataset = json.load(
        open(
            "/".join(
                [cfg.data.data_dir, cfg.data.dataset, cfg.data.inference_data_filename]
            ),
            encoding="utf-8",
        )
    )
    demo_dataset = json.load(
        open(
            "/".join(
                [cfg.data.data_dir, cfg.data.dataset, cfg.data.demo_data_filename]
            ),
            encoding="utf-8",
        )
    )

    # TODO(@smamooler): can this be safely removed?
    # if random_addition_factor:
    #     random_entity_list = json.load(
    #         open(ENTITY2EXAMPLESFILE[entity_type], encoding="utf-8")
    #     )
    #     demo_dataset = [
    #         add_random_entities(sample, random_addition_factor, random_entity_list)
    #         for sample in demo_dataset
    #     ]

    if cfg.get("data", {}).get("demo_data_size", None):
        demo_dataset_indices = random.sample(
            range(len(demo_dataset)), cfg.data.demo_data_size
        )
        demo_dataset = [demo_dataset[i] for i in demo_dataset_indices]
    else:
        demo_dataset_indices = range(len(demo_dataset))

    dataset_label_space = list(
        set([e for sample in demo_dataset for e in sample["entities"]])
    )
    ## Distinguish between samples with and without entities in them. Needed for perturbation experiments and ensuring that the demonstration data is not without entities.
    null_indices = [
        i for i, _ in enumerate(demo_dataset) if len(demo_dataset[i]["entities"]) == 0
    ]
    non_null_indices = [
        i for i, _ in enumerate(demo_dataset) if len(demo_dataset[i]["entities"]) > 0
    ]
    demo_dict = {
        "non-null": [demo_dataset[i] for i in non_null_indices],
        "null": [demo_dataset[i] for i in null_indices],
        "all": demo_dataset,
    }

    # load the embeddings for the demonstration and inference data if needed
    if cfg.demonstration_retrieval.method in [
        "kmeans",
        "kmeans_knn",
        "knn",
        "specialized_kmeans",
    ]:
        demo_embedding_file = f"{cfg.data.data_dir}/{cfg.data.dataset}/{cfg.data.demo_data_filename.replace('.json', '_embeddings.pt')}"
        if os.path.exists(demo_embedding_file):
            all_demo_embeddings = torch.load(demo_embedding_file)
        else:
            all_demo_embeddings = embed_samples(
                demo_dataset,
                entity_embedding_type=cfg.demonstration_retrieval.entity_embedding_type,
            )
            torch.save(all_demo_embeddings, demo_embedding_file)
        demo_embeddings = all_demo_embeddings[demo_dataset_indices]

        inference_embedding_file = f"{cfg.data.data_dir}/{cfg.data.dataset}/{cfg.data.inference_data_filename.replace('.json', '_embeddings.pt')}"
        if os.path.exists(inference_embedding_file):
            inference_embeddings = torch.load(inference_embedding_file)
        else:
            inference_embeddings = embed_samples(
                inference_dataset, entity_embedding_type=None
            )
            torch.save(inference_embeddings, inference_embedding_file)

        demo_embeddings_dict = {
            "non-null": demo_embeddings[non_null_indices],
            "null": demo_embeddings[null_indices],
            "all": demo_embeddings,
        }

    log.info(f"Number of samples in the demo dataset: {len(demo_dataset)}")
    log.info(f"Number of null samples in the demo dataset: {len(demo_dict['null'])}")
    log.info(
        f"Number of non-null samples in the demo dataset: {len(demo_dict['non-null'])}"
    )

    # process demonstrations according to the demonstration retrieval method
    cosine_sim_matrix_dict = None
    kmeans_dict = None

    if "kmeans" in cfg.demonstration_retrieval.method:
        cluster_file_name = f"{cfg.data.demo_data_filename[:-5]}_cluster.csv"
        if cfg.data.demo_data_size:
            cluster_file_name = cluster_file_name.replace(
                ".csv", f"_size{cfg.data.demo_data_size}_seed{cfg.seed}.csv"
            )
        if cfg.demonstration_retrieval.entity_embedding_type:
            cluster_file_name = cluster_file_name.replace(
                ".csv",
                f"{cfg.demonstration_retrieval.entity_embedding_type}-entity-embedding.csv",
            )
        kmeans_dict, clusters_df = cluster_demos(
            demo_dataset,
            demo_dict,
            demo_embeddings_dict,
            cfg.demonstration_retrieval.num_shots,
        )
        clusters_df.to_csv(
            f"{cfg.data.data_dir}/{cfg.data.dataset}/{cluster_file_name}", index=False
        )

    if "knn" in cfg.demonstration_retrieval.method:
        cosine_sim_matrix_dict = compute_cosin_sim_matrix(
            demo_dict, demo_embeddings_dict, inference_embeddings
        )

    for index in tqdm(range(0, len(inference_dataset), cfg.batch_size)):
        if index <= last_index:
            continue
        batch_samples = inference_dataset[index : index + cfg.batch_size]
        batch_gt_entities = [
            [e.lower() for e in sample["entities"]] for sample in batch_samples
        ]
        batch_demos = []
        batch_messages = []

        batch_demo_prc = []
        batch_demo_rec = []
        batch_demo_num_entities = []
        batch_demo_num_entities_diff = []

        for sample_index, sample in enumerate(batch_samples):
            sample_demo_prc = []
            sample_demo_rec = []
            sample_demo_num_entities = []
            sample_demo_num_entities_diff = []

            peturbation_type = cfg.get("perturbation_type", None).get("name")
            perturbation_factor = cfg.get("perturbation_factor", None).get("value")

            demos = get_demos(
                cfg.demonstration_retrieval.method,
                demo_dict,
                cfg.demonstration_retrieval.num_shots,
                index,
                sample_index,
                peturbation_type is not None,
                cosine_sim_matrix_dict,
                kmeans_dict,
                cfg.get("demonstration_retrieval", {}).get("cluster_id", None),
            )

            if peturbation_type:

                for d_i, d in enumerate(demos):

                    (
                        demos[d_i],
                        demo_prc,
                        demo_rec,
                        num_entities,
                        num_entities_diff,
                    ) = perturb_annotations(
                        d,
                        dataset_label_space,
                        peturbation_type,
                        perturbation_factor,
                    )

                    sample_demo_prc.append(demo_prc)
                    sample_demo_rec.append(demo_rec)
                    sample_demo_num_entities.append(num_entities)
                    sample_demo_num_entities_diff.append(num_entities_diff)

                batch_demo_prc.append(sum(sample_demo_prc) / len(sample_demo_prc))
                batch_demo_rec.append(sum(sample_demo_rec) / len(sample_demo_rec))
                batch_demo_num_entities.append(
                    sum(sample_demo_num_entities) / len(sample_demo_num_entities)
                )
                batch_demo_num_entities_diff.append(
                    sum(sample_demo_num_entities_diff)
                    / len(sample_demo_num_entities_diff)
                )

            batch_demos.append(demos)
            log.debug(f"demos: {demos}")

            messages = get_prompt(
                sample["text"],
                demos,
                entity_type,
                cfg.prompting.method,
                cfg.prompting.use_entity_definition,
                cfg.prompting.num_entity_examples,
            )
            batch_messages.append(messages)

        if "gpt" in cfg.model:
            batch_output, batch_pred_entities, batch_cost = call_gpt(
                cfg.model,
                batch_messages,
                cfg.prompting.method,
                cfg.demonstration_retrieval.num_shots,
                verbose=cfg.verbose,
                temperature=cfg.generation.temperature,
                top_p=cfg.generation.top_p,
                max_tokens=cfg.generation.max_tokens,
            )
        else:
            batch_output, batch_pred_entities, batch_cost = call_hf(
                langauge_model,
                tokenizer,
                batch_messages,
                cfg.prompting.method,
                temperature=cfg.generation.temperature,
                top_p=cfg.generation.top_p,
                max_tokens=cfg.generation.max_tokens,
                seed=cfg.seed,
            )

        batch_output = list(map(lambda x: x.replace("\n", " ### "), batch_output))
        cost_sum += batch_cost

        batch_entry = pd.DataFrame(
            {
                "text": [sample["text"] for sample in batch_samples],
                "prompt": batch_messages,
                "demo_prc": (
                    batch_demo_prc
                    if batch_demo_prc
                    else [[] for _ in range(len(batch_samples))]
                ),
                "demo_rec": (
                    batch_demo_rec
                    if batch_demo_rec
                    else [[] for _ in range(len(batch_samples))]
                ),
                "demo_num_entities": (
                    batch_demo_num_entities
                    if batch_demo_num_entities
                    else [[] for _ in range(len(batch_samples))]
                ),
                "demo_num_entities_diff(pert-gt)": (
                    batch_demo_num_entities_diff
                    if batch_demo_num_entities_diff
                    else [[] for _ in range(len(batch_samples))]
                ),
                "gpt output": batch_output,
                "pred entities": batch_pred_entities,
                "gt entities": batch_gt_entities,
            }
        )

        # update the predictions file
        batch_entry.to_csv(
            output_file, mode="a", header=not os.path.exists(output_file), index=False
        )

    log.info(f"Average cost for GPT: {cost_sum / len(inference_dataset)}")


def evaluate_ned(
    results_dir: str,
    eval_method: str,
    resolve_overlapping_entities: bool,
):
    """Evaluate the NED predictions.

    Args:
        results_dir (str): The directory containing the results.
        eval_method (str): The evaluation method to use. Can be "IOB" or "list".
        resolve_overlapping_entities (bool): Whether to resolve overlapping entities.

    Raises:
        ValueError: If the evaluation method is not supported.
    """
    # load predictions and ground truth
    results_csv_path = results_dir + "/results.csv"
    results_df = pd.read_csv(results_csv_path, encoding="utf-8", engine="python")
    results_df.fillna("[]", inplace=True)

    ### /!\ note that there are some duplicates in the test set of bc2gm, bc5chem and bc5disease datasets
    log.info(f"Raw number of samples: {len(results_df)}")
    results_df.drop_duplicates(subset=["text"], inplace=True)
    log.info(f"Number of samples after removing duplicates: {len(results_df)}")

    ground_truth = results_df["gt entities"].apply(ast.literal_eval).tolist()
    contexts = results_df["text"].tolist()
    predictions = results_df["pred entities"].apply(ast.literal_eval).tolist()
    processed_pred = []

    for i in range(len(ground_truth)):
        ordered_entites = post_process_extractions(
            predictions[i],
            contexts[i],
            treat_special_tokens="bc" in results_csv_path,
            resolve_overlapping_entities=resolve_overlapping_entities,
        )
        processed_pred.append(ordered_entites)
        predictions[i] = ordered_entites

    if resolve_overlapping_entities:
        results_df["processed_pred_wo_overlapping_entities"] = processed_pred
    else:
        results_df["processed_pred"] = processed_pred

    results_df.to_csv(results_csv_path, index=False)

    if eval_method == "IOB":
        eval_path = results_dir + "/ner_iob_evaluation.json"
        ground_truth = [
            entity_list_to_iob_format(contexts[i], ground_truth[i])
            for i in range(len(ground_truth))
        ]
        predictions = [
            entity_list_to_iob_format(contexts[i], processed_pred[i])
            for i in range(len(processed_pred))
        ]
        report = classification_report(
            ground_truth, predictions, scheme=IOB2, output_dict=True
        )
        log.info(report)
        json.dump(
            report,
            open(eval_path, "w"),
            indent=4,
            separators=(", ", ": "),
            ensure_ascii=False,
            cls=NumpyEncoder,
        )

    elif eval_method == "list":
        eval_path = results_dir + "/ner_list_evaluation.json"
        metrics = ner_list_eval(predictions, ground_truth)
        log.info(metrics)
        json.dump(metrics, open(eval_path, "w"), indent=4, ensure_ascii=False)
    else:
        raise ValueError(f"method {eval_method} not supported")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO if cfg.verbose else logging.WARNING)

    output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if (
        cfg.demonstration_retrieval.method == "specialized_kmeans"
        and cfg.demonstration_retrieval.cluster_id is None
    ):
        raise ValueError(
            "cluster id must be specified for specialized kmeans retrieval."
        )

    output_file = f"{output_dir}/results.csv"

    entity_type = const.DATA_ENTITY_DICT[cfg.data.dataset]
    log.info(f"Extracting entities of type: {entity_type}")

    if os.path.exists(output_file):
        predictions_gt_df = pd.read_csv(output_file)
        last_index = predictions_gt_df.index[-1]
        log.info(f"last index: {last_index}")
    else:
        last_index = -1

    run_incontext_ned(cfg, entity_type, last_index, output_file)

    log.info("Evaluating the predictions.")
    evaluate_ned(
        output_dir, cfg.evaluation.method, cfg.evaluation.resolve_overlapping_entities
    )


if __name__ == "__main__":
    main()


# TODO(smamooler): is it safe to remove these?
# def parse_float_list(input_str):
#     try:
#         float_list = [float(item) for item in input_str.split(",")]
#         return float_list
#     except ValueError:
#         raise argparse.ArgumentTypeError("Invalid float values in the list")


# def parse_string_list(input_str):
#     try:
#         string_list = [item for item in input_str.split(",")]
#         return string_list
#     except ValueError:
#         raise argparse.ArgumentTypeError("Invalid string values in the list")
