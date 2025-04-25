"""
Runs pseudo-annotation to generate demonstration data for in-context NED (Named Entity Detection),
and saves the output to the path specified by `demo_data_filename` in the config file.

Usage:
    python picle_annotation.py data=your_dataset_name
"""

import ast
import json
import logging
import os

import CONSTANTS as const
import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from incontext_ned import evaluate_ned, run_incontext_ned
from omegaconf import DictConfig
from picle_self_verification import self_verification

log = logging.getLogger(__name__)


@hydra.main(
    config_path="configs/picle_annotation", config_name="config", version_base=None
)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO if cfg.verbose else logging.WARNING)

    output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/results.csv"

    entity_type = const.DATA_ENTITY_DICT[cfg.data.dataset]
    log.info(f"Extracting entities of type: {entity_type}")

    if os.path.exists(output_file):
        predictions_gt_df = pd.read_csv(output_file)
        last_index = predictions_gt_df.index[-1]
        log.info(f"last index: {last_index}")
    else:
        last_index = -1

    pseudo_annotation_file = os.path.join(
        cfg.data.data_dir, cfg.data.dataset, cfg.data.demo_data_filename
    )
    if not os.path.exists(pseudo_annotation_file):
        log.info("Running pseudo-annotation.")

        pseudo_annotation_cfg = cfg.copy()
        pseudo_annotation_cfg.demonstration_retrieval.num_shots = 0
        pseudo_annotation_last_index = -1
        run_incontext_ned(
            pseudo_annotation_cfg,
            entity_type,
            pseudo_annotation_last_index,
            output_file,
        )

        evaluate_ned(
            output_file,
            cfg.evaluation.method,
            cfg.evaluation.resolve_overlapping_entities,
        )

        log.info("Running self-verification.")
        sv_annotation_file_path = output_file.replace(".csv", "_sv.csv")
        annotations = pd.read_csv(output_file)

        self_verification(cfg, annotations, sv_annotation_file_path)

        evaluate_ned(
            sv_annotation_file_path,
            cfg.evaluation.method,
            cfg.evaluation.resolve_overlapping_entities,
        )
        sv_annotation_df = pd.read_csv(sv_annotation_file_path)
        sv_annotation_df["type"] = entity_type
        sv_annotation_df = sv_annotation_df[["id", "type", "text", "pred entities"]]
        sv_annotation_df.rename(columns={"pred entities": "entities"}, inplace=True)
        sv_annotation_df["entities"] = sv_annotation_df["entities"].apply(
            ast.literal_eval
        )

        pseudo_demo_data = sv_annotation_df.to_dict(orient="records")
        with open(pseudo_annotation_file, "w") as f:
            json.dump(pseudo_demo_data, f, indent=4, ensure_ascii=False)

    else:
        log.info(
            f"The pseudo-annotated data already exists at {os.path.join(cfg.data.data_dir, cfg.data.demo_data_filename)}. Skipping this step."
        )


if __name__ == "__main__":
    main()
