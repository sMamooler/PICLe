"""
Runs inference stage of PICLe and evaluates the predictions.

Usage:
    python picle_inference.py data=your_dataset_name
"""

import logging
import os

import CONSTANTS as const
import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from incontext_ned import evaluate_ned, run_incontext_ned
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(
    config_path="configs/picle_inference", config_name="config", version_base=None
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

    if not os.path.exists(
        os.path.join(cfg.data.data_dir, cfg.data.dataset, cfg.data.demo_data_filename)
    ):
        raise Exception(
            "The pseudo-annotated data does not exist. Please run the pseudo-annotation step first."
        )

    # NOTE: this is only one round. It should be done for all clusters.
    run_incontext_ned(
        cfg,
        entity_type,
        last_index,
        output_file,
    )

    log.info("Evaluating the predictions.")
    evaluate_ned(
        output_file, cfg.evaluation.method, cfg.evaluation.resolve_overlapping_entities
    )


if __name__ == "__main__":
    main()
