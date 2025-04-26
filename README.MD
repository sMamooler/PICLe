# PICLe: Pseudo-Annotations for In-Context Learning in Low-Resource Named Entity Detection

[![Paper](https://img.shields.io/badge/Paper-arXiv%20Preprint-b31b1b)](https://arxiv.org/abs/2412.11923)
[![Video](https://img.shields.io/badge/Video-Presentation-red)](https://www.youtube.com/watch?v=Ma6wkgvXWU0)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-blue)](https://huggingface.co/datasets/smamooler/PICLe)


## Overview
In this paper we analyze what matters in in-context entity detection, then propose a framework to effectively leverage pseudo-annotated in-context examples.


## TODOs

- [ ] **Ablation Studies**  
  Add support for running ablations:
  - Different models
  - Different merging strategies
  - With and without self-verification

- [ ] **Code Cleanup**  
  Resolve all existing `TODO` comments and ensure codebase consistency.


## Setup
### Environment
```bash
conda create -n picle python=3.9 
conda activate picle
pip install -r requirements.txt
```
### HuggingFace and OpenAI API keys
```bash
export HF_API_KEY=<YOUR_HF_API_KEY>
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
```

## Dataset preparation (WIP)
Instructions coming soon.

## Experiments

### PICLe: In-context NED with pseudo-annotated demonstrations
#### Pseudo-annotation and self-verification
To pseudo-annotate a dataset and run self-verification, run the command below:

```bash
python picle_annotation.py data=<your_dataset_name>
```

This script will create a dataset in your data folder that contains the pseudo-annotations for the dataset you specified.

#### Inferences and self-verification
To run inference with specialized-kmeans, run the command below for each cluster:

```bash
python picle_inference.py data=<your_dataset_name> demonstration_retrieval.cluster_id=<cluster_id>
```

Once the results are ready for all clusters, merge them and run self verification over them with the command below:

```bash
python picle_self_verification.py data=<your_dataset_name>
```

**Note** that this script will consider the **latest** run of the inference script for each cluster.

To plot the results use the [picle](picle.ipynb) notebook.

### Input-output correspondence of in-context demonstrations
To run the experiments with random corrupted demonstrations for all datasets run the command below:

```bash
bash scripts/analysis_random_corrupted_demos.sh
```

To aggregate the results of these experiments and plot them use the [analysis_random_corrupted_demos](analysis_random_corrupted_demos.ipynb) notebook.


### Partially Correct ICL demonstrations
To run the experiments with partially correct demonstrations for all datasets run the command below:

```bash
bash scripts/analysis_partially_correct_demos.sh
```

To aggregate the results of these experiments and plot them use the [analysis_partially_correct_demos](analysis_partially_correct_demos.ipynb) notebook.

### Zero-shot NED
```bash 
python incontext_ned.py --config-name=zero_shot_ned data=<dataset_name>
```

### Few-shot NED
```bash
python incontext_ned.py --config-name=ten_shot_ned data=<dataset_name>
```


## Acknowledgements
- [vLLM](https://github.com/vllm-project/vllm) library for providing an efficient and flexible serving and inference framework for large language models.
- [HuggingFace](https://huggingface.co) for offering a comprehensive ecosystem of models, datasets, and tools that significantly facilitated model development and evaluation.
- Datasets:
  - [bigbio/blurb](https://huggingface.co/datasets/bigbio/blurb) for bc5chem, bc5diseas, and bc2gm datasets 
  - [bigbio/chemprot](https://huggingface.co/datasets/bigbio/chemprot) for chemprot-chem and chemprot-gene datasets

Special thanks to the maintainers and contributors of these open-source projects and datasets for making their work publicly available and well-documented.

## Citation
If you use PICLe in your research, please cite our paper:

```bibtex
@article{mamooler2024picle,
  title={PICLe: Pseudo-Annotations for In-Context Learning in Low-Resource Named Entity Detection},
  author={Mamooler, Sepideh and Montariol, Syrielle and Mathis, Alexander and Bosselut, Antoine},
  journal={arXiv preprint arXiv:2412.11923},
  year={2024}
}
```