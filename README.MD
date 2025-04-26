# TODOS:

- [ ] add support for ablation studies: different models, different merging strategies, w/ and w/o self-verification
- [ ] write notebook for generating PICLe vs baselines plots
- [ ] resolve all the todos in the code
- [ ] add links to paper, and video

## Setup (WIP)
### Environment
```
conda create -n picle python=3.9 
conda activate picle
pip install -r requirements.txt
```
### HuggingFace and OpenAI API keys
```
export HF_API_KEY=<YOUR_HF_API_KEY>
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
```
## Data preparation (WIP)
Instructions coming soon.

## Experiments

### PICLe: In-context NED with pseudo-annotated demonstrations
#### Pseudo-annotation and self-verification
To pseudo-annotate a dataset and run self-verification, run the command below:

```python picle_annotation.py data=<your_dataset_name>```

This script will create a dataset in your data folder that contains the pseudo-annotations for the dataset you specified.

#### Inferenaces and self-verification
To run inference with specialized-kmeans, run the command below for each cluster:

```python picle_inference.py data=<your_dataset_name> demonstration_retrieval.cluster_id=<cluster_id>```

Once the results are ready for all clusters, merge results from all clusters and run self verification over them with the command below:

```python picle_self_verification.py data=<your_dataset_name>```

**Note** that this script will consider the **latest** run of the inference script for each cluster.

### Input-output correspondence of in-context demonstrations
To run the experiments with random corrupted demonstrations for all datasets run the command below:

```bash scripts/analysis_random_corrupted_demos.sh```

To aggregate the results of these experiments and plot them use the [analysis_random_corrupted_demos](analysis_random_corrupted_demos.ipynb) notebook.

**NOTE** - Reproducibility note: results for `random ood label from text` is not the same as the one in the paper. Investigating this further...
Potential causes:
- incorrect test data (non_null instead of full)
- different random seed
- different model

### Partially Correct ICL demonstrations
To run the experiments with partially correct demonstrations for all datasets run the command below:

```bash scripts/analysis_partially_correct_demos.sh```

To aggregate the results of these experiments and plot them use the [analysis_partially_correct_demos](analysis_partially_correct_demos.ipynb) notebook.

### Zero-shot NED
```python incontext_ned.py --config-name=zero_shot_ned data=<dataset_name>```

### Few-shot NED
```python incontext_ned.py --config-name=ten_shot_ned data=<dataset_name>```


# Acknowledgements (WIP)
- vLLM library
- HuggingFace
- Datasets

# Citation
If you use PICLe in your research, please cite our paper:

```bibtex
@article{mamooler2024picle,
  title={PICLe: Pseudo-Annotations for In-Context Learning in Low-Resource Named Entity Detection},
  author={Mamooler, Sepideh and Montariol, Syrielle and Mathis, Alexander and Bosselut, Antoine},
  journal={arXiv preprint arXiv:2412.11923},
  year={2024}
}
```