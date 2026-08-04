<div align="center">

## Virtues and Vices of Equivariant Transformers

[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

<!-- [![arXiv]()](https://arxiv.org/abs/) -->

</div>

This repository contains a systematic study of jet tagging architectures. We provide a unified training pipeline to benchmark modern taggers across datasets, together with scaling fits in network size, training data, and computational cost. We used this repository to train taggers on the JetClass, ATLAS top, and JetSet datasets and we provide mini samples for quick tests. 

## Installation

Clone the repository and install the package with its dependencies

```bash
git clone https://github.com/heidelberg-hepml/tagging-guide.git
cd tagging-guide

python -m venv venv
source venv/bin/activate
pip install -e .
pip install -r requirements.txt
pip install -r requirements_nodeps.txt --no-deps
```

Note that `flash-attn` and `xformers` might require separate installations. The `--no-deps` requirements install core components for the PET and Salt transformers.

## Quickstart

The repository contains `mini` versions of the main datasets used in the paper which can be used to test the code. The default config folder `config_quick` uses them

```bash
python run.py -cn toptagging save=false
python run.py -cn jetclass save=false
python run.py -cn toptagxl save=false
python run.py -cn atlastop save=false
python run.py -cn jetset save=false
python run.py -cn pretrain save=false
```

The full training requires the download of the public full datasets and a switch to the `config` folder with appropriate model, training schedule and network size, for instance

```bash
python run.py -cp config -cn jetclass model=tr training=jc_5epoch model.net.size=0
```

Multi-GPU and multi-node trainings are supported through `torchrun`:

```bash
torchrun --nproc-per-node=4 run.py save=false
```

A complete explanation on how to reproduce the results of the paper is available in [REPRODUCE.md](REPRODUCE.md). This contains useful information on 
- setting up the code and start single/multi GPU trainings
- collecting the full datasets
- using of the computational cost estimations
- reproducing the figures in the paper

## Citation

If you find this code useful in your research, please cite our paper

```bibtex
@article{tagging-guide,
    author = "Luigi Favaro, Tilman Plehn, Huilin Qu, and Jonas Spinner",
    title = "{Virtues and Vices of Equivariant Transformers}",
    eprint = "XXXX.XXXXX",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    year = "2026"
}
```
