<div align="center">

## Virtues and Vices of Equivariant Transformers

[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![arXiv](http://img.shields.io/badge/paper-arxiv.2608.02735-B31B1B.svg)](https://arxiv.org/abs/2608.02735)

</div>

This repository contains a systematic study of jet tagging architectures. We provide a unified training pipeline to benchmark modern taggers across datasets, together with scaling fits in network size, training data, and computational cost. We used this repository to train taggers on the JetClass, ATLAS top, and JetSet datasets and we provide mini samples for quick tests. 

A complete explanation on how to reproduce the results of the paper is available in [REPRODUCE.md](REPRODUCE.md). This contains useful information on 
- setting up the code and start single/multi GPU trainings
- collecting the full datasets
- using of the computational cost estimations
- reproducing the figures in the paper

## Citation

If you find this code useful in your research, please cite our paper

```bibtex
@article{Favaro:2026amj,
    author = "Favaro, Luigi and Plehn, Tilman and Qu, Huilin and Spinner, Jonas",
    title = "{Virtues and Vices of Equivariant Transformers}",
    eprint = "2608.02735",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "IRMP-CP3-26-23",
    month = "8",
    year = "2026"
}
```
