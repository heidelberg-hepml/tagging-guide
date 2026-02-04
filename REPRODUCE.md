
## Reproducing our results

### 1) Setup environment

```bash
git clone https://github.com/heidelberg-hepml/tagging-guide
cd tagging-guide
```

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

### 2) Collect datasets

```bash
python data/collect_data.py toptagging
```

- Download [JetClass dataset](https://zenodo.org/records/6619768) and put its path in `config/jctagging.yaml` `data.data_dir`.
- Download [TopTagXL dataset](https://zenodo.org/records/10878355) and put its path in `config/toptagxl.yaml` `data.data_dir`.

### 3) Computational cost estimates

The `cost_estimate/*.json` files contain measures of the computational cost of our baseline networks. They were executed on the Heidelberg ITP H100 nodes. To get the corresponding numbers for your environment (CPUs/GPUs), run these commands:

```bash
python basics.py  # uses GPU
python inference_gpu.py  # uses CPU
python inference_gpu.py  # uses GPU
python train_gpu.py  # uses GPU
```

Additionally, the following command runs our hard-coded energy cost models. In contrast to the scripts above, this script does not create any trial experiments and runs neural networks. It just evaluates hard-coded equations.

```bash
python energy_model.py  # no machine-learning
```

### 4) Train baseline networks

Our baseline networks are defined in `config/model/`. To train them on the different datasets, use the commands below.

```bash
python run.py -cp config -cn jctagging model=tr_xs training=default_xs
python run.py -cp config -cn jctagging model=tr_s training=default_s
python run.py -cp config -cn jctagging model=tr_m training=default_m
python run.py -cp config -cn jctagging model=tr_l training=default_l
python run.py -cp config -cn jctagging model=tr_xl training=default_xl
python run.py -cp config -cn jctagging model=tr_xxl training=default_xxl

# repeat for other architectures
python run.py -cp config -cn jctagging model=lloca_xs training=default_xs  # also s,m...
python run.py -cp config -cn jctagging model=part_xs training=default_xs  # also s,m...
python run.py -cp config -cn jctagging model=slim_xs training=default_slim_xs  # also s,m...; NOTE: special training config

# repeat for other datasets
python run.py -cp config -cn toptagxl model=tr_xs training=default_xs  # also s,m... and lloca,part,slim
```

We collect results for these trainings in `results/*.json`.

### 5) Scaling plots

Finally, to create scaling plots as a function of the computational cost metrics and network performance metrics created above, run the following command. This command loads the entries of the `.json` files `cost_estimate/*.json` and `results/*.json`, visualizes them, and fits scaling laws

```bash
python results/evaluate.py
```
