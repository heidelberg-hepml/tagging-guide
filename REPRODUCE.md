
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
pip install -r requirements_nodeps.txt --no-deps
```

The repo already contains 'mini' versions of all datasets in the `data/` folder, allowing to run everything without downloading big datasets. Use the (default) `config_quick` folder to use them. Typical commands are
```bash
python run.py -cn toptagging save=false
python run.py -cn jetclass save=false
python run.py -cn toptagxl save=false
```

The code supports multi-GPU and multi-node runs using `torchrun`. We currently do not use this widely, but it should be correctly implemented. For instance, the syntax for running on 1 node with 4 GPUs, or 2 nodes with 4 GPUs each is
```bash
# single-node
torchrun --nproc-per-node=4 run.py -cn toptagging save=false

# multi-node (using slurm)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
srun --cpu-bind=none torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc-per-node=gpu \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  --rdzv-id=$SLURM_JOB_ID \
  run.py -cn toptagging save=false
```

### 2) Collect datasets

To download the full datasets, do this:

- Top-tagging: `python data/collect_data.py toptagging`
- JetClass: Download 170GB dataset from https://zenodo.org/records/6619768; update the path in `config/jctagging.yaml` `data.data_dir`.
- TopTagXL: Download 200GB dataset from https://zenodo.org/records/10878355; update the path in `config/toptagxl.yaml` `data.data_dir`.

### 3) Train baseline networks

Our baseline networks are defined in `config/model/`. To train them on the different datasets, use the commands below. These are single-epoch trainings on JetClass which are our baseline, well-understood results for multi-epoch trainings and other datasets are on their way.

```bash
python run.py -cp config -cn jetclass model=tr training=jc_1epoch model.net.size=-2
python run.py -cp config -cn jetclass model=tr training=jc_1epoch model.net.size=-1
python run.py -cp config -cn jetclass model=tr training=jc_1epoch model.net.size=0
python run.py -cp config -cn jetclass model=tr training=jc_1epoch model.net.size=1
python run.py -cp config -cn jetclass model=tr training=jc_1epoch model.net.size=2

# repeat for other architectures (also larger sizes...)
python run.py -cp config -cn jetclass model=lloca training=jc_1epoch model.net.size=-2
python run.py -cp config -cn jetclass model=slim training=jc_1epoch model.net.size=-2
python run.py -cp config -cn jetclass model=part training=jc_1epoch model.net.size=-2
python run.py -cp config -cn jetclass model=gn3 training=jc_1epoch model.net.size=-2

# repeat for other datasets
python run.py -cp config -cn toptagxl model=tr training=jc_1epoch model.net.size=-2  # repeat across models and sizes like above
```

Comments:

- The above commands load the optimized default parameters. See the files in `config/` for more options.
- By default, the `tr`, `lloca` and `slim` networks represent jets as sparse objects to avoid zero-padding. This reduces memory usage (2-3x) and yields significant speedups (1.5-2x) for for large networks, but for small networks dense representations can be faster. The key `model.zeropad` controls this.
- `model.net.size` supports continuous values. However, this currently requires `model.zeropad=true` for `tr`, `lloca` and `slim` because sparse attention kernels have constraints on the embedding shape.
- Use `data.train_files_range` and `data.fraction_of_file` to control the amount of training data.
- The code supports tracking with `mlflow`, which requires `pip install mlflow` (not just `mlflow-skinny` which is a placeholder) and setting `use_mlflow=true`.

We collect results for these trainings in `scaling/*.json` files, which are the basis for the scaling law fits described in section 5. These `.json` files can be created automatically based on a set of runs using the command, after modifying the `PATTERN`, `OUTPUT`, `KEYS` variables in the python script
```bash
python scaling/parse_runs.py
```

### 4) Computational cost estimates

The `cost_estimate/*.json` files contain measures of the computational cost of our baseline networks. They were executed on the Heidelberg ITP H100 nodes. To get the corresponding numbers for your environment (CPUs/GPUs), run these commands:

```bash
python basics.py  # uses GPU
python inference_gpu.py  # uses GPU
python train_gpu.py  # uses GPU

python inference_gpu.py  # uses CPU
```

Additionally, the following command runs our hard-coded energy cost models. In contrast to the scripts above, this script does not create any trial experiments and runs neural networks. It just evaluates hard-coded equations.

```bash
python energy_model.py  # no network calls
```

### 5) Scaling law fits

Finally, to create scaling plots as a function of the computational cost metrics and network performance metrics created above, run the following command. This command loads the entries of the `.json` files `cost_estimate/*.json` and `scaling/*.json`, fits scaling laws, and creates plots. Note that we only recreate the most recent set of scaling laws, but keep old results to be on the safe side.

```bash
python scaling/scan_cost.py
python scaling/scan_perf.py
```
