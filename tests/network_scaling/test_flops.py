# Should be evaluated on GPU
# otherwise the transformer FLOPs will be off, because it is not using flash-attention
import hydra
import pytest
from torch.utils.flop_counter import FlopCounterMode

import experiments.logger
from experiments.tagging.experiment import TopTaggingExperiment


@pytest.mark.parametrize(
    "model",
    [
        "tag_tr",
        "tag_lloca",
        "tag_part",
        "tag_slim",
    ],
)
@pytest.mark.parametrize("size", ["xxs", "xs", "s", "m", "l", "xl"])
def test_tagging(model, size, jet_size=50):
    experiments.logger.LOGGER.disabled = True  # turn off logging
    model = f"{model}_{size}"

    # create experiment environment
    with hydra.initialize(config_path="../../config", version_base=None):
        overrides = [
            f"model={model}",
            "save=false",
            "training.batchsize=1",
            "data.dataset=mini",
            "gpus=0",
        ]
        cfg = hydra.compose(config_name="toptagging", overrides=overrides)
        exp = TopTaggingExperiment(cfg)
    exp._init()
    exp.init_physics()
    exp.init_model()
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()

    num_parameters = sum(p.numel() for p in exp.model.parameters())
    flops = 0
    
    iterator = iter(exp.train_loader)
    data = next(iterator)
    while data.x.shape[0] < jet_size:
        data = next(iterator)
    data.x = data.x[:jet_size]
    data.scalars = data.scalars[:jet_size]
    data.batch = data.batch[:jet_size]
    data.ptr[-1] = jet_size

    with FlopCounterMode(display=False) as flop_counter:
        exp._get_ypred_and_label(data)
    flops = flop_counter.get_total_flops()

    print(
        f"flops(batchsize=1)={flops:.2e}; parameters={num_parameters}",
        model,
    )
    # print(flop_counter.get_table(depth=5))
