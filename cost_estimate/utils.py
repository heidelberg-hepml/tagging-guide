import platform
import sys
from datetime import datetime

import cpuinfo
import psutil
import torch

from experiments.tagging.embedding import embed_tagging_data


def get_system_info():
    info = cpuinfo.get_cpu_info()
    system_info = {
        "time": datetime.now().isoformat(),
        "cpu": info.get("brand_raw"),
        "cpu_cores": info.get("cores"),
        "cpu_Hz": info.get("hz_advertized_friendly"),
        "cpu_memory": psutil.virtual_memory().total,
        "cpu_arch": info.get("arch_string_raw"),
        "machine": f"{platform.system()} {platform.release()}",
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        system_info["cuda_version"] = torch.version.cuda
        system_info["cudnn_version"] = (torch.backends.cudnn.version(),)
        props = torch.cuda.get_device_properties(0)
        system_info["gpu_name"] = props.name
        system_info["gpu_memory"] = props.total_memory
    return system_info


def get_rnd_batch(
    cfg_data,
    batchsize=1,
    jet_size=50,
    num_scalars=0,
    device=None,
    dtype=None,
    momentum_dtype=torch.float64,
):
    mass = torch.randn(batchsize, jet_size, 1, device=device, dtype=momentum_dtype).exp()
    p3 = torch.randn(batchsize, jet_size, 3, device=device, dtype=momentum_dtype)
    energy = (mass**2 + p3.norm()).sqrt()
    p4 = torch.cat([energy, p3], dim=-1)
    scalars = torch.randn(batchsize, jet_size, num_scalars, device=device, dtype=dtype)

    embedding = embed_tagging_data(p4, scalars, cfg_data)
    embedding["num_graphs"] = batchsize
    return embedding
