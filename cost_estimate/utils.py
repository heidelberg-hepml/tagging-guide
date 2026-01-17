import platform
import sys
from datetime import datetime

import cpuinfo
import psutil
import torch


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
