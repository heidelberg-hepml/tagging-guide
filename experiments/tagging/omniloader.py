"""h5 Dataset/DataLoader for OmniLearned shards; adapted from upstream omnilearned/dataloader.py."""

import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urljoin

import h5py
import numpy as np
import requests
import torch
from torch.utils.data import DataLoader, Dataset

from experiments.logger import LOGGER

# Per-source label shifts (mirror upstream omnilearned). Comments show post-shift
# classes verified against data/omnilearned/<src>/train; pretrain mode uses shift=0.
_LABEL_SHIFT = {
    "top": 0,  # classes: 0, 1
    "h1": 0,  # classes: 0, 1
    "atlas": 0,  # classes: 2, 10  (no shift; shares jetclass {qcd=2, top=10})
    "jetclass": 2,  # classes: 0..9
    "jetclass2": 12,  # classes: 0..187
    "aspen": 200,  # classes: 0
    "cms_qcd": 201,  # classes: 0
    "cms_bsm": 202,  # classes: 0..7
}

_SUPPORTED_DATASETS = frozenset(
    {"top", "pretrain", "atlas", "aspen", "jetclass", "jetclass2", "h1", "cms_qcd", "cms_bsm"}
)

_PRETRAIN_SOURCES = ["atlas", "aspen", "jetclass", "jetclass2", "h1", "cms_qcd", "cms_bsm"]


def collate_point_cloud(batch):
    """Stack per-sample X and y into a batch dict."""
    return {
        "X": torch.stack([item["X"] for item in batch]),
        "y": torch.stack([item["y"] for item in batch]),
    }


def get_url(
    dataset_name,
    dataset_type,
    base_url="https://portal.nersc.gov/cfs/dasrepo/omnilearned/",
):
    """Return the NERSC portal URL for the given dataset/split, or None on failure."""
    url = f"{base_url}/{dataset_name}/{dataset_type}/"
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        response.raise_for_status()
        return url
    except requests.RequestException as e:
        LOGGER.error(
            f"HEAD {url} failed ({e}); see https://www.nersc.gov/users/status for portal status"
        )
        return None


def download_h5_files(base_url, destination_folder):
    """Download all .h5 files from the given directory URL using aria2c."""
    response = requests.get(base_url, timeout=60)
    response.raise_for_status()

    resolved_base_url = response.url
    if not resolved_base_url.endswith("/"):
        resolved_base_url += "/"

    file_links = re.findall(r'href="([^"]+\.h5)"', response.text)
    if not file_links:
        LOGGER.warning(f"No .h5 files found at {resolved_base_url}")
        return

    aria2c = shutil.which("aria2c")
    if aria2c is None:
        raise RuntimeError("aria2c not found in PATH. Please install or load aria2 before running.")

    destination_folder = Path(destination_folder).resolve()
    destination_folder.mkdir(parents=True, exist_ok=True)

    url_list_file = destination_folder / "download_urls.txt"
    with open(url_list_file, "w") as f:
        for file_name in file_links:
            f.write(urljoin(resolved_base_url, file_name) + "\n")

    cmd = [
        aria2c,
        "--input-file",
        str(url_list_file),
        "--dir",
        str(destination_folder),
        "--continue=true",
        "--max-concurrent-downloads=8",
        "--split=1",
        "--max-connection-per-server=1",
    ]
    LOGGER.info(f"Starting aria2c download from {resolved_base_url}")
    LOGGER.info(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


class HEPDataset(Dataset):
    """Per-event h5 reads with cached file handles; fork-safe via _hepdataset_worker_init."""

    def __init__(self, file_paths, file_indices, label_shift=0):
        self.file_paths = file_paths
        self.file_indices = file_indices
        self.label_shift = label_shift
        self._file_cache = {}

    def __len__(self):
        return len(self.file_indices)

    def _get_file(self, file_idx):
        if file_idx not in self._file_cache:
            self._file_cache[file_idx] = h5py.File(self.file_paths[file_idx], "r")
        return self._file_cache[file_idx]

    def __getitem__(self, idx):
        file_idx, sample_idx = self.file_indices[idx]
        f = self._get_file(file_idx)
        return {
            "X": torch.tensor(f["data"][sample_idx], dtype=torch.float32),
            "y": torch.tensor(f["pid"][sample_idx] - self.label_shift, dtype=torch.int64),
        }


def _hepdataset_worker_init(worker_id):
    """Reset cached h5py handles per worker (fork-inherited handles are unsafe to reuse)."""
    info = torch.utils.data.get_worker_info()
    if info is not None and isinstance(info.dataset, HEPDataset):
        info.dataset._file_cache = {}


def load_data(
    dataset_name,
    path,
    batch=100,
    dataset_type="train",
    num_workers=16,
    prefetch_factor=None,
    rank=0,
    size=1,
    shuffle=True,
):
    """Build a DataLoader over OmniLearned shards, partitioned per DDP rank."""
    if dataset_name not in _SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. Choose from {sorted(_SUPPORTED_DATASETS)}."
        )

    names = _PRETRAIN_SOURCES if dataset_name == "pretrain" else [dataset_name]
    dataset_paths = [Path(path) / name / dataset_type for name in names]

    file_list = []
    file_indices = []
    index_shift = 0
    for iname, dataset_path in enumerate(dataset_paths):
        if not dataset_path.is_dir() or not any(dataset_path.iterdir()):
            raise FileNotFoundError(
                f"No data in {dataset_path}; run data/collect_omnilearned.py to populate it"
            )

        h5_files = list(dataset_path.glob("*.h5")) + list(dataset_path.glob("*.hdf5"))
        file_list.extend(map(str, h5_files))

        index_file = dataset_path / "file_index.npy"
        if not index_file.is_file():
            # Atomic rename so racing ranks can't read a torn file.
            LOGGER.info(f"Creating index list for dataset {names[iname]}")
            local = []
            for file_idx, h5_path in enumerate(h5_files):
                try:
                    with h5py.File(h5_path, "r") as f:
                        num_samples = len(f["data"])
                except Exception as e:
                    raise RuntimeError(f"Failed to read {h5_path} (corrupted h5?)") from e
                local.extend([(file_idx, i) for i in range(num_samples)])
            tmp = index_file.with_suffix(".npy.tmp")
            np.save(tmp, np.array(local, dtype=np.int32))
            os.replace(tmp, index_file)
            LOGGER.info(f"Number of events: {len(local)}")

        all_indices = np.load(index_file, mmap_mode="r")
        if shuffle:
            # Equal counts per rank, else DDP all-reduce hangs.
            n_per_rank = len(all_indices) // size
            indices = all_indices[rank::size][:n_per_rank]
        else:
            # Contiguous slab; uneven counts are fine for eval (gather_concat handles it).
            n = len(all_indices)
            indices = all_indices[n * rank // size : n * (rank + 1) // size]
        file_indices.extend(
            (int(file_idx) + index_shift, int(sample_idx)) for file_idx, sample_idx in indices
        )
        index_shift += len(h5_files)

    data = HEPDataset(file_list, file_indices, label_shift=_LABEL_SHIFT.get(dataset_name, 0))
    loader_kwargs = {
        "batch_size": batch,
        "num_workers": num_workers,
        "collate_fn": collate_point_cloud,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["worker_init_fn"] = _hepdataset_worker_init
        loader_kwargs["persistent_workers"] = True
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(data, **loader_kwargs)
