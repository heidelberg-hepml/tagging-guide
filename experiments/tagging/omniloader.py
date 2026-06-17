"""h5 streaming dataset for OmniLearned shards; adapted from upstream omnilearned/dataloader.py."""

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
from torch.utils.data import IterableDataset, get_worker_info

from experiments.logger import LOGGER

# List of per-class label shifts in the stored data.
# For finetuning, the label shift should be applied to move labels back to the range [0, n].
_LABEL_SHIFT = {
    "top": 0,  # raw pid: 0, 1
    "h1": 0,  # raw pid: 0, 1
    "atlas": 0,  # raw pid: 2, 10
    "jetclass": 2,  # raw pid: 0, 2..11 (0 only in train)
    "jetclass2": 12,  # raw pid: 12..199
    "aspen": 200,  # raw pid: 200
    "cms_qcd": 201,  # raw pid: 201 (test shard mislabeled as 0)
    "cms_bsm": 202,  # raw pid: 202..209
}

# Pretrain uses shift=0, so raw pid is the label. atlas {2,10} would collide with jetclass;
# +208 moves it to {210,218}, giving 219 pretrain classes (211..217 empty).
_PRETRAIN_LABEL_OFFSET = {"atlas": 208}

_SUPPORTED_DATASETS = frozenset(
    {"top", "pretrain", "atlas", "aspen", "jetclass", "jetclass2", "h1", "cms_qcd", "cms_bsm"}
)

_PRETRAIN_SOURCES = ["atlas", "aspen", "jetclass", "jetclass2", "h1", "cms_qcd", "cms_bsm"]

# Streaming I/O defaults, overridable via cfg.data.slab_events / cfg.data.buffer_mb.
SLAB_EVENTS = 512
BUFFER_BYTES = 256 * 1024**2

# Fixed seed for fractional subsampling: pins the subset across runs and across ranks.
_SUBSAMPLE_SEED = 0


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


class HEPIterableDataset(IterableDataset):
    """Streams events as contiguous slabs from many shards into a per-worker shuffle
    buffer, so single-class shards get mixed within each batch. Slabs are striped across
    ranks then workers and reshuffled per epoch via set_epoch.
    """

    def __init__(
        self,
        file_paths,
        file_counts,
        label_shift=0,
        label_offsets=None,
        rank=0,
        world_size=1,
        shuffle=True,
        seed=0,
        fraction=1.0,
        slab_events=SLAB_EVENTS,
        buffer_bytes=BUFFER_BYTES,
        event_nbytes=None,
    ):
        self.file_paths = list(file_paths)
        self.label_shift = label_shift
        # Per-file additive label offset (0 for every file outside pretrain remapping).
        self.label_offsets = (
            np.zeros(len(self.file_paths), dtype=np.int64)
            if label_offsets is None
            else np.asarray(label_offsets, dtype=np.int64)
        )
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        if event_nbytes is None:
            with h5py.File(self.file_paths[0], "r") as f:
                d = f["data"]
                event_nbytes = int(np.prod(d.shape[1:])) * d.dtype.itemsize
        self.buffer_slabs = max(1, int(buffer_bytes // (slab_events * event_nbytes)))

        slabs = []
        for file_idx, count in enumerate(file_counts):
            file_slabs = [
                (file_idx, lo, min(lo + slab_events, count)) for lo in range(0, count, slab_events)
            ]
            if fraction < 1.0:
                # Random (not leading) subset keeps class-clustered shards representative; fixed seed → identical across ranks.
                subsample_rng = np.random.default_rng([_SUBSAMPLE_SEED, file_idx])
                n_keep = max(1, int(np.ceil(fraction * len(file_slabs))))
                kept = sorted(subsample_rng.choice(len(file_slabs), size=n_keep, replace=False))
                file_slabs = [file_slabs[i] for i in kept]
            slabs.extend(file_slabs)
        all_slabs = np.array(slabs, dtype=np.int64).reshape(-1, 3)
        self.rank_slabs = all_slabs[rank::world_size]
        assert len(self.rank_slabs) > 0, (
            f"rank {rank}/{world_size} got 0 of {len(all_slabs)} slabs from "
            f"{len(self.file_paths)} files (fraction={fraction}); lower world_size "
            f"or raise fraction"
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return int((self.rank_slabs[:, 2] - self.rank_slabs[:, 1]).sum())

    def __iter__(self):
        info = get_worker_info()
        worker_id, num_workers = (0, 1) if info is None else (info.id, info.num_workers)
        worker_slabs = self.rank_slabs[worker_id::num_workers]

        rng = None
        if self.shuffle:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch, self.rank, worker_id])
            )
            worker_slabs = worker_slabs.copy()
            rng.shuffle(worker_slabs)

        for start in range(0, len(worker_slabs), self.buffer_slabs):
            data_slabs, pid_slabs = [], []
            for file_idx, lo, hi in worker_slabs[start : start + self.buffer_slabs]:
                with h5py.File(self.file_paths[int(file_idx)], "r") as f:
                    data = f["data"][int(lo) : int(hi)]
                    pid = f["pid"][int(lo) : int(hi)]
                # Drop fully zero-padded events (no particle with nonzero log pT in column 2).
                real = (data[..., 2] != 0).any(axis=1)
                if not real.all():
                    data, pid = data[real], pid[real]
                if len(data):
                    data_slabs.append(data)
                    pid_slabs.append(pid + self.label_offsets[int(file_idx)])
            if not data_slabs:
                continue
            buffer_data = np.concatenate(data_slabs)
            buffer_labels = np.concatenate(pid_slabs) - self.label_shift
            # Skip events whose shifted label is negative (e.g. stray jetclass pid=0 under shift=2).
            order = np.flatnonzero(buffer_labels >= 0)
            if rng is not None:
                rng.shuffle(order)
            for i in order:
                yield {
                    "X": torch.from_numpy(buffer_data[i].copy()),
                    "y": torch.tensor(int(buffer_labels[i]), dtype=torch.int64),
                }


def _file_event_counts(dataset_path, h5_files, name):
    """Events per file, cached to <dataset_path>/file_counts.npy (atomic write)."""
    counts_file = dataset_path / "file_counts.npy"
    if counts_file.is_file():
        cached = np.load(counts_file)
        if len(cached) == len(h5_files):
            return [int(c) for c in cached]
    LOGGER.info(f"Counting events per file for dataset {name}")
    counts = []
    for h5_path in h5_files:
        try:
            with h5py.File(h5_path, "r") as f:
                counts.append(int(f["data"].shape[0]))
        except Exception as e:
            raise RuntimeError(f"Failed to read {h5_path} (corrupted h5?)") from e
    arr = np.asarray(counts, dtype=np.int64)
    tmp = counts_file.with_name(counts_file.name + ".tmp")
    with open(tmp, "wb") as fp:
        np.save(fp, arr)
    # Atomic rename so racing ranks can't read a torn file.
    os.replace(tmp, counts_file)
    LOGGER.info(f"Number of events: {int(arr.sum())} across {len(h5_files)} files")
    return counts


def load_data(
    dataset_name,
    path,
    dataset_type="train",
    rank=0,
    size=1,
    shuffle=True,
    fraction=1.0,
    seed=0,
    slab_events=SLAB_EVENTS,
    buffer_bytes=BUFFER_BYTES,
):
    """Build a streaming HEPIterableDataset over OmniLearned shards, partitioned per DDP rank.

    `fraction` (in (0, 1]) keeps a fixed random fraction of each shard's slabs.
    """
    if dataset_name not in _SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. Choose from {sorted(_SUPPORTED_DATASETS)}."
        )
    assert 0.0 < fraction <= 1.0, f"fraction must be in (0, 1], got {fraction}"

    names = _PRETRAIN_SOURCES if dataset_name == "pretrain" else [dataset_name]
    dataset_paths = [Path(path) / name / dataset_type for name in names]

    file_paths, file_counts, file_offsets = [], [], []
    event_nbytes = 0
    for name, dataset_path in zip(names, dataset_paths, strict=True):
        if not dataset_path.is_dir() or not any(dataset_path.iterdir()):
            raise FileNotFoundError(
                f"No data in {dataset_path}; run data/collect_omnilearned.py to populate it"
            )
        h5_files = sorted(
            [*dataset_path.glob("*.h5"), *dataset_path.glob("*.hdf5")], key=lambda p: p.name
        )
        # Largest per-event width across sources, so the buffer never overshoots its RAM target.
        with h5py.File(h5_files[0], "r") as f:
            d = f["data"]
            event_nbytes = max(event_nbytes, int(np.prod(d.shape[1:])) * d.dtype.itemsize)
        file_paths.extend(str(p) for p in h5_files)
        file_counts.extend(_file_event_counts(dataset_path, h5_files, name))
        offset = _PRETRAIN_LABEL_OFFSET.get(name, 0) if dataset_name == "pretrain" else 0
        file_offsets.extend([offset] * len(h5_files))

    return HEPIterableDataset(
        file_paths=file_paths,
        file_counts=file_counts,
        label_shift=_LABEL_SHIFT.get(dataset_name, 0),
        label_offsets=file_offsets,
        rank=rank,
        world_size=size,
        shuffle=shuffle,
        seed=seed,
        fraction=fraction,
        slab_events=slab_events,
        buffer_bytes=buffer_bytes,
        event_nbytes=event_nbytes,
    )
