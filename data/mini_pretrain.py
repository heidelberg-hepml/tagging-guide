import sys
from pathlib import Path

import h5py
import numpy as np

SOURCE_ROOT = "omnilearned-data/"
OUTPUT_ROOT = "data/pretrain/"
N_PER_SPLIT = {"train": 1000, "val": 500, "test": 500}
SOURCES = ["top", "atlas", "aspen", "jetclass", "jetclass2", "h1", "cms_qcd", "cms_bsm"]
SPLITS = ["train", "test", "val"]


def stride_indices(n, target):
    """Sorted, unique, monotonic indices in [0, n) spread across the full file."""
    if n <= target:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, target).round().astype(np.int64))


def process_split(src_split, out_dir, n_per_split):
    """Build the mini for one (source, split). Returns (F, max_pid, n_errors) or None."""
    source, split = src_split.parent.name, src_split.name
    shards = sorted(src_split.glob("*.h5"), key=lambda p: p.name)
    if not shards:
        print(f"  SKIP no .h5 shards in {src_split}", file=sys.stderr)
        return None

    per_shard = max(1, n_per_split // len(shards))
    data_chunks, pid_chunks = [], []
    full_unique = np.empty(0, dtype=np.float64)

    for shard in shards:
        with h5py.File(shard, "r") as f:
            if "data" not in f or "pid" not in f:
                raise RuntimeError(f"{shard}: missing keys; got {list(f.keys())}")
            data_ds, pid_ds = f["data"], f["pid"]
            if data_ds.ndim != 3:
                raise RuntimeError(f"{shard}: expected ndim==3, got {data_ds.shape}")
            n_total, F = data_ds.shape[0], data_ds.shape[-1]
            idx = stride_indices(n_total, per_shard)
            data_chunks.append(data_ds[idx])
            pid_chunks.append(pid_ds[idx])
            full_unique = np.union1d(full_unique, np.unique(pid_ds[:]))
            span = (idx[-1] - idx[0]) if len(idx) > 1 else 0
            print(f"  {source}/{split}/{shard.name}: N={n_total} F={F} -> kept={len(idx)} span={span}/{n_total - 1}")

    shapes = {chunk.shape[1:] for chunk in data_chunks}
    if len(shapes) != 1:
        raise RuntimeError(f"{source}/{split}: shards disagree on (P, F): {shapes}")

    data_out = np.concatenate(data_chunks, axis=0)
    pid_out = np.concatenate(pid_chunks, axis=0)

    unique_sample = np.unique(pid_out)
    missing = np.setdiff1d(full_unique, unique_sample)
    n_errors = 0
    if missing.size:
        sample_max, full_max = float(unique_sample.max()), float(full_unique.max())
        level = "ERROR" if sample_max < full_max else "WARN"
        n_errors = 1 if level == "ERROR" else 0
        print(f"  {level} {source}/{split}: dropped pids {missing.tolist()} (sample_max={sample_max} full_max={full_max})", file=sys.stderr)

    # Idempotency: remove stale shards and the omniloader's cached per-file event counts
    # (regenerated shards may have different counts).
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in [*out_dir.glob("*.h5"), *out_dir.glob("*.hdf5"), *out_dir.glob("file_counts.npy")]:
        stale.unlink()

    out_path = out_dir / f"{split}_{source}.h5"
    with h5py.File(out_path, "w") as g:
        g.create_dataset("data", data=data_out, compression="gzip", compression_opts=1)
        g.create_dataset("pid", data=pid_out, compression="gzip", compression_opts=1)

    return data_out.shape[-1], float(unique_sample.max()), n_errors


def main():
    src_root, out_root = Path(SOURCE_ROOT), Path(OUTPUT_ROOT)
    pairs_total = len(SOURCES) * len(SPLITS)
    pairs_done = pairs_skipped = total_errors = 0
    min_F = max_pid_train = None

    for source in SOURCES:
        for split in SPLITS:
            src_split = src_root / source / split
            if not src_split.is_dir():
                print(f"  SKIP missing {src_split}", file=sys.stderr)
                pairs_skipped += 1
                continue
            n_target = N_PER_SPLIT[split]
            print(f"[make_pretrain_mini] {source}/{split} (target={n_target}):")
            result = process_split(src_split, out_root / source / split, n_target)
            if result is None:
                pairs_skipped += 1
                continue
            F, max_pid, n_errors = result
            pairs_done += 1
            total_errors += n_errors
            min_F = F if min_F is None else min(min_F, F)
            if split == "train":
                max_pid_train = max_pid if max_pid_train is None else max(max_pid_train, max_pid)

    extra_scalars = (min_F - 3) if min_F is not None else None
    num_classes = (int(max_pid_train) + 1) if max_pid_train is not None else None
    print(f"[make_pretrain_mini] DONE: {pairs_done}/{pairs_total} pairs, num_classes={num_classes}, min_F={min_F}, extra_scalars={extra_scalars}, skipped={pairs_skipped}, errors={total_errors}")
    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main())
