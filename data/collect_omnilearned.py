from argparse import ArgumentParser
from pathlib import Path

from experiments.tagging.omniloader import download_h5_files, get_url

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", default="top", help="Dataset name to download")
    parser.add_argument("-f", "--folder", default="data", help="Folder to save the dataset")
    args = parser.parse_args()

    for tag in ("train", "test", "val"):
        url = get_url(args.dataset, tag)
        if url is None:
            raise ValueError(f"No download URL for {args.dataset}/{tag}")
        download_h5_files(url, Path(args.folder) / args.dataset / tag)
