# Adapted from https://github.com/ViniciusMikuni/OmniLearned/blob/main/src/omnilearned/dataloader.py

from argparse import ArgumentParser

from experiments.tagging.omniloader import load_data

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        default="top",
        help="Dataset name to download",
    )
    parser.add_argument(
        "-f",
        "--folder",
        default="data",
        help="Folder to save the dataset",
    )
    args = parser.parse_args()

    for tag in ["train", "test", "val"]:
        load_data(args.dataset, args.folder, dataset_type=tag)
