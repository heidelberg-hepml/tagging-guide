import json
import os
import time
from glob import glob

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from experiments.logger import LOGGER
from experiments.tagging.experiment import BinaryTaggingExperiment
from experiments.tagging.miniweaver.dataset import SimpleIterDataset
from experiments.tagging.miniweaver.loader import to_filelist


class ATLASTopExperiment(BinaryTaggingExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_outputs = 1
        self.eval_systs = self.cfg.data.eval_systs
        if self.eval_systs:
            self.systs_set = self.cfg.data.systs_set
            self.syst_set_names = [
                "all",
                "angular",
                "bias",
                "cer",
                "cluster",
                "cpos",
                "dipole",
                "esdown",
                "esup",
                "string",
                "teg",
                "tej",
                "tfj",
                "tfl",
                "ttbar_herwig",
                "ttbar_pythia",
            ]
            assert all(syst in self.syst_set_names for syst in self.systs_set)
            if self.systs_set[0] == "all":
                self.syst_set_names.remove("all")
                self.syst_folders = {syst: f"{syst}" for syst in self.syst_set_names}
                self.syst_datasets = {syst: None for syst in self.syst_set_names}
            else:
                self.syst_folders = {syst: f"{syst}" for syst in self.systs_set}
                self.syst_datasets = {syst: None for syst in self.systs_set}

        if self.cfg.data.features == "default":
            self.extra_scalars = 0
            self.cfg.data.config = {
                "train": "experiments/tagging/miniweaver/configs_atlastop/default.yaml",
                "val": "experiments/tagging/miniweaver/configs_atlastop/default.yaml",
                "test": "experiments/tagging/miniweaver/configs_atlastop/default_noweights.yaml",
                "syst": "experiments/tagging/miniweaver/configs_atlastop/default_noweights.yaml",
                "onlyqcd": "experiments/tagging/miniweaver/configs_atlastop/default_onlyqcd.yaml",
                "onlytop": "experiments/tagging/miniweaver/configs_atlastop/default_onlytop.yaml",
            }
        else:
            raise ValueError(f"Input feature option {self.cfg.data.features} not implemented")

    def init_data(self):
        LOGGER.info("Creating SimpleIterDataset")
        t0 = time.time()

        datasets = {"train": None, "test": None, "val": None}

        for_training = {"train": True, "val": True, "test": False}
        folder = {"train": "train_nominal", "test": "test_nominal", "val": "val_nominal"}
        files_range = {
            "train": self.cfg.data.train_files_range,
            "test": self.cfg.data.test_files_range,
            "val": self.cfg.data.val_files_range,
        }
        self.num_files = {label: frange[1] - frange[0] for label, frange in files_range.items()}
        for label in ["train", "test", "val"]:
            path = os.path.join(self.cfg.data.data_dir, folder[label])
            flist = [
                f"{folder[label]}:{path}/{folder[label]}_{str(i).zfill(3)}.root"
                for i in range(*files_range[label])
            ]
            file_dict, _ = to_filelist(flist)

            LOGGER.info(f"Using {len(flist)} files for {label}ing from {path}")
            fraction_of_file = self.cfg.data.fraction_of_file if label == "train" else 1
            datasets[label] = SimpleIterDataset(
                file_dict,
                self.cfg.data.config[label],
                for_training=for_training[label],
                extra_selection=self.cfg.data.extra_selection,
                remake_weights=not self.cfg.data.not_remake_weights,
                load_range_and_fraction=((0, fraction_of_file), 1, 1),
                file_fraction=1,
                fetch_by_files=self.cfg.data.fetch_by_files,
                fetch_step=self.cfg.data.fetch_step,
                infinity_mode=self.cfg.data.steps_per_epoch is not None,
                in_memory=self.cfg.data.in_memory,
                name=label,
                events_per_file=self.cfg.data.events_per_file,
                async_load=self.cfg.data.async_load,
            )
        self.data_train = datasets["train"]
        self.data_test = datasets["test"]
        self.data_val = datasets["val"]

        if self.eval_systs:
            for syst in self.syst_folders.keys():
                path = os.path.join(self.cfg.data.data_dir, self.syst_folders[syst])
                flist = glob(f"{path}/{self.syst_folders[syst]}_*.root")
                file_dict, _ = to_filelist(flist)

                LOGGER.info(f"Using {len(flist)} files for syst {syst} from {path}")
                self.syst_datasets[syst] = SimpleIterDataset(
                    file_dict,
                    self.cfg.data.config["syst"],
                    for_training=False,
                    extra_selection=self.cfg.data.extra_selection,
                    remake_weights=not self.cfg.data.not_remake_weights,
                    load_range_and_fraction=((0, 1), 1, 1),
                    file_fraction=1,
                    fetch_by_files=self.cfg.data.fetch_by_files,
                    fetch_step=self.cfg.data.fetch_step,
                    infinity_mode=self.cfg.data.steps_per_epoch is not None,
                    in_memory=self.cfg.data.in_memory,
                    name=syst,
                    events_per_file=self.cfg.data.events_per_file,
                    async_load=self.cfg.data.async_load,
                )

            additional_datasets = ["onlyqcd", "onlytop"]
            for label in additional_datasets:
                path = os.path.join(self.cfg.data.data_dir, "test_nominal")
                flist = [
                    f"{label}:{path}/test_nominal_{str(i).zfill(3)}.root"
                    for i in range(*files_range["test"])
                ]
                file_dict, _ = to_filelist(flist)

                LOGGER.info(f"Using {len(flist)} files for dataset {label} from {path}")
                self.syst_datasets[label] = SimpleIterDataset(
                    file_dict,
                    self.cfg.data.config[label],
                    for_training=False,
                    extra_selection=self.cfg.data.extra_selection,
                    remake_weights=not self.cfg.data.not_remake_weights,
                    load_range_and_fraction=((0, 1), 1, 1),
                    file_fraction=1,
                    fetch_by_files=self.cfg.data.fetch_by_files,
                    fetch_step=self.cfg.data.fetch_step,
                    infinity_mode=self.cfg.data.steps_per_epoch is not None,
                    in_memory=self.cfg.data.in_memory,
                    name=label,
                    events_per_file=self.cfg.data.events_per_file,
                    async_load=self.cfg.data.async_load,
                )

        dt = time.time() - t0
        LOGGER.info(f"Finished creating datasets after {dt:.2f} s = {dt / 60:.2f} min")

    def _init_dataloader(self):
        self.loader_kwargs = {
            "pin_memory": True,
            "persistent_workers": self.cfg.data.num_workers > 0
            and self.cfg.data.steps_per_epoch is not None,
        }
        num_workers = {
            label: min(self.cfg.data.num_workers, self.num_files[label])
            for label in ["train", "test", "val"]
        }

        self.train_loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.training.batchsize // self.world_size,
            drop_last=True,
            num_workers=num_workers["train"],
            multiprocessing_context="fork",
            **self.loader_kwargs,
        )
        self.val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            drop_last=True,
            num_workers=num_workers["val"],
            multiprocessing_context="fork",
            **self.loader_kwargs,
        )
        self.test_loader = DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            drop_last=False,
            num_workers=num_workers["test"],
            multiprocessing_context="fork",
            **self.loader_kwargs,
        )

        if self.eval_systs:
            self.syst_loaders = {
                syst: DataLoader(
                    dataset=self.syst_datasets[syst],
                    batch_size=self.cfg.evaluation.batchsize // self.world_size,
                    drop_last=False,
                    num_workers=num_workers["test"],
                    multiprocessing_context="fork",
                    **self.loader_kwargs,
                )
                for syst in self.syst_datasets.keys()
            }

        self.init_standardization()

    def _extract_batch(self, batch):
        fourmomenta = batch[0]["pf_vectors"].transpose(1, 2).to(self.device, self.momentum_dtype)
        weights = batch[0]["ev_weights"].to(self.device, self.momentum_dtype)[..., 0]
        if self.cfg.data.features == "default":
            scalars = torch.empty(
                fourmomenta.shape[0],
                fourmomenta.shape[1],
                0,
                device=fourmomenta.device,
                dtype=self.dtype,
            )
        label = batch[1]["_label_"].to(self.device, self.dtype)
        return fourmomenta, scalars, label, weights

    def evaluate(self):
        super().evaluate()
        if self.eval_systs:
            for syst in self.syst_datasets.keys():
                self.results[syst] = self.evaluate_single_syst(self.syst_loaders[syst], syst)

            # experimental uncertainties
            track_syst_keys = ["tej", "teg", "tfl", "tfj", "bias"]
            cluster_syst_keys = ["esup", "esdown", "cer", "cpos", "teg"]
            if all(key in self.results.keys() for key in track_syst_keys):
                LOGGER.info("### Start to evaluate tracking uncertainties")
                self.calculate_metrics(self.results["tej"], title="tej")
                self.calculate_metrics(self.results["teg"], title="teg")
                self.calculate_metrics(self.results["tfl"], title="tfl")
                self.calculate_metrics(self.results["tfj"], title="tfj")
                self.calculate_metrics(self.results["bias"], title="bias")

            if all(key in self.results.keys() for key in cluster_syst_keys):
                LOGGER.info("### Start to evaluate clustering uncertainties")
                self.calculate_metrics(self.results["esup"], title="esup")
                self.calculate_metrics(self.results["esdown"], title="esdown")
                self.calculate_metrics(self.results["cer"], title="cer")
                self.calculate_metrics(self.results["cpos"], title="cpos")

            # theoretical uncertainties
            if ("angular" in self.results.keys()) and ("dipole" in self.results.keys()):
                LOGGER.info("### Start to evaluate hadronization uncertainties")
                for key in self.results["dipole"].keys():
                    self.results["dipole"][key] = torch.cat(
                        (self.results["dipole"][key], self.results["onlytop"][key]), dim=0
                    )
                    self.results["angular"][key] = torch.cat(
                        (self.results["angular"][key], self.results["onlytop"][key]), dim=0
                    )
                self.calculate_metrics(self.results["dipole"], title="dipole")
                self.calculate_metrics(self.results["angular"], title="angular")

            if ("cluster" in self.results.keys()) and ("string" in self.results.keys()):
                LOGGER.info("### Start to evaluate shower uncertainties")
                for key in self.results["cluster"].keys():
                    self.results["cluster"][key] = torch.cat(
                        (self.results["cluster"][key], self.results["onlytop"][key]), dim=0
                    )
                    self.results["string"][key] = torch.cat(
                        (self.results["string"][key], self.results["onlytop"][key]), dim=0
                    )
                self.calculate_metrics(self.results["cluster"], title="cluster")
                self.calculate_metrics(self.results["string"], title="string")

            if ("ttbar_herwig" in self.results.keys()) and ("ttbar_pythia" in self.results.keys()):
                LOGGER.info("### Start to evaluate signal modeling uncertainties")
                for key in self.results["ttbar_herwig"].keys():
                    self.results["ttbar_herwig"][key] = torch.cat(
                        (self.results["ttbar_herwig"][key], self.results["onlyqcd"][key]), dim=0
                    )
                    self.results["ttbar_pythia"][key] = torch.cat(
                        (self.results["ttbar_pythia"][key], self.results["onlyqcd"][key]), dim=0
                    )
                self.calculate_metrics(self.results["ttbar_herwig"], title="ttbar_herwig")
                self.calculate_metrics(self.results["ttbar_pythia"], title="ttbar_pythia")

    @torch.inference_mode()
    def evaluate_single_syst(self, loader, title):
        LOGGER.info(
            f"### Starting to evaluate model on {title} dataset with "
            f"{len(loader.dataset)} elements, batchsize {loader.batch_size} ###"
        )
        metrics = {}

        # predictions
        labels_true, labels_predict = [], []
        self.model.eval()
        for batch in loader:
            y_pred, label, _, _, _ = self._get_ypred_and_label(batch)
            labels_true.append(label.cpu().float())
            labels_predict.append(y_pred.cpu().float())
        labels_true, labels_predict = torch.cat(labels_true), torch.cat(labels_predict)

        metrics["labels_true"], metrics["labels_predict"] = (
            labels_true,
            labels_predict,
        )
        labels_predict = torch.nn.functional.sigmoid(labels_predict)
        labels_true, labels_predict = labels_true.numpy(), labels_predict.numpy()
        return metrics

    def calculate_metrics(self, metrics, title=None):
        labels_true, labels_predict = metrics["labels_true"], metrics["labels_predict"]
        accuracy = accuracy_score(labels_true, np.round(labels_predict))
        metrics["accuracy"] = accuracy
        LOGGER.info(f"Accuracy on {title} dataset: {accuracy:.6f}")

        # roc (fpr = epsB, tpr = epsS)
        fpr, tpr, th = roc_curve(labels_true, labels_predict)
        metrics["fpr"], metrics["tpr"] = fpr, tpr
        metrics["auc"] = roc_auc_score(labels_true, labels_predict)

        LOGGER.info(f"AUC score on {title} dataset: {metrics['auc']:.6f}")

        # 1/epsB at fixed epsS
        def get_rej(epsS):
            idx = np.argmin(np.abs(tpr - epsS))
            return 1 / fpr[idx]

        metrics["rej03"] = get_rej(0.3)
        metrics["rej05"] = get_rej(0.5)
        metrics["rej08"] = get_rej(0.8)
        LOGGER.info(
            f"Rejection rate {title} dataset: {metrics['rej03']:.0f} (epsS=0.3), "
            f"{metrics['rej05']:.0f} (epsS=0.5), {metrics['rej08']:.0f} (epsS=0.8)"
        )
        LOGGER.info("/-------------------------/")
        if self.cfg.save:
            metrics_json = {
                title: {
                    "accuracy": metrics["accuracy"],
                    "auc": metrics["auc"],
                    "rej03": metrics["rej03"],
                    "rej05": metrics["rej05"],
                    "rej08": metrics["rej08"],
                }
            }
            filename = os.path.join(self.cfg.run_dir, f"results_{title}_{self.cfg.run_idx}.json")
            with open(filename, "w") as file:
                json.dump(metrics_json, file, indent=2)
