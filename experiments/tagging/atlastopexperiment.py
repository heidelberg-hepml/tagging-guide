import json
import os
import time
from glob import glob

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from experiments.distributed import gather_concat, total_size_across_ranks
from experiments.logger import LOGGER
from experiments.tagging.experiment import BinaryTaggingExperiment
from experiments.tagging.miniweaver.dataset import SimpleIterDataset
from experiments.tagging.miniweaver.loader import to_filelist

ATLAS_SYST_NAMES = (
    "angular", "bias", "cer", "cluster", "cpos", "dipole",
    "esdown", "esup", "string", "teg", "tej", "tfj", "tfl",
    "ttbar_herwig", "ttbar_pythia",
)
ATLAS_BKG_ONLY_SYSTS = ("angular", "cluster", "dipole", "string")
ATLAS_SIG_ONLY_SYSTS = ("ttbar_herwig", "ttbar_pythia")


def _concat_into(target, source):
    for key in ("labels_true", "labels_predict"):
        target[key] = np.concatenate([target[key], source[key]], axis=0)


def _compute_atlas_metrics(labels_true, labels_predict):
    fpr, tpr, _ = roc_curve(labels_true, labels_predict)
    assert (tpr > 0.5).any()
    rej05 = 1.0 / fpr[np.argmax(tpr > 0.5)]
    auc = roc_auc_score(labels_true, labels_predict)
    return rej05, auc


def _safe_max(*xs):
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


def _safe_quad(*xs):
    if any(x is None for x in xs):
        return None
    return float(np.sqrt(sum(x**2 for x in xs)))


class ATLASTopExperiment(BinaryTaggingExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_outputs = 1
        self.eval_systs = self.cfg.data.eval_systs
        if self.eval_systs:
            self.systs_set = self.cfg.data.systs_set
            self.syst_set_names = ["all", *ATLAS_SYST_NAMES]
            assert all(syst in self.syst_set_names for syst in self.systs_set)
            if self.systs_set[0] == "all":
                active = ATLAS_SYST_NAMES
            else:
                active = self.systs_set
            self.syst_folders = {syst: f"{syst}" for syst in active}
            self.syst_datasets = {syst: None for syst in active}

        if self.cfg.data.features == "fourmomenta":
            self.extra_scalars = 0
            self.cfg.data.config = {
                "train": "experiments/tagging/miniweaver/configs_atlastop/fourmomenta.yaml",
                "val": "experiments/tagging/miniweaver/configs_atlastop/fourmomenta.yaml",
                "test": "experiments/tagging/miniweaver/configs_atlastop/fourmomenta_noweights.yaml",
                "syst": "experiments/tagging/miniweaver/configs_atlastop/fourmomenta_noweights.yaml",
                "onlyqcd": "experiments/tagging/miniweaver/configs_atlastop/fourmomenta_onlyqcd.yaml",
                "onlytop": "experiments/tagging/miniweaver/configs_atlastop/fourmomenta_onlytop.yaml",
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
        for label, n in self.num_files.items():
            assert n >= self.world_size, (
                f"{label}: {n} files is less than world_size={self.world_size}; "
                "increase the file range or reduce world_size"
            )
        for label in ["train", "test", "val"]:
            path = os.path.join(self.cfg.data.data_dir, folder[label])
            flist = [
                f"{folder[label]}:{path}/{folder[label]}_{str(i).zfill(3)}.root"
                for i in range(*files_range[label])
            ]
            file_dict, _ = to_filelist(flist)
            file_dict = {n: f[self.rank :: self.world_size] for n, f in file_dict.items()}

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
                file_dict = {n: f[self.rank :: self.world_size] for n, f in file_dict.items()}

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
                file_dict = {n: f[self.rank :: self.world_size] for n, f in file_dict.items()}

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
        # cap by per-rank file count: with external rank sharding each rank holds
        # only num_files // world_size files per class
        num_workers = {
            label: min(self.cfg.data.num_workers, self.num_files[label] // self.world_size)
            for label in ["train", "test", "val"]
        }

        self.train_loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.training.batchsize // self.world_size,
            drop_last=True,
            num_workers=num_workers["train"],
            **self.loader_kwargs,
        )
        self.val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            drop_last=True,
            num_workers=num_workers["val"],
            **self.loader_kwargs,
        )
        self.test_loader = DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            drop_last=False,
            num_workers=num_workers["test"],
            **self.loader_kwargs,
        )

        if self.eval_systs:
            # ttbar_* are single-file datasets; force num_workers=1 (SimpleIterDataset
            # asserts every worker gets >=1 file).
            self.syst_loaders = {
                syst: DataLoader(
                    dataset=self.syst_datasets[syst],
                    batch_size=self.cfg.evaluation.batchsize // self.world_size,
                    drop_last=False,
                    num_workers=1 if "ttbar" in syst else num_workers["test"],
                    **self.loader_kwargs,
                )
                for syst in self.syst_datasets.keys()
            }

        self._record_train_size()
        self.init_standardization()

    def _extract_batch(self, batch):
        fourmomenta = batch[0]["pf_vectors"].transpose(1, 2).to(self.device, self.momentum_dtype)
        weights = batch[0]["ev_weights"].to(self.device, self.dtype)[..., 0, 0]
        if self.cfg.data.features == "fourmomenta":
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
        if not self.eval_systs:
            return

        self.model.eval()
        for title, loader in self.syst_loaders.items():
            n = len(loader.dataset)
            if isinstance(loader.dataset, torch.utils.data.IterableDataset):
                n = total_size_across_ranks(n, self.device)
            LOGGER.info(
                f"### Starting to evaluate model on {title} dataset with "
                f"{n} elements, batchsize {loader.batch_size * self.world_size} ###"
            )
            labels_true, labels_predict = [], []
            with torch.inference_mode():
                for batch in loader:
                    y_pred, label, _, _, _ = self._get_ypred_and_label(batch)
                    labels_true.append(label.float())
                    labels_predict.append(y_pred.float())
            labels_true = gather_concat(torch.cat(labels_true)).cpu()
            labels_predict = gather_concat(torch.cat(labels_predict)).cpu()
            labels_predict = torch.nn.functional.sigmoid(labels_predict)
            self.results[title] = {
                "labels_true": labels_true.numpy(),
                "labels_predict": labels_predict.numpy(),
            }

        if not self.is_master:
            return

        # alt-sample-only systs are class-incomplete; append nominal opposite-class jets
        for syst in ATLAS_BKG_ONLY_SYSTS:
            if syst in self.results and "onlytop" in self.results:
                _concat_into(self.results[syst], self.results["onlytop"])
        for syst in ATLAS_SIG_ONLY_SYSTS:
            if syst in self.results and "onlyqcd" in self.results:
                _concat_into(self.results[syst], self.results["onlyqcd"])

        if not self.cfg.save:
            return

        assert "test" in self.results, (
            "ATLAS systematics aggregation requires nominal test predictions; "
            "set evaluation.eval_set to include 'test'."
        )
        LOGGER.info("### Computing ATLAS systematics summary")
        test = self.results["test"]
        nom_rej05, nom_auc = _compute_atlas_metrics(
            test["labels_true"], test["labels_predict"]
        )
        nominal = {"rej05": nom_rej05, "auc": nom_auc}

        per_syst = {}
        for syst in ATLAS_SYST_NAMES:
            if syst in self.results:
                r, a = _compute_atlas_metrics(
                    self.results[syst]["labels_true"],
                    self.results[syst]["labels_predict"],
                )
                per_syst[syst] = {"rej05": r, "auc": a}

        metrics_json = {}
        for metric_name in ("rej05", "auc"):
            nom = nominal[metric_name]
            rel_to_nom = {
                s: abs(per_syst[s][metric_name] - nom) / nom for s in per_syst
            }
            ratios = {}
            for a, b in (
                ("ttbar_herwig", "ttbar_pythia"),
                ("dipole", "angular"),
                ("cluster", "string"),
            ):
                if a in per_syst and b in per_syst:
                    ratios[(a, b)] = abs(
                        per_syst[a][metric_name] / per_syst[b][metric_name] - 1
                    )

            leaves = {
                "unc_es": _safe_max(rel_to_nom.get("esup"), rel_to_nom.get("esdown")),
                "unc_cer": rel_to_nom.get("cer"),
                "unc_cpos": rel_to_nom.get("cpos"),
                "unc_eff": _safe_max(rel_to_nom.get("teg"), rel_to_nom.get("tej")),
                "unc_fake": _safe_max(rel_to_nom.get("tfl"), rel_to_nom.get("tfj")),
                "unc_bias": rel_to_nom.get("bias"),
                "unc_sig_model": ratios.get(("ttbar_herwig", "ttbar_pythia")),
                "unc_bkg_ps": ratios.get(("dipole", "angular")),
                "unc_bkg_had": ratios.get(("cluster", "string")),
            }
            groups = {
                "unc_cluster": _safe_quad(
                    leaves["unc_es"], leaves["unc_cer"], leaves["unc_cpos"]
                ),
                "unc_track": _safe_quad(
                    leaves["unc_eff"], leaves["unc_fake"], leaves["unc_bias"]
                ),
                "unc_bkg_model": _safe_quad(
                    leaves["unc_bkg_ps"], leaves["unc_bkg_had"]
                ),
            }
            total = _safe_quad(
                groups["unc_cluster"],
                groups["unc_track"],
                leaves["unc_sig_model"],
                groups["unc_bkg_model"],
            )

            out = {"nominal": nom}
            for s, d in per_syst.items():
                out[s] = d[metric_name]
            out.update({k: v for k, v in leaves.items() if v is not None})
            out.update({k: v for k, v in groups.items() if v is not None})
            if total is not None:
                out["unc_total"] = total
            for k, v in out.items():
                metrics_json[f"{metric_name}_{k}"] = v

        for metric_name in ("rej05", "auc"):
            key = f"{metric_name}_unc_total"
            if key in metrics_json:
                LOGGER.info(f"{key} = {metrics_json[key]:.4f}")

        metrics_json = {k: float(f"{v:.6g}") for k, v in metrics_json.items()}
        metrics_json.update(self.metadata)
        filename = os.path.join(self.cfg.run_dir, f"results_sys_{self.cfg.run_idx}.json")
        with open(filename, "w") as f:
            json.dump(metrics_json, f, indent=2)
