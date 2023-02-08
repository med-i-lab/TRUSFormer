from typing import Any, Dict, Optional, Sequence, Tuple, Union, Literal, List, overload

import numpy as np
import torch
from pytorch_lightning import Callback, LightningModule, Trainer

from ....data.exact.splits import SplitsConfig, get_splits, CENTERS
from ....data.exact.resources import metadata

from torchmetrics import (
    MetricCollection,
    Metric,
)
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.auroc import AUROC
from torchmetrics.classification.specificity import Specificity
from torchmetrics.classification.precision_recall_curve import PrecisionRecallCurve
from torchmetrics.classification.f_beta import F1Score
from torchmetrics.classification.avg_precision import AveragePrecision
from torchmetrics.classification.precision_recall import Recall


from torchmetrics.functional import auc, precision_recall_curve


# from torchdrug.metrics import area_under_prc

GLEASONS = ["3+4", "4+3", "4+4", "4+5", "5+4", "5+5"]


# class AUPRC(Metric):
#    def update(self, pred, targets):
#        self.pred = pred
#        self.targets = targets
#
#    def compute(self):
#        pred = self.pred
#        pred = pred[:, 1].view(-1)
#        return area_under_prc(pred, self.targets)
#
#    def reset(self):
#        self.pred = None
#        self.targets = None


def get_metrics(pre_name, num_classes, use_all_metrics=False):
    if use_all_metrics:
        metrics = MetricCollection(
            {
                pre_name + "_auc": AUROC(num_classes=num_classes),
                pre_name + "_ap": AveragePrecision(num_classes=num_classes),
                pre_name + "_acc_macro": Accuracy(num_classes=num_classes, average='macro', multiclass=True),
                pre_name + "_sen": Recall(num_classes=num_classes, average='macro', multiclass=True),
                pre_name + "_spe": Specificity(num_classes=num_classes, average='macro', multiclass=True),
                pre_name + "_f1": F1Score(num_classes=num_classes, average='macro', multiclass=True),
            }
        )
        return metrics
    else:
        metrics = MetricCollection(
            {
                # pre_name + '_acc_macro': Accuracy(num_classes=num_classes, average='macro', multiclass=True),
                # pre_name + '_auc': AUROC(num_classes=num_classes),
                pre_name + "_sen": Recall(num_classes=num_classes, average='macro', multiclass=True),
            }
        )
        return metrics


def aggregate_patch_preds(patch_logits, core_len):
    """This function takes the mean of all patch preds inside a core as prediction of that core."""
    patch_preds = patch_logits.argmax(dim=1)
    core_preds = []
    corelen_cumsum = torch.cumsum(torch.tensor([0] + list(core_len)), dim=0)
    for i, value in enumerate(corelen_cumsum):
        if i == 0:  # or value > patch_preds.size()[0]:
            continue
        minus1_idx = corelen_cumsum[i - 1]
        _core_pred = patch_preds[minus1_idx:value]
        core_preds.append(_core_pred.sum() / len(_core_pred))
    core_preds = torch.tensor(core_preds).reshape(-1, 1)
    return torch.cat((1.0 - core_preds, core_preds), dim=1)


class GleasonMetric:
    def __init__(
        self,
        GS_name: str,
        prefix,
        mode,
        num_classes=2,
        corewise: bool = False,
        device="cuda:0",
    ):
        # self.GS = GS
        self.GS_name = GS_name
        self.prefix = prefix
        self.mode = mode
        self.num_classes = num_classes
        self.kwargs = {
            "on_step": False,
            "on_epoch": True,
            "sync_dist": True,
            "prog_bar": False,
            "add_dataloader_idx": False,
        }

        self.logits = []
        self.labels = []
        self.grades = []
        self.metric_over_epochs = []

        _core_prefix = "_core" if corewise else ""
        self.metric_collection = get_metrics(
            mode + f"_{GS_name}" + _core_prefix, num_classes
        ).clone(prefix)
        if GS_name == "ALL":
            self.metric_collection = get_metrics(
                mode + _core_prefix, num_classes, use_all_metrics=True
            ).clone(prefix)

    @property
    def get_allMetricValues(self):
        return self.metric_over_epochs

    def update(self, logits, labels, grades=None):
        if len(labels) == 0:
            return
        self.labels.append(labels)
        self.logits.append(logits)
        if grades is not None:
            self.grades.append(grades)

    def compute(self):
        if len(self.labels) == 0:
            return "empty"
        _labels = torch.cat(self.labels)
        _logits = torch.cat(self.logits)
        return self.metric_collection(_logits, _labels)

    def log(self, logger):
        metric_dict = self.compute()
        if metric_dict == "empty":
            self.metric_over_epochs.append({})
            return

        self.metric_over_epochs.append(metric_dict)
        if isinstance(logger, LightningModule):
            logger.log_dict(metric_dict, **self.kwargs)
        else:
            logger.log_dict(metric_dict)

    def log_optimum(self, logger, ep_number):
        opt_metric = self.metric_over_epochs[ep_number]
        for i, key in enumerate(opt_metric.keys()):
            if "auc" in key or "acc" in key or "ap" in key:
                logger.log(key + "_best", opt_metric[key], **self.kwargs)

    def reset(self, reset_tracker=False):
        self.logits = []
        self.labels = []
        self.grades = []
        self.metric_collection.reset()

        if reset_tracker:
            self.metric_over_epochs = []


class CenterMetric:
    def __init__(
        self,
        center_name,
        mode,
        num_classes=2,
        corewise: bool = False,
        device="cuda:0",
        GS=[["3+4", "4+3"], ["4+4", "4+5", "5+4", "5+5"]],
    ):
        self.center_name = center_name
        self.mode = mode
        self.num_classes = num_classes
        self.corewise = corewise
        self.device = device
        self.GS = GS

        self.val_prefix = (
            f"val/{center_name}/" if mode == "finetune" else f"val/ssl/{center_name}/"
        )
        self.test_prefix = (
            f"test/{center_name}/" if mode == "finetune" else f"test/ssl/{center_name}/"
        )
        if center_name == "ALL":
            self.val_prefix = f"val/" if mode == "finetune" else f"val/ssl/"
            self.test_prefix = f"test/" if mode == "finetune" else f"test/ssl/"

        self.gs_groups = self.get_gs_groups()
        self.initialize()

    def get_gs_groups(self):
        gs_groups = {}
        get_gs_as_tuple_primary_secondry = lambda gs_string: (int(gs_string[0]), int(gs_string[2]))

        for i, GS in enumerate(self.GS):
            if isinstance(GS, str):
                gs_groups[GS] = list(
                    get_gs_as_tuple_primary_secondry(GS)
                )
                
            elif isinstance(GS, list):
                if GS == ["4+4", "4+5", "5+4", "5+5"]:
                    gs_groups["highGS"] = list(
                        map(get_gs_as_tuple_primary_secondry, GS)
                        )
                    
                elif GS == ["3+4", "4+3"]:
                    gs_groups["lowGS"] = list(
                        map(get_gs_as_tuple_primary_secondry, GS)
                        )
                    
                else:
                    raise ValueError(f"No group defined for this GS group {GS}.")

        return gs_groups

    def initialize(self):
        # initializing GSMetric classes
        self.gs_valMetric_dict = {}
        self.gs_testMetric_dict = {}

        for i, (GS_name, GS) in enumerate(self.gs_groups.items()):

            self.gs_valMetric_dict[GS_name] = GleasonMetric(
                GS_name,
                self.val_prefix,
                self.mode,
                self.num_classes,
                self.corewise,
                self.device,
            )

            self.gs_testMetric_dict[GS_name] = GleasonMetric(
                GS_name,
                self.test_prefix,
                self.mode,
                self.num_classes,
                self.corewise,
                self.device,
            )

        # for all GS
        self.gs_valMetric_dict["ALL"] = GleasonMetric(
            "ALL",
            self.val_prefix,
            self.mode,
            self.num_classes,
            self.corewise,
            self.device,
        )

        self.gs_testMetric_dict["ALL"] = GleasonMetric(
            "ALL",
            self.test_prefix,
            self.mode,
            self.num_classes,
            self.corewise,
            self.device,
        )

        # ["3+4", "4+3", "4+4", "4+5", "5+4", "5+5"]

    # def get_score_tuples(self, grades):
    #     _grades = grades.numpy()
    #     list_tuple_grades = list(zip(_grades[0, :], _grades[1, :]))
    #     return np.array(list_tuple_grades, dtype=object)

    def update(self, set: Literal["val", "test"], logits, labels, grades):
        # gleason_scores = self.get_score_tuples(grades)

        if set == "val":
            GS_objects_dict = self.gs_valMetric_dict
        elif set == "test":
            GS_objects_dict = self.gs_testMetric_dict
            
        # update GS metric for all scores
        GS_objects_dict["ALL"].update(logits, labels, grades)
        
        # update GS metric for separately defined GS groups
        for i, (GS_group_name, GS_group_tuples) in enumerate(self.gs_groups.items()):
            indx = [tuple(grade.numpy()) in GS_group_tuples for grade in grades.T]
                
            # | np.isnan(grades[0, :].numpy())
            _logits = logits[indx]
            _labels = labels[indx]
            GS_objects_dict[GS_group_name].update(_logits, _labels)

    def log(self, logger):
        for i, key in enumerate(self.gs_valMetric_dict.keys()):
            self.gs_valMetric_dict[key].log(logger)
            self.gs_testMetric_dict[key].log(logger)

    def log_optimum(self, logger, ep_number):
        for i, key in enumerate(self.gs_valMetric_dict.keys()):
            self.gs_valMetric_dict[key].log_optimum(logger, ep_number)
            self.gs_testMetric_dict[key].log_optimum(logger, ep_number)

    def compute(self, set):
        if set == "val":
            gs_metric_dict = self.gs_valMetric_dict
        elif set == "test":
            gs_metric_dict = self.gs_valMetric_dict

        all_gs_metrics = {}
        for i, key in enumerate(gs_metric_dict.keys()):
            gs_name, gs_metric = gs_metric_dict[key].compute()
            all_gs_metrics[gs_name] = gs_metric

        return all_gs_metrics

    @overload
    def compute(self, set, gs: str = "ALL"):
        if set == "val":
            gs_metric_dict = self.gs_valMetric_dict
        elif set == "test":
            gs_metric_dict = self.gs_valMetric_dict
        return gs_metric_dict[gs].compute()

    def _getMetric(self, set, gs) -> GleasonMetric:
        if set == "val":
            gs_metric_dict = self.gs_valMetric_dict
        elif set == "test":
            gs_metric_dict = self.gs_testMetric_dict
        return gs_metric_dict[gs]

    def reset(self, reset_tracker=False):
        for i, key in enumerate(self.gs_valMetric_dict.keys()):
            self.gs_valMetric_dict[key].reset(reset_tracker)
            self.gs_testMetric_dict[key].reset(reset_tracker)


class MetricManager:
    def __init__(
        self,
        cohort_specifier: Union[str, List[str]] = "UVA600",
        mode: Literal["online", "finetune"] = "finetune",
        num_classes=2,
        corewise: bool = False,
        device="cuda:0",
    ):
        self.cohort_specifier = cohort_specifier
        self.mode = mode
        self.num_classes = num_classes
        self.corewise = corewise
        self.device = device

        self.initialize()
        # self.metadata = get_splits(SplitsConfig(cohort_specifier))

    def initialize(self):
        self.center_metric_dict = {}
        for i, center in enumerate(self.cohort_specifier):
            self.center_metric_dict[center] = CenterMetric(
                center, self.mode, self.num_classes, self.corewise, self.device
            )

        self.center_metric_dict["ALL"] = CenterMetric(
            "ALL", self.mode, self.num_classes, self.corewise, self.device
        )

    def update(self, set: Literal["val", "test"], center: str, logits, labels, grades):
        self.center_metric_dict[center].update(set, logits, labels, grades)
        self.center_metric_dict["ALL"].update(set, logits, labels, grades)

    def compute(self, set: Literal["val", "test"], center: str):
        return self.center_metric_dict[center].compute(set)

    @overload
    def compute(self, set: Literal["val", "test"], center: str, gs: str = "ALL"):
        return self.center_metric_dict[center].compute(set, gs)

    def _getMetric(self, set: Literal["val", "test"], center: str, gs: str = "ALL"):
        return self.center_metric_dict[center]._getMetric(set, gs)

    def log(self, logger):
        for i, center in enumerate(self.center_metric_dict):
            self.center_metric_dict[center].log(logger)

    def log_optimum(self, logger, ep_number):
        for i, center in enumerate(self.center_metric_dict):
            self.center_metric_dict[center].log_optimum(logger, ep_number)

    def reset(self, reset_tracker=False):
        for i, center in enumerate(self.center_metric_dict):
            self.center_metric_dict[center].reset(reset_tracker)


class PatchMetricManager(MetricManager):
    def __init__(
        self,
        cohort_specifier: Union[str, List[str]] = "UVA600",
        mode: Literal["online", "finetune"] = "finetune",
        num_classes=2,
        device="cuda:0",
    ):
        super(PatchMetricManager, self).__init__(
            cohort_specifier=cohort_specifier,
            mode=mode,
            num_classes=num_classes,
            corewise=False,
            device=device,
        )


class CoreMetricManager(MetricManager):
    def __init__(
        self,
        val_datasets,
        test_datasets,
        cohort_specifier: Union[str, List[str]] = "UVA600",
        mode: Literal["online", "finetune"] = "finetune",
        num_classes=2,
        device="cuda:0",
    ):
        super(CoreMetricManager, self).__init__(
            cohort_specifier=cohort_specifier,
            mode=mode,
            num_classes=num_classes,
            corewise=True,
            device=device,
        )
        self.val_datasets = val_datasets
        self.test_datasets = test_datasets

        self.corelen_dict = self.get_centers_corelen()

    def get_centers_corelen(self):
        corelen_dict = {"val": {}, "test": {}}

        all_val_corelen = []
        all_test_corelen = []
        for i, center in enumerate(self.cohort_specifier):
            corelen_dict["val"][center] = (
                self.val_datasets[center].core_lengths
                if isinstance(self.val_datasets, dict)
                else self.val_datasets.core_lengths
            )
            corelen_dict["test"][center] = (
                self.test_datasets[center].core_lengths
                if isinstance(self.test_datasets, dict)
                else self.test_datasets.core_lengths
            )
            all_val_corelen.append(corelen_dict["val"][center])
            all_test_corelen.append(corelen_dict["test"][center])

        return corelen_dict

    def get_logits_from_patchManager(
        self, set: Literal["val", "test"], patch_manager: PatchMetricManager
    ):
        logits_dict = {}
        labels_dict = {}
        grades_dict = {}
        for i, center in enumerate(self.cohort_specifier):
            gleason_metric = patch_manager._getMetric(set, center=center, gs="ALL")
            if len(gleason_metric.labels) != 0:
                labels_dict[center] = torch.cat(gleason_metric.labels)
                logits_dict[center] = torch.cat(gleason_metric.logits)
                grades_dict[center] = torch.cat(gleason_metric.grades, dim=1)

        return logits_dict, labels_dict, grades_dict

    def aggregate_patches(self, logits, labels, grades, core_len):
        _logits = logits.detach().cpu()
        _core_logits = aggregate_patch_preds(patch_logits=_logits, core_len=core_len)
        corelen_cumsum = torch.cumsum(torch.tensor([0] + list(core_len)), dim=0)[:-1]
        _labels = labels.detach().cpu()
        _core_labels = torch.tensor(
            [_labels[i] for i in corelen_cumsum if i < _labels.size()[0]]
        )
        _grades = grades.detach().cpu()
        _core_grades = torch.cat(
            [_grades[..., i : i + 1] for i in corelen_cumsum if i < _grades.size()[1]],
            dim=1,
        )
        return _core_logits[: _core_labels.size()[0], ...], _core_labels, _core_grades

    def update(
        self, set: Literal["val", "test"], logits: dict, labels: dict, grades: dict
    ):
        for i, center in enumerate(logits.keys()):
            _logits, _labels, _grades = logits[center], labels[center], grades[center]
            _core_len = self.corelen_dict[set][center]
            _core_logits, _core_labels, _core_grade = self.aggregate_patches(
                _logits, _labels, _grades, _core_len
            )
            super().update(set, center, _core_logits, _core_labels, _core_grade)


class CoreMetricManagerTest(CoreMetricManager):
    def __init__(
        self,
        test_datasets,
        cohort_specifier: Union[str, List[str]] = "UVA600",
        mode: Literal["online", "finetune"] = "finetune",
        num_classes=2,
        device="cuda:0",
    ):
        super(CoreMetricManagerTest, self).__init__(
            test_datasets,
            test_datasets,
            cohort_specifier=cohort_specifier,
            mode=mode,
            num_classes=num_classes,
            device=device,
        )