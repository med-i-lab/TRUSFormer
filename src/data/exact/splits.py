from dataclasses import dataclass
from itertools import chain
from typing import Any, Callable, Sequence, Union, overload, NewType
from sklearn.model_selection import train_test_split
from .resources import metadata, miccai_splits
from typing import Optional, List, Callable, Mapping, Tuple
import pandas as pd
import numpy as np
from collections import OrderedDict


# TODO needs complete refactor


CoresList = List[str]
SplitsTuple = Tuple[CoresList, CoresList, CoresList]


CENTERS = ["CRCEO", "JH", "PCC", "PMCC", "UVA"]


from sklearn.model_selection import train_test_split
import numpy as np


@dataclass
class SplitsConfig:

    _target_: str = "src.data.exact.splits.Splits"

    cohort_specifier: Tuple[str] = ("UVA600",)
    train_centers: Any = "all"
    eval_centers: Any = "all"
    train_val_split_seed: int = 0
    train_val_ratio: float = 0.1
    test_as_val: bool = False
    undersample_benign_train: bool = True
    undersample_benign_eval: bool = False
    benign_cores_selection_seed: int = 0
    merge_train_centers: bool = True
    merge_val_centers: bool = False
    merge_test_centers: bool = False


class PatientCohort:
    def __init__(self, train_val_split_seed=0):
        ...


class Splits:
    def __init__(
        self,
        cohort_specifier=("UVA600",),
        train_centers="all",
        eval_centers="all",
        train_val_split_seed=0,
        train_val_ratio=0.1,
        test_as_val=False,
        undersample_benign_train=True,
        undersample_benign_eval=False,
        benign_cores_selection_seed=0,
        merge_train_centers: bool = True,
        merge_val_centers: bool = False,
        merge_test_centers: bool = False,
    ):

        self.cohort_specifier = cohort_specifier
        self.test_as_val = test_as_val

        self._splits = get_splits(
            cohort_specifier,
            resample_train_val=True,
            split_seed=train_val_split_seed,
            train_val_ratio=train_val_ratio,
            undersample_benign_eval=undersample_benign_eval,
            undersample_benign_train=undersample_benign_train,
            benign_cores_selection_seed=benign_cores_selection_seed,
        )

        self._centers = (train_centers, eval_centers, eval_centers)

        self.merge_train_centers = merge_train_centers
        self.merge_val_centers = merge_val_centers
        self.merge_test_centers = merge_test_centers

        self._split2idx = {
            "train": 0,
            "val": 1,
            "test": 2,
        }

    def get_train(
        self, merge_centers=None
    ) -> Union[CoresList, Mapping[str, CoresList]]:
        if merge_centers is None:
            merge_centers = self.merge_train_centers
        return self._get_splits_impl(idx=0, merge_centers=merge_centers)

    def get_val(self, merge_centers=None) -> Union[CoresList, Mapping[str, CoresList]]:
        if merge_centers is None:
            merge_centers = self.merge_val_centers
        return self._get_splits_impl(idx=1, merge_centers=self.merge_val_centers)

    def get_test(self, merge_centers=None) -> Union[CoresList, Mapping[str, CoresList]]:
        if merge_centers is None:
            merge_centers = self.merge_test_centers
        return self._get_splits_impl(idx=2, merge_centers=merge_centers)

    def get_all(
        self,
    ):
        return list(
            chain(self.get_train(True), self.get_val(True), self.get_test(True))
        )

    def _get_splits_impl(
        self, idx, merge_centers=False
    ) -> Union[CoresList, Mapping[str, CoresList]]:

        if isinstance(self._splits, dict):
            splits = {name: split[idx] for name, split in self._splits.items()}
            splits = {
                name: split
                for name, split in splits.items()
                if (self._centers[idx] == "all" or name in self._centers[idx])
            }
            if merge_centers:
                return list(chain(*splits.values()))
            else:
                return splits
        else:
            return self._splits[idx]  # type:ignore

    def to_flattened_dict(self, merge_train=True):
        out = {}
        if merge_train:
            out["train"] = self.get_train(merge_centers=True)
        else:
            train = self.get_train(merge_centers=False)
            if isinstance(train, dict):
                for k, v in train.items():
                    out[f"train_{k}"] = v
            else:
                out["train"] = train

        val = self.get_val(merge_centers=False)
        if isinstance(val, dict):
            for k, v in val.items():
                out[f"val_{k}"] = v
        else:
            out["val"] = val

        test = self.get_test(merge_centers=False)
        if isinstance(test, dict):
            for k, v in test.items():
                out[f"test_{k}"] = v

        else:
            out["test"] = test

        return out

    def apply_filters(self, *filters: Callable):
        for filter in filters:
            self._splits = filter_splits(self._splits, filter)
        return self

    def lengths(self):
        return {k: len(v) for k, v in self.to_flattened_dict(merge_train=False).items()}

    def to_dataframe(self):

        specifiers = []
        split = []

        for specifier in self.get_train():
            specifiers.append(specifier)
            split.append("train")

        for specifier in self.get_val():
            specifiers.append(specifier)
            split.append("val")

        for specifier in self.get_test():
            specifiers.append(specifier)
            split.append("test")

        return pd.DataFrame(dict(specifier=specifiers, split=split))

    @staticmethod
    def get_metadata(split: List[str]):
        from .resources import metadata

        return metadata().query("core_specifier in @split")


def select_patient_splits_with_specified_cancer_core_ratio(
    patient_cohort,
    target_cancer_core_test_size: float = 0.15,
    random_state=0,
    tolerance=0.02,
):

    TOLERANCE = tolerance
    patient_group_test_size = target_cancer_core_test_size
    SEARCH_RATE = 0.02
    curr_random_state = random_state
    n_iters = 1

    _metadata = metadata()

    while True:

        if patient_group_test_size < 0.01 or patient_group_test_size > 0.99:
            raise ValueError("Not possible to achieve desired splits")

        train, test = train_test_split(
            patient_cohort,
            test_size=patient_group_test_size,
            random_state=curr_random_state,
        )

        # look at the number of cancerous cores for each split
        test_cores_len = len(
            _metadata.query('patient_specifier in @test and grade != "Benign"')
        )
        train_cores_len = len(
            _metadata.query('patient_specifier in @train and grade != "Benign"')
        )

        # compute the ratio
        test_ratio = test_cores_len / (train_cores_len + test_cores_len)

        if abs(test_ratio - target_cancer_core_test_size) <= TOLERANCE:
            import logging

            logging.getLogger("Splits").debug(
                f"It took {n_iters} iterations to obtain the desired splits"
            )
            return list(np.sort(train)), list(np.sort(test))

        # if too test cores, increase the size of test patient cohort
        # for next iteration.
        if test_ratio < target_cancer_core_test_size:
            patient_group_test_size += SEARCH_RATE

        # if too few many cores, decrease the size of test patient cohort
        else:
            patient_group_test_size -= SEARCH_RATE

        curr_random_state += 1
        n_iters += 1


# def invert_splits(splits) -> Any:
#    """Inverts the splits from a dict of train, val, test tuples to a tuple train, val, test of dicts"""
#
#    if not isinstance(splits, dict):
#        # nothing to do
#        return splits
#
#    else:
#        splits = splits.copy()
#        train = {k: v[0] for k, v in splits.items()}
#        val = {k: v[1] for k, v in splits.items()}
#        test = {k: v[2] for k, v in splits.items()}
#        return train, val, test


def resample_splits(train_patients, val_patients, split_seed, train_val_ratio=0.25):
    val_size = train_val_ratio
    all_patients = train_patients + val_patients
    return train_test_split(all_patients, test_size=val_size, random_state=split_seed)


def filter_splits(splits, _filter):
    def _apply_filter(split):
        return list(filter(_filter, split))

    if isinstance(splits, list) and isinstance(splits[0], str):
        return _apply_filter(splits)

    if isinstance(splits, tuple):
        return tuple([_apply_filter(split) for split in splits])

    if isinstance(splits, Mapping):
        return {
            center: filter_splits(center_splits, _filter)
            for center, center_splits in splits.items()
        }

    else:
        raise ValueError(
            f"Filter splits accepts dictionaries of tuples train, val, test."
        )


def select_cohort(
    center,
    train_val_split_seed=0,
    val_ratio=0.10,
    undersample_benign_train=True,
    undersample_benign_eval=True,
    benign_cores_selection_seed=0,
):

    from . import resources

    patient_test_sets = resources.patient_test_sets()
    metadata = resources.metadata()
    test_patients = patient_test_sets[center]
    train_patients = metadata.query(
        "center == @center and patient_specifier not in @test_patients"
    )["patient_specifier"].unique()

    # from sklearn.model_selection import train_test_split

    (
        train_patients,
        val_patients,
    ) = select_patient_splits_with_specified_cancer_core_ratio(
        train_patients,
        target_cancer_core_test_size=val_ratio,
        random_state=train_val_split_seed,
    )

    # train_test_split(
    #    train_patients, test_size=val_ratio, random_state=train_val_split_seed
    # )

    test_cores_cancer = metadata.query(
        'center == @center and grade != "Benign" and patient_specifier in @test_patients'
    )
    test_cores_benign = metadata.query(
        'center == @center and grade == "Benign" and patient_specifier in @test_patients'
    )
    val_cores_cancer = metadata.query(
        'center == @center and grade != "Benign" and patient_specifier in @val_patients'
    )
    val_cores_benign = metadata.query(
        'center == @center and grade == "Benign" and patient_specifier in @val_patients'
    )
    train_cores_cancer = metadata.query(
        'center == @center and grade != "Benign" and patient_specifier in @train_patients'
    )
    train_cores_benign = metadata.query(
        'center == @center and grade == "Benign" and patient_specifier in @train_patients'
    )

    import pandas as pd

    def select_benign_cores(
        cores_cancer: pd.DataFrame, cores_benign: pd.DataFrame, random_state
    ):
        num_cancer = len(cores_cancer)
        benign = cores_benign.sample(
            num_cancer, replace=False, random_state=random_state
        )
        return cores_cancer, benign

    if undersample_benign_eval:
        test_cores_cancer, test_cores_benign = select_benign_cores(
            test_cores_cancer,
            test_cores_benign,
            random_state=benign_cores_selection_seed,
        )
        val_cores_cancer, val_cores_benign = select_benign_cores(
            val_cores_cancer, val_cores_benign, random_state=benign_cores_selection_seed
        )

    if undersample_benign_train:
        train_cores_cancer, train_cores_benign = select_benign_cores(
            train_cores_cancer,
            train_cores_benign,
            random_state=benign_cores_selection_seed,
        )

    test_cores = pd.concat([test_cores_cancer, test_cores_benign], axis=0)
    val_cores = pd.concat([val_cores_cancer, val_cores_benign], axis=0)
    train_cores = pd.concat([train_cores_cancer, train_cores_benign], axis=0)

    import numpy as np

    train_cores = list(np.sort(train_cores["core_specifier"].values))  # type:ignore
    val_cores = list(np.sort(val_cores["core_specifier"].values))  # type:ignore
    test_cores = list(np.sort(test_cores["core_specifier"].values))  # type:ignore

    return train_cores, val_cores, test_cores


def get_uva_600_splits(resample_train_val, split_seed, train_val_ratio=0.15):

    uva_cohort = pd.merge(
        metadata(), miccai_splits(), on=["center", "patient_id", "loc"]
    )

    test = list(uva_cohort.query('split == "test"')["core_specifier"])

    train_patients = list(
        uva_cohort.query('split == "train"')["patient_specifier"].unique()
    )
    val_patients = list(
        uva_cohort.query('split == "val"')["patient_specifier"].unique()
    )

    if resample_train_val:

        train_patients, val_patients = resample_splits(
            train_patients, val_patients, split_seed, train_val_ratio
        )

    train = list(
        uva_cohort.loc[uva_cohort["patient_specifier"].isin(train_patients)][
            "core_specifier"
        ]
    )
    val = list(
        uva_cohort.loc[uva_cohort["patient_specifier"].isin(val_patients)][
            "core_specifier"
        ]
    )

    return train, val, test


def get_crceo_428_splits(resample_train_val, split_seed, train_val_ratio=0.25):

    from .resources import metadata, crceo_428_splits

    cohort = pd.merge(metadata(), crceo_428_splits(), on=["core_specifier"])

    test = list(cohort.query('split == "test"')["core_specifier"])

    train_patients = list(
        cohort.query('split == "train"')["patient_specifier"].unique()
    )
    val_patients = list(cohort.query('split == "val"')["patient_specifier"].unique())

    if resample_train_val:

        train_patients, val_patients = resample_splits(
            train_patients, val_patients, split_seed, train_val_ratio
        )

    train = list(
        cohort.loc[cohort["patient_specifier"].isin(train_patients)]["core_specifier"]
    )
    val = list(
        cohort.loc[cohort["patient_specifier"].isin(val_patients)]["core_specifier"]
    )

    return train, val, test


# @overload
# def get_splits(
#    cohort_specifier: List[str],
#    resample_train_val=...,
#    split_seed=26,
#    train_val_ratio=0.25,
#    undersample_benign_train=False,
#    undersample_benign_eval=False,
#    benign_cores_selection_seed=0,
# ) -> Mapping[str, SplitsTuple]:
#    ...


# @overload
# def get_splits(config: SplitsConfig) -> Union[SplitsTuple, Mapping[str, SplitsTuple]]:
#    ...


# def get_splits(*args, **kwargs) -> Union[SplitsTuple, Mapping[str, SplitsTuple]]:
#    if isinstance(config := args[0], SplitsConfig):
#        return _get_splits_impl(
#            config.cohort_specifier,
#            config.resample_train_val,
#            config.split_seed,
#            config.train_val_ratio,
#            undersample_benign_train=config.undersample_benign_train,
#            undersample_benign_eval=config.undersample_benign_eval,
#        )
#    else:
#        return _get_splits_impl(*args, **kwargs)
#


def get_splits(
    cohort_specifier: Union[str, List[str]],
    resample_train_val=...,
    split_seed=26,
    train_val_ratio=0.25,
    undersample_benign_eval=False,
    undersample_benign_train=True,
    benign_cores_selection_seed=0,
) -> Union[SplitsTuple, Mapping[str, SplitsTuple]]:

    if cohort_specifier == "UVA600":
        return get_uva_600_splits(resample_train_val, split_seed, train_val_ratio)

    if cohort_specifier == "CRCEO428":
        return get_crceo_428_splits(resample_train_val, split_seed, train_val_ratio)

    if cohort_specifier == "all":

        _metadata = metadata()

        all_patients = list(_metadata["patient_specifier"].unique())

        train_patients, test_patients = train_test_split(
            all_patients, random_state=0, test_size=0.15
        )

        train_patients, val_patients = train_test_split(
            train_patients, random_state=split_seed, test_size=train_val_ratio
        )

        train = list(
            _metadata.loc[_metadata["patient_specifier"].isin(train_patients)][
                "core_specifier"
            ]
        )

        val = list(
            _metadata.loc[_metadata["patient_specifier"].isin(val_patients)][
                "core_specifier"
            ]
        )

        test = list(
            _metadata.loc[_metadata["patient_specifier"].isin(test_patients)][
                "core_specifier"
            ]
        )

        return train, val, test

    else:
        if isinstance(cohort_specifier, str):
            assert (
                cohort_specifier in CENTERS
            ), f"Name {cohort_specifier} not the name of a center."

            return select_cohort(
                cohort_specifier,
                train_val_split_seed=split_seed,
                val_ratio=train_val_ratio,
                undersample_benign_eval=undersample_benign_eval,
                undersample_benign_train=undersample_benign_train,
                benign_cores_selection_seed=benign_cores_selection_seed,
            )

        if isinstance(cohort_specifier, Sequence):
            return {
                _cohort_specifier: get_splits(
                    _cohort_specifier,
                    resample_train_val=resample_train_val,
                    split_seed=split_seed,
                    train_val_ratio=train_val_ratio,
                    undersample_benign_train=undersample_benign_train,
                    undersample_benign_eval=undersample_benign_eval,
                    benign_cores_selection_seed=benign_cores_selection_seed,
                )
                for _cohort_specifier in cohort_specifier  # type:ignore
            }

        else:
            raise NotImplementedError(f"cohort {cohort_specifier} is not implemented")


class HasProstateMaskFilter:
    def __init__(self):
        from .server.segmentation import list_available_prostate_segmentations

        self.patches_with_masks = list_available_prostate_segmentations()

    def __call__(self, core_specifier):
        return core_specifier in self.patches_with_masks


class InvolvementThresholdFilter:
    def __init__(self, inv_threshold):
        self.pct_cancer_threshold = inv_threshold * 100

        self.lookup = {}
        _metadata = metadata()
        for row in range(len(_metadata)):
            core_specifier = _metadata.loc[
                _metadata.index[row], "core_specifier"
            ]  # type:ignore

            pct_cancer = _metadata.loc[
                _metadata.index[row], "pct_cancer"
            ]  # type:ignore

            grade = _metadata.loc[_metadata.index[row], "grade"]  # type:ignore

            self.lookup[core_specifier] = {"pct_cancer": pct_cancer, "grade": grade}

    def __call__(self, core_specifier):

        if self.lookup[core_specifier]["grade"] == "Benign":
            return True

        if self.lookup[core_specifier]["pct_cancer"] >= self.pct_cancer_threshold:
            return True

        return False
