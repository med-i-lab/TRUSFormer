def register_configs():
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()

    from .data.exact.splits import SplitsConfig

    cs.store("splits_config_base", SplitsConfig, group="datamodule/splits")

    from .data.exact.transforms import TransformConfig

    cs.store("transform_config_base", TransformConfig, group="datamodule/transform")

    from .lightning.datamodules.exact_datamodule import (
        ExactPatchDMConfig,
        ExactPatchDMConfigSL,
        ExactCoreDMConfigWithMixing
    )

    cs.store("patch_datamodule_base", ExactPatchDMConfig, group="datamodule")
    cs.store("patch_dm_sl_base", ExactPatchDMConfigSL, group="datamodule")
    cs.store("core_dm_mixing_base", ExactCoreDMConfigWithMixing, group="datamodule")

    from .lightning.lightning_modules.configure_optimizer_mixin import OptimizerConfig

    cs.store("optim_defaults", OptimizerConfig, group="optimizer")

    from .lightning.lightning_modules.self_supervised.vicreg import VICRegConfig

    cs.store("vicreg_base", VICRegConfig, group="model")

    from .modeling.registry import register_configs

    register_configs()

    from .lightning.lightning_modules.self_supervised.finetune import FinetunerConfig

    cs.store("finetuner_base", FinetunerConfig, "model")

    from .lightning.lightning_modules.supervised.supervised_patch_model import (
        SupervisedModelConfig,
        SupervisedPatchModelWithCenterDiscConfig,
    )

    cs.store("supervised_patch_model_base", SupervisedModelConfig, "model")
    # cs.store(
    #     "supervised_with_center_disc_base",
    #     SupervisedPatchModelWithCenterDiscConfig,
    #     "model",
    # )

    from .modeling.seq_models import TransformerConfig

    cs.store("transformer_config_base", TransformerConfig, "model")

    from src.lightning.lightning_modules.supervised.core_classifier_model import (
        TransformerCoreClassifierConfig,
        TransformerCoreClassifierWithLearnableFeatureReductionConfig,
    )

    cs.store("attn_core_clf_base", TransformerCoreClassifierConfig, "model")
    cs.store(
        "attn_core_clf_with_learnable_feat_reduction_base",
        TransformerCoreClassifierWithLearnableFeatureReductionConfig,
        "model",
    )

    from .driver.core_clf_refactor import ExperimentConfigWrapper

    cs.store("core_clf_base", ExperimentConfigWrapper, "driver")

    from src.lightning.lightning_modules.self_supervised.finetune import (
        CoreFinetunerConfig,
    )

    cs.store("core_finetuner_base", CoreFinetunerConfig, "model")

    # from .driver import supervised

    # cs.store("supervised_base", supervised.ExperimentConfigWrapper, "driver")

    from .data.exact.segmentation_transforms import SegmentationTransformConfig

    cs.store("seg_transform_base", SegmentationTransformConfig, "datamodule/transforms")

    from src.lightning.lightning_modules.self_supervised.vicreg import (
        VICRegWithCenterDiscConfig,
    )

    cs.store("vicreg_with_disc_base", VICRegWithCenterDiscConfig, group="model")

    from .lightning.lightning_modules.supervised.center_classifier import (
        CenterClassifierConfig,
    )

    cs.store("center_classifier_base", CenterClassifierConfig, "model")

    from .lightning.lightning_modules.supervised.center_classifier import (
        RandomPatientDivisionClassifierConfig,
    )

    cs.store(
        "random_patient_division_classifier_base",
        RandomPatientDivisionClassifierConfig,
        "model",
    )

    from .lightning.lightning_modules.supervised.center_classifier import (
        CorePositionRegressorConfig,
    )

    cs.store("core_position_regressor_base", CorePositionRegressorConfig, "model")
    