def register_configs():
    """
    Registers all the structured configs 
    used with hydra. Hydra structured 
    configs can be used to inherit basic config properties 
    for the given object.

    for example: 
    @dataclass
    class FooConfig: 
        a: int = 1
        b: int = 2

    from hydra.core.config_store import ConfigStore
    cs = ConfigStore.instance()
    cs.store(name="foo_base", node=FooConfig)


    Then in a config file you can do:
    foo.yaml: 

    defaults:
        - foo_base

    a: 3
    """
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()

    from src.data.exact.splits import SplitsConfig

    cs.store("splits_config_base", SplitsConfig, group="datamodule/splits")

    from src.data.exact.transforms import TransformConfig

    cs.store("transform_config_base", TransformConfig, group="datamodule/transform")

    from src.lightning.datamodules.exact_datamodule import (
        ExactPatchDMConfig,
        ExactPatchDMConfigSL,
        ExactCoreDMConfigWithMixing
    )

    cs.store("patch_datamodule_base", ExactPatchDMConfig, group="datamodule")
    cs.store("patch_dm_sl_base", ExactPatchDMConfigSL, group="datamodule")
    cs.store("core_dm_mixing_base", ExactCoreDMConfigWithMixing, group="datamodule")

    from src.lightning.lightning_modules.configure_optimizer_mixin import OptimizerConfig

    cs.store("optim_defaults", OptimizerConfig, group="optimizer")

    from src.lightning.lightning_modules.self_supervised.vicreg import VICRegConfig

    cs.store("vicreg_base", VICRegConfig, group="model")

    from src.lightning.lightning_modules.self_supervised.finetune import FinetunerConfig

    cs.store("finetuner_base", FinetunerConfig, "model")

    from src.lightning.lightning_modules.supervised.supervised_patch_model import (
        SupervisedModelConfig,
        SupervisedPatchModelWithCenterDiscConfig,
    )

    cs.store("supervised_patch_model_base", SupervisedModelConfig, "model")

    from src.modeling.seq_models import TransformerConfig

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

    from src.driver.core_clf_refactor import ExperimentConfigWrapper

    cs.store("core_clf_base", ExperimentConfigWrapper, "driver")

    from src.lightning.lightning_modules.self_supervised.finetune import (
        CoreFinetunerConfig,
    )

    cs.store("core_finetuner_base", CoreFinetunerConfig, "model")

    # from .driver import supervised

    # cs.store("supervised_base", supervised.ExperimentConfigWrapper, "driver")

    from src.data.exact.segmentation_transforms import SegmentationTransformConfig

    cs.store("seg_transform_base", SegmentationTransformConfig, "datamodule/transforms")

    from src.lightning.lightning_modules.self_supervised.vicreg import (
        VICRegWithCenterDiscConfig,
    )

    cs.store("vicreg_with_disc_base", VICRegWithCenterDiscConfig, group="model")

    from src.lightning.lightning_modules.supervised.center_classifier import (
        CenterClassifierConfig,
    )

    cs.store("center_classifier_base", CenterClassifierConfig, "model")

    from src.lightning.lightning_modules.supervised.center_classifier import (
        RandomPatientDivisionClassifierConfig,
    )

    cs.store(
        "random_patient_division_classifier_base",
        RandomPatientDivisionClassifierConfig,
        "model",
    )

    from src.lightning.lightning_modules.supervised.center_classifier import (
        CorePositionRegressorConfig,
    )

    cs.store("core_position_regressor_base", CorePositionRegressorConfig, "model")
    