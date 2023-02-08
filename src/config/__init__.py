# from . import register_configs
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize as _initialize
from hydra import compose
from omegaconf import OmegaConf
from hydra.utils import instantiate as hydra_instantiate
from .registry import register_configs
from .resolvers import register_resolvers


instantiate = hydra_instantiate


RELATIVE_CONFIGS_PATH = "../../configs"


global _is_initialized
_is_initialized = False


def initialize():
    """
    Sets up hydra and registers configs and resolvers
    """

    register_configs()
    register_resolvers()
    global _is_initialized
    _is_initialized = True


def load_config(config_path, resolve=True, instantiate=False):
    """
    Loads a config from the configs directory
    :param dotpath: dotpath to the config eg. datamodule/splits/uva600
    :param resolve: whether to resolve the config
    :param instantiate: whether to instantiate the config into a class
    """
    if not GlobalHydra().is_initialized():
        _initialize(config_path=RELATIVE_CONFIGS_PATH)

    if not _is_initialized:
        initialize()

    if config_path.endswith(".yaml"):
        config_path = config_path[:-5]

    # separate config path into directory and file
    config_path = config_path.split("/")
    config_dir = "/".join(config_path[:-1])
    config_file = config_path[-1]

    override = f"+{config_dir}@_here_={config_file}"
    config = compose(overrides=[override])

    if resolve:
        OmegaConf.resolve(config)

    config = OmegaConf.to_object(config)

    if instantiate:
        config = hydra_instantiate(config)

    return config
