from datetime import datetime
import json
import os
from os.path import join
import pdb
import hydra
from hydra.utils import instantiate
from rich import print as pprint
from omegaconf import OmegaConf
import logging
from src.config import initialize


from torch.multiprocessing import set_sharing_strategy


set_sharing_strategy("file_system")


def maybe_resume_previous_run(config):
    """
    If `run_id` is set in the config, then we will attempt to resume the previous run.
    If the previous run is found, then the config will be loaded from the previous run.
    If the previous run is not found, then the config will be saved to the current run.
    """
    if config.resume_id is not None:
        logging.info("Experiment is set up with auto-resume")
        if os.path.exists("config.yaml"):
            logging.info("Found config for previous experiment")
            config = OmegaConf.load("config.yaml")
        else:
            logging.info("No config found. Starting from scratch.")
            conf = OmegaConf.to_yaml(config, resolve=True)
            with open("config.yaml", "w") as f:
                f.write(conf)
    else:
        conf = OmegaConf.to_yaml(config, resolve=True)
        with open("config.yaml", "w") as f:
            f.write(conf)

    return config


@hydra.main(
    config_path="configs",
    config_name="config",
    version_base="1.1",
)
def main(config):
    config = maybe_resume_previous_run(config)
    pprint(OmegaConf.to_object(config))

    if "driver" not in config:
        raise ValueError("No driver specified in config")

    driver = instantiate(config.driver)
    out = driver.run()
    import wandb

    wandb.finish()
    return out


if __name__ == "__main__":
    initialize()
    main()
