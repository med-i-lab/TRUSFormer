# Description: Resolvers for config files
from datetime import datetime
from os.path import join
import os
import json
import logging
import hydra
from omegaconf import OmegaConf


_RESOLVERS = {}


def register_resolver(func):
    _RESOLVERS[func.__name__] = func
    return func


def register_resolvers():
    for name, func in _RESOLVERS.items():
        OmegaConf.register_new_resolver(name, func)


def default_workdir(name):
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return join("logs", "experiments", "runs", name, time)


@register_resolver
def workdir_lookup(name, resume_id=None):
    """
    Uses the file .workdirs.json to lookup the workdir for a given job_id.

    usage:
    example.yaml
    ---------

    workdir: ${workdir_lookup:example}
    """
    if resume_id is None:
        return default_workdir(name)

    file = ".workdirs.json"
    if not os.path.exists(file):
        logging.info("Making workdir lookup file")
        with open(file, "w") as f:
            f.write(json.dumps({}))

    with open(file, "r") as f:
        workdir_lookup = json.load(f)

    if (dir := workdir_lookup.get(str(resume_id))) is None:
        out = default_workdir(name)
        workdir_lookup[resume_id] = out
        with open(file, "w") as f:
            f.write(json.dumps(workdir_lookup))
        return out

    else:
        return dir


@register_resolver
def named_checkpoint(name):
    """
    Looks up the path to a named checkpoint in the file named_checkpoints.json
    """

    original_workdir = hydra.utils.get_original_cwd()
    named_checkpoints_file = join(original_workdir, ".named_checkpoints.json")
    with open(named_checkpoints_file, "r") as f:
        named_checkpoints = json.load(f)
    return named_checkpoints[name]


@register_resolver
def uuid():
    import uuid

    return str(uuid.uuid4())
