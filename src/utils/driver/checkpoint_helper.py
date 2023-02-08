import os
import torch
from typing import Union
import logging


log = logging.getLogger(__name__).info


def setup_ckpt_dir(dir, symlink_target=None):
    if not os.path.isdir(dir):
        os.mkdir(dir)
    if symlink_target is not None:
        os.symlink(symlink_target, dir)
        log(f"Creating symbolic link to the checkpoint directory {symlink_target}.")


class CheckpointHelper:
    def __init__(self, dir, memory: int = 1):
        self.dir = dir
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)

        # first element most recent
        self._checkpoints_memory = []
        self.memory = memory

    def save_checkpoint(self, obj, name):
        """
        Saves the checkpoint object using torch.save and deletes older checkpoints
        """
        f = os.path.join(self.dir, name)
        torch.save(obj, f)

        self._checkpoints_memory.insert(0, name)
        # delete old checkpoints
        while len(self._checkpoints_memory) > self.memory:
            old_ckpt_name = self._checkpoints_memory.pop(-1)
            if old_ckpt_name == name:
                continue
            f = os.path.join(self.dir, old_ckpt_name)
            os.remove(f)

    def get_fname(self, name):
        return os.path.join(self.dir, name)

    def load_state_dict(self, sd):
        self._checkpoints_memory = sd["checkpoints"]

    def state_dict(self):
        return {"checkpoints": self._checkpoints_memory}


class CheckpointMemory:
    """
    Handles deletion of past checkpoints
    """

    def __init__(self, memory=5):
        self.memory = memory
        self._old_checkpoints = []

    def update_and_cleanup(self, fpath):
        self.update(fpath)
        self.cleanup()

    def update(self, fpath):
        self._old_checkpoints.insert(0, fpath)

    def cleanup(self):
        while len(self._old_checkpoints) > self.memory:
            old_ckpt = self._old_checkpoints.pop(-1)
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)

    def state_dict(self):
        return {"old_checkpoints": self._old_checkpoints}

    def load_state_dict(self, state_dict, strict=False):
        self._old_checkpoints = state_dict["old_checkpoints"]
