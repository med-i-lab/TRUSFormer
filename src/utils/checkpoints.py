import stat
import os


def instantiate_model_with_checkpoint(model=None, checkpoint_path=None):
    return model.load_from_checkpoint(checkpoint_path)


IMAGE_SERVER_CHECKPOINT_DIR = "/med-i_data/exact_prostate_segemnts/checkpoints"


def checkpoint_dir():
    from src.data import data_dir

    dir = os.path.join(data_dir(), "checkpoint_store")
    if not os.path.isdir(dir):
        os.mkdir(dir)
    return dir


def push_checkpoint_to_image_server(checkpoint_name, checkpoint_path):
    from src.data.exact.server import get_sftp

    sftp = get_sftp()

    remote_path = f"{IMAGE_SERVER_CHECKPOINT_DIR}/{checkpoint_name}.ckpt"
    sftp.put(checkpoint_path, remote_path)
    sftp.chmod(remote_path, stat.S_IROTH | stat.S_IRWXU | stat.S_IRWXG)
    return remote_path


def download_checkpoint_from_image_server(checkpoint_name) -> str:
    local_path_target = os.path.join(checkpoint_dir(), checkpoint_name) + ".ckpt"

    from src.data.exact.server import get_sftp

    sftp = get_sftp()

    sftp.get(
        f"{IMAGE_SERVER_CHECKPOINT_DIR}/{checkpoint_name}.ckpt",
        local_path_target,
    )

    return local_path_target


def get_named_checkpoint(checkpoint_name):

    local_path_target = os.path.join(checkpoint_dir(), checkpoint_name + ".ckpt")
    if not os.path.isfile(local_path_target):
        download_checkpoint_from_image_server(checkpoint_name)

    return local_path_target
