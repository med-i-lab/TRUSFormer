import torch
import time
import os


def register_pretrained_model(
    name,
    architecture_name,
    version,
    local_weights_path,
    local_config_path,
    performance_description,
    data_description,
    full_description,
):
    """
    Registers a model to the database and pushes the weights to the image server.
    :param name: Name of the model
    :param architecture_name: Name of the architecture. It should be registered in the registry.
        It should be possible to directly instantiate the model from the registry, then load the weights as is.
    :param version: Version of the model
    :param local_weights_path: Path to the weights on the local machine. It will be pushed to the image server.
    :param local_config_path: Path to the config file on the local machine. Full experiment config which was used to train the model.
    :param performance_description: Description of the performance of the model
    :param data_description: Description of the data used to train the model
    :param full_description: Full description of the model
    """

    # make sure the model can be instantiated
    from src.modeling.registry.registry import _MODELS

    assert architecture_name in _MODELS, f"Model {architecture_name} not registered"
    try:
        from src.modeling.registry import create_model

        model = create_model(architecture_name)
        sd = torch.load(local_weights_path)
        model.load_state_dict(sd)
        model.eval()
    except Exception as e:
        raise Exception(
            f"Could not instantiate model {architecture_name} from {local_weights_path}. Refusing to register the model."
        ) from e

    # We need to move the weights to the image server
    from src.utils.checkpoints import push_checkpoint_to_image_server

    datetime = time.strftime("%Y%m%d_%H%M%S")
    pathname = f"{name}_{version}_{datetime}"
    server_filepath = push_checkpoint_to_image_server(pathname, local_weights_path)

    # We need to read the config file
    with open(local_config_path, "r") as f:
        config = f.read()

    from src.data.database import connect

    with connect() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO ModelRegistry (
                name, architecture_name, version, data_description, performance_description, full_description, experiment_config, weights_filepath
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                name,
                architecture_name,
                version,
                data_description,
                performance_description,
                full_description,
                config,
                server_filepath,
            ),
        )
        conn.commit()

    print(f"Registered model {name} version {version}.")


def create_pretrained_model(id):
    from src.data.database import connect

    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT architecture_name, weights_filepath FROM ModelRegistry WHERE id = %s",
            (id,),
        )
        architecture_name, server_weights_filepath = cur.fetchone()
        filename = server_weights_filepath.split("/")[-1]

        from src.data import data_dir

        local_weights_filepath = os.path.join(data_dir(), "checkpoint_store", filename)

        from src.data.image_server import get_sftp

        sftp = get_sftp()
        sftp.get(server_weights_filepath, local_weights_filepath)

        from src.modeling.registry import create_model

        model = create_model(architecture_name)
        sd = torch.load(local_weights_filepath)
        model.load_state_dict(sd)

        return model
