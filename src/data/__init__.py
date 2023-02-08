import os


DATASET_IDENTIFIERS = [
    "ExactNCT02079025",
]


def data_dir():
    """
    default location of a top-level data directory ('~/data'),
    or read from DATA environment variable.
    """
    if not os.environ.get("DATA"):
        print("Environment variable DATA not set")
    else:
        return os.environ["DATA"]

    root = input("Enter data root: ")
    if not os.path.isdir(root):
        raise ValueError(f"root {root} is not a directory")
    os.environ["DATA"] = root
    return root
