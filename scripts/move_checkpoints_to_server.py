import json
import argparse
from src.utils.checkpoints import push_checkpoint_to_image_server
from tqdm import tqdm

"""
Based on a json file contained a mapping from checkpoint names to paths, load all of those 
named checkpoints to the server
"""


def main(args):

    with open(args.checkpoints_file, "r") as f:
        d = json.load(f)

    for k, v in tqdm(d.items(), desc="Pushing checkpoints"):
        push_checkpoint_to_image_server(k, v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints_file",
        help="file to parse for moving named points to server",
    )
    args = parser.parse_args()
    main(args)
