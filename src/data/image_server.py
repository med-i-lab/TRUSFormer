import pickle
import errno
import functools
import logging
import dotenv
import paramiko
from typing import Literal, Optional
import mat73
from PIL import Image
from functools import lru_cache, wraps
from scipy import io
import numpy as np
import os
from ..utils import load_dotenv


load_dotenv()

logged_in = False
ssh: paramiko.SSHClient
sftp: paramiko.SFTPClient


def login():

    if (username := os.getenv("SERVER_USERNAME")) is None:
        username = input("Enter username for image.cs.queensu.ca: ")

    if (password := os.getenv("SERVER_PASSWORD")) is None:
        password = input(f"Enter password for {username}@image.cs.queensu.ca: ")

    global ssh
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("image.cs.queensu.ca", username=username, password=password)

    global sftp
    sftp = ssh.open_sftp()
    sftp.chdir("/med-i_data/Data/Exact_Ultrasound/data/full_data")

    global logged_in
    logged_in = True


def prompt_login(func):
    @functools.wraps(func)
    def logged_in_func(*args, **kwargs):

        logging.debug(f"login prompted by {func}")
        login()

        return func(*args, **kwargs)

    return logged_in_func


@prompt_login
def get_ssh():
    return ssh


@prompt_login
def get_sftp():
    return sftp
