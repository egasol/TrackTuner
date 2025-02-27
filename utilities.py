from pathlib import Path
import json


def get_data_path():
    return Path("data")


def get_media_path():
    return Path("media")


def load_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def save_json(filepath, data, indent=4):
    filepath.parent.mkdir(exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=indent)
