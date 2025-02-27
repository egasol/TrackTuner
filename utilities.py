from pathlib import Path
import json
from typing import Any


def get_data_path() -> Path:
    return Path("data")


def get_media_path() -> Path:
    return Path("media")


def load_json(filepath: Path) -> Any:
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def save_json(filepath: Path, data: Any, indent: int = 4) -> None:
    filepath.parent.mkdir(exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=indent)
