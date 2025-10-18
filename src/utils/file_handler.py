import os
import shutil
from pathlib import Path
from typing import Optional


def ensure_dir(path: str) -> Path:
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def cleanup_temp_files(temp_dir: str) -> None:
    temp_path = Path(temp_dir)
    if temp_path.exists():
        shutil.rmtree(temp_path)
        temp_path.mkdir(parents=True, exist_ok=True)


def get_output_path(input_path: str, output_dir: str, suffix: str = "_translated") -> str:
    input_file = Path(input_path)
    output_path = Path(output_dir) / f"{input_file.stem}{suffix}{input_file.suffix}"
    return str(output_path)


def file_exists(path: str) -> bool:
    return Path(path).exists()


def get_file_size(path: str) -> int:
    return Path(path).stat().st_size