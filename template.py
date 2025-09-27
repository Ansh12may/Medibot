import os

from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s:')

list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "requirements.txt",
    "setup.py",
    "app.py",
    "research/trials.ipynb"
]


for file_path in list_of_files:
    file = Path(file_path)
    if not file.exists():
        # create parent directories if they don't exist
        if not file.parent.exists():
            logging.info(f"Creating directory: {file.parent}")
            file.parent.mkdir(parents=True, exist_ok=True)
        # create empty file
        logging.info(f"Creating file: {file}")
        file.touch()
    else:
        logging.info(f"File already exists: {file}")