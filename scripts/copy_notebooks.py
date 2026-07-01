import shutil
from pathlib import Path

source_dir = Path("examples")
dest_dir = Path("docs/examples")

dest_dir.mkdir(parents=True, exist_ok=True)

for notebook in source_dir.glob("*.ipynb"):
    shutil.copy2(notebook, dest_dir / notebook.name)
