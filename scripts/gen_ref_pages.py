"""Generate the code reference pages."""

from pathlib import Path

import sparse

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

for item in dir(sparse):
    if item.startswith("_") or not getattr(getattr(sparse, item), "__module__", "").startswith("sparse"):
        continue
    with mkdocs_gen_files.open(Path("api", f"{item}.md"), "w") as fd:
        print("::: " + f"sparse.{item}", file=fd)
