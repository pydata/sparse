"""Generate the code reference pages."""

from pathlib import Path

import sparse

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent

for item in dir(sparse):
    if item.startswith("_") or not getattr(getattr(sparse, item), "__module__", "").startswith("sparse"):
        continue
    full_doc_path = Path("api/" + item + ".md")
    with mkdocs_gen_files.open(Path("api", f"{item}.md"), "w") as fd:
        print(f"# {item}", file=fd)
        print("::: " + f"sparse.{item}", file=fd)
    mkdocs_gen_files.set_edit_path(full_doc_path, root)
