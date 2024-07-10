"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

root = Path(__file__).parent.parent

x = ["COO"]
# for i in sparse.__all__:
for i in x:
    if i in ["acos", "acosh", "add", "asin", "asinh"]:
        continue
    file_name = i + ".md"
    full_doc_path = Path("api/" + file_name)
    print(file_name)
    with mkdocs_gen_files.open(full_doc_path, "w") as file_name:
        print("::: sparse." + i, file=file_name)
    mkdocs_gen_files.set_edit_path(full_doc_path, root)
