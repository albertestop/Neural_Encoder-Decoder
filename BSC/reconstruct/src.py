from pathlib import Path
import re
import os
import tempfile
import pprint

def update_py_constants(file_path: str, updates: dict):
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")

    for name, value in updates.items():
        # Convert python objects to valid python source on the RHS
        rhs = value if isinstance(value, str) else pprint.pformat(value, width=88)

        # Match only top-level assignments (start of line, no indentation)
        pattern = re.compile(rf"^{re.escape(name)}\s*=\s*.*$", re.MULTILINE)

        if not pattern.search(text):
            raise KeyError(f"{name} not found as a top-level assignment in {file_path}")

        text = pattern.sub(f"{name} = {rhs}", text, count=1)

    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tf:
        tf.write(text)
        tmp = tf.name
    os.replace(tmp, path)

def next_folder_number(parent_dir: str) -> int:
    p = Path(parent_dir)
    nums = []
    for d in p.iterdir():
        if d.is_dir():
            try:
                nums.append(int(d.name))
            except ValueError:
                pass  # ignore non-numeric folder names
    return (max(nums) + 1) if nums else 0