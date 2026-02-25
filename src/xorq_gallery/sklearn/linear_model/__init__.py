"""Linear model examples from scikit-learn."""

import pathlib


scripts = tuple(
    p
    for p in sorted(pathlib.Path(__file__).parent.glob("*.py"))
    if p.name != "__init__.py"
)
