import json

from toolz import excepts

from xorq_gallery.sklearn import (
    get_exprs_for_script,
    scripts,
)
from xorq_gallery.utils import get_data_file


exprs_json_path = get_data_file("exprs.json")


def get_exprs_dict():
    return {
        script.name: tuple(exprs)
        for (script, exprs) in (
            (script, excepts(Exception, get_exprs_for_script)(script))
            for script in scripts
        )
        if exprs
    }


def update_exprs_json_cache():
    dct = get_exprs_dict()
    exprs_json_path.write_text(json.dumps(dct))
    return exprs_json_path


def load_exprs_json_cache():
    return json.loads(exprs_json_path.read_text())
