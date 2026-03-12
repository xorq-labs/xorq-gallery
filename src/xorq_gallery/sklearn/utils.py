import importlib
import json

from toolz.curried import excepts as cexcepts

from xorq_gallery.sklearn import (
    get_exprs_for_script,
    scripts,
)
from xorq_gallery.utils import get_data_file


exprs_json_path = get_data_file("exprs.json")
build_paths_json_path = get_data_file("build_paths.json")


def _module_name_for_script(script):
    parts = script.parts
    idx = next(i for i, p in enumerate(parts) if p == "xorq_gallery")
    return ".".join(parts[idx:]).removesuffix(".py")


def get_exprs_dict():
    return {
        script.name: tuple(exprs)
        for (script, exprs) in (
            (script, cexcepts(Exception, get_exprs_for_script)(script))
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


@cexcepts(Exception, handler=lambda _: None)
def _build_expr(script, expr_name):
    import xorq.api as xo

    mod = importlib.import_module(_module_name_for_script(script))
    expr = mod.__dict__[expr_name]
    return str(xo.build_expr(expr))


def get_build_paths_dict():
    script_by_name = {s.name: s for s in scripts}
    cache = load_exprs_json_cache()
    return {
        script_name: {
            expr_name: result
            for expr_name in expr_names
            if (result := _build_expr(script_by_name[script_name], expr_name))
            is not None
        }
        for script_name, expr_names in cache.items()
        if script_name in script_by_name
    }


def update_build_paths_json_cache():
    dct = get_build_paths_dict()
    build_paths_json_path.write_text(json.dumps(dct))
    return build_paths_json_path


def load_build_paths_json_cache():
    return json.loads(build_paths_json_path.read_text())
