import importlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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


def get_exprs_dict(on_script=None):
    result = {}
    for script in scripts:
        if on_script:
            on_script(script.name)
        exprs = cexcepts(Exception, get_exprs_for_script)(script)
        if exprs:
            result[script.name] = tuple(exprs)
    return result


def update_exprs_json_cache(on_script=None):
    dct = get_exprs_dict(on_script=on_script)
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


def _build_script_exprs(script_name, script_path, expr_names):
    """Build all exprs for a single script. Runs in a worker process."""
    script = Path(script_path)
    return {
        expr_name: built
        for expr_name in expr_names
        if (built := _build_expr(script, expr_name)) is not None
    }


def _get_build_paths_dict_parallel(items, script_by_name, max_workers, on_script):
    result = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_to_script = {
            pool.submit(
                _build_script_exprs,
                script_name,
                str(script_by_name[script_name]),
                tuple(expr_names),
            ): script_name
            for script_name, expr_names in items
        }
        for future in as_completed(future_to_script):
            script_name = future_to_script[future]
            if on_script:
                on_script(script_name)
            try:
                result[script_name] = future.result()
            except Exception:
                result[script_name] = {}
    return result


def get_build_paths_dict(script_names=None, on_script=None, max_workers=None):
    script_by_name = {s.name: s for s in scripts}
    cache = load_exprs_json_cache()
    items = [
        (sn, ens)
        for sn, ens in cache.items()
        if sn in script_by_name and (script_names is None or sn in script_names)
    ]
    match max_workers:
        case None | 1:
            result = {}
            for script_name, expr_names in items:
                if on_script:
                    on_script(script_name)
                result[script_name] = {
                    expr_name: built
                    for expr_name in expr_names
                    if (built := _build_expr(script_by_name[script_name], expr_name))
                    is not None
                }
            return result
        case _:
            return _get_build_paths_dict_parallel(
                items, script_by_name, max_workers, on_script
            )


def update_build_paths_json_cache(script_names=None, on_script=None, max_workers=None):
    rebuilt = get_build_paths_dict(
        script_names=script_names, on_script=on_script, max_workers=max_workers
    )
    if script_names is not None:
        dct = load_build_paths_json_cache()
        dct.update(rebuilt)
    else:
        dct = rebuilt
    build_paths_json_path.write_text(json.dumps(dct))
    return build_paths_json_path


def load_build_paths_json_cache():
    return json.loads(build_paths_json_path.read_text())
