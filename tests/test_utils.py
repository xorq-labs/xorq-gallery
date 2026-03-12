from xorq_gallery.sklearn.utils import (
    get_exprs_dict,
    load_build_paths_json_cache,
    load_exprs_json_cache,
)


def test_load_exprs_json_cache_matches_get_exprs_dict():
    cached = load_exprs_json_cache()
    calculated = get_exprs_dict()
    left, right = (
        set((k, tuple(v)) for k, v in el.items()) for el in (cached, calculated)
    )
    assert left == right, (left.difference(right), right.difference(left))


def test_load_build_paths_json_cache_keys_match_exprs_cache():
    exprs_cache = load_exprs_json_cache()
    build_cache = load_build_paths_json_cache()
    for script_name, expr_names in exprs_cache.items():
        assert script_name in build_cache, f"missing script: {script_name}"
        for expr_name in expr_names:
            assert expr_name in build_cache[script_name], (
                f"missing expr: {script_name}:{expr_name}"
            )
