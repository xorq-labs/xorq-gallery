from xorq_gallery.sklearn.utils import (
    get_exprs_dict,
    load_exprs_json_cache,
)


def test_load_exprs_json_cache_matches_get_exprs_dict():
    cached = load_exprs_json_cache()
    calculated = get_exprs_dict()
    left, right = (
        set((k, tuple(v)) for k, v in el.items()) for el in (cached, calculated)
    )
    assert left == right, (left.difference(right), right.difference(left))
