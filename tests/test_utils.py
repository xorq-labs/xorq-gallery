from xorq_gallery.sklearn.utils import (
    get_exprs_dict,
    load_exprs_json_cache,
    update_exprs_json_cache,
)


def test_load_exprs_json_cache_matches_get_exprs_dict():
    update_exprs_json_cache()
    assert load_exprs_json_cache() == get_exprs_dict()
