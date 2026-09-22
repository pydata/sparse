"""Separate sparse-array and NumPy-scalar inputs in the pinned masking test."""

import pytest


def pytest_generate_tests(metafunc):
    if metafunc.definition.nodeid == "array_api_tests/test_array_object.py::test_getitem_masking":
        metafunc.parametrize("masking_source", ["sparse-array", "numpy-scalar"], indirect=True)


@pytest.fixture(autouse=True)
def masking_source(request, monkeypatch):
    if not hasattr(request, "param"):
        return

    from array_api_tests import _array_module as xp
    from array_api_tests import hypothesis_helpers as hh

    original_arrays = hh.arrays

    def arrays(dtype, *args, **kwargs):
        is_source = dtype is hh.all_dtypes
        if request.param == "numpy-scalar":
            # Exercise the unsupported scalar-source/sparse-mask combination
            # deterministically, even with CI's single unvectorized example.
            kwargs["shape"] = ()

        strategy = original_arrays(dtype, *args, **kwargs)
        if is_source or request.param == "numpy-scalar":
            strategy = strategy.map(xp.asarray)
        if is_source and request.param == "numpy-scalar":
            strategy = strategy.map(lambda x: x.dtype.type(x[()]))
        return strategy

    # Keep the upstream assertions and mask generation for sparse inputs.
    # Its helper otherwise sometimes turns a zero-dimensional source into a
    # NumPy scalar with x[()], which has a different indexing implementation.
    monkeypatch.setattr(hh, "arrays", arrays)
