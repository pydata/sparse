import sparse


def test_namespace():
    assert set(sparse.__all__) == {
        "COO",
        "DOK",
        "GCXS",
        "SparseArray",
        "abs",
        "acos",
        "acosh",
        "add",
        "all",
        "any",
        "argmax",
        "argmin",
        "argwhere",
        "asCOO",
        "as_coo",
        "asarray",
        "asin",
        "asinh",
        "asnumpy",
        "astype",
        "atan",
        "atan2",
        "atanh",
        "bitwise_and",
        "bitwise_invert",
        "bitwise_left_shift",
        "bitwise_not",
        "bitwise_or",
        "bitwise_right_shift",
        "bitwise_xor",
        "bool",
        "broadcast_arrays",
        "broadcast_to",
        "can_cast",
        "ceil",
        "clip",
        "complex128",
        "complex64",
        "concat",
        "concatenate",
        "cos",
        "cosh",
        "diagonal",
        "diagonalize",
        "divide",
        "dot",
        "e",
        "einsum",
        "elemwise",
        "empty",
        "empty_like",
        "equal",
        "exp",
        "expand_dims",
        "expm1",
        "eye",
        "finfo",
        "flip",
        "float16",
        "float32",
        "float64",
        "floor",
        "floor_divide",
        "full",
        "full_like",
        "greater",
        "greater_equal",
        "iinfo",
        "inf",
        "int16",
        "int32",
        "int64",
        "int8",
        "isfinite",
        "isinf",
        "isnan",
        "isneginf",
        "isposinf",
        "kron",
        "less",
        "less_equal",
        "load_npz",
        "log",
        "log10",
        "log1p",
        "log2",
        "logaddexp",
        "logical_and",
        "logical_not",
        "logical_or",
        "logical_xor",
        "matmul",
        "max",
        "mean",
        "min",
        "moveaxis",
        "multiply",
        "nan",
        "nanmax",
        "nanmean",
        "nanmin",
        "nanprod",
        "nanreduce",
        "nansum",
        "negative",
        "newaxis",
        "nonzero",
        "not_equal",
        "ones",
        "ones_like",
        "outer",
        "pad",
        "permute_dims",
        "pi",
        "positive",
        "pow",
        "prod",
        "random",
        "remainder",
        "reshape",
        "result_type",
        "roll",
        "round",
        "save_npz",
        "sign",
        "sin",
        "sinh",
        "sqrt",
        "square",
        "squeeze",
        "stack",
        "std",
        "subtract",
        "sum",
        "tan",
        "tanh",
        "tensordot",
        "tril",
        "triu",
        "trunc",
        "uint16",
        "uint32",
        "uint64",
        "uint8",
        "unique_counts",
        "unique_values",
        "var",
        "where",
        "zeros",
        "zeros_like",
    }

    for attr in sparse.__all__:
        assert hasattr(sparse, attr)

    assert sorted(sparse.__all__) == sparse.__all__
