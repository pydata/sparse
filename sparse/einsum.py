import operator
import functools

from sparse import COO


def _einsum_single(lhs, rhs, array):
    # XXX: ensure  COO format?
    if lhs == rhs:
        # XXX: take copy?
        return array

    # check if repeated / 'trace' indices mean we are only taking a subset
    where = {}
    for i, ix in enumerate(lhs):
        where.setdefault(ix, []).append(i)

    selector = None
    for ix, locs in where.items():
        loc0, *rlocs = locs
        if rlocs:
            # repeated index - only select data where all indices match
            loc0, *locs = locs
            subselector = (array.coords[loc0] == array.coords[rlocs]).all(
                axis=0
            )
            if selector is None:
                selector = subselector
            else:
                selector &= subselector

    # indices that are removed (i.e. not in the output / `perm`)
    # are handled by `has_duplicates=True` below
    perm = [lhs.index(ix) for ix in rhs]
    new_shape = tuple(array.shape[i] for i in perm)

    # select the new COO data
    if selector is not None:
        new_coords = array.coords[:, selector][perm]
        new_data = array.data[selector]
    else:
        new_coords = array.coords[perm]
        new_data = array.data

    return COO(new_coords, new_data, shape=new_shape, has_duplicates=True)


def einsum(eq, *arrays):
    if "->" not in eq:
        # from opt_einsum: calc the output autmatically
        lhs = eq
        tmp_subscripts = lhs.replace(",", "")
        rhs = "".join(
            # sorted sequence of indices
            s
            for s in sorted(set(tmp_subscripts))
            # that appear exactly once
            if tmp_subscripts.count(s) == 1
        )
        eq = f"{lhs}->{rhs}"
    else:
        lhs, rhs = eq.split("->")

    if len(arrays) == 1:
        return _einsum_single(lhs, rhs, arrays[0])

    # if multiple arrays: align, broadcast multiply and then use single einsum
    terms = lhs.split(",")

    # get ordered union of indices from all, indicies that only appear
    # in a single term will be removed in the 'preparation' step below
    total = {}
    for t, term in enumerate((*terms, rhs)):
        for ix in term:
            total.setdefault(ix, set()).add(t)
    aligned_term = "".join(ix for ix, apps in total.items() if len(apps) > 1)

    # NB: if every index appears exactly twice,
    # we could dispatch to tensordot here?

    parrays = []
    for term, array in zip(terms, arrays):
        # calc the target indices for this term
        pterm = "".join(ix for ix in aligned_term if ix in term)
        if pterm != term:
            # perform necessary transpose and reductions
            array = _einsum_single(term, pterm, array)
        # calc broadcastable shape
        shape = [
            array.shape[pterm.index(ix)] if ix in pterm else 1
            for ix in aligned_term
        ]
        parrays.append(array.reshape(shape))

    aligned_array = functools.reduce(operator.mul, parrays)

    return _einsum_single(aligned_term, rhs, aligned_array)
