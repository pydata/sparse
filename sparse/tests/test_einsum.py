import pytest
import numpy as np
import sparse


einsum_cases = [
    "a,->a",
    "ab,->ab",
    ",ab,->ab",
    ",,->",
    "a,ab,abc->abc",
    "a,b,ab->ab",
    "ea,fb,gc,hd,abcd->efgh",
    "ea,fb,abcd,gc,hd->efgh",
    "abcd,ea,fb,gc,hd->efgh",
    "acdf,jbje,gihb,hfac,gfac,gifabc,hfac",
    "cd,bdhe,aidb,hgca,gc,hgibcd,hgac",
    "abhe,hidj,jgba,hiab,gab",
    "bde,cdh,agdb,hica,ibd,hgicd,hiac",
    "chd,bde,agbc,hiad,hgc,hgi,hiad",
    "chd,bde,agbc,hiad,bdi,cgh,agdb",
    "bdhe,acad,hiab,agac,hibd",
    "ab,ab,c->",
    "ab,ab,c->c",
    "ab,ab,cd,cd->",
    "ab,ab,cd,cd->ac",
    "ab,ab,cd,cd->cd",
    "ab,ab,cd,cd,ef,ef->",
    "ab,cd,ef->abcdef",
    "ab,cd,ef->acdf",
    "ab,cd,de->abcde",
    "ab,cd,de->be",
    "ab,bcd,cd->abcd",
    "ab,bcd,cd->abd",
    "eb,cb,fb->cef",
    "dd,fb,be,cdb->cef",
    "bca,cdb,dbf,afc->",
    "dcc,fce,ea,dbf->ab",
    "fdf,cdd,ccd,afe->ae",
    "abcd,ad",
    "ed,fcd,ff,bcf->be",
    "baa,dcf,af,cde->be",
    "bd,db,eac->ace",
    "fff,fae,bef,def->abd",
    "efc,dbc,acf,fd->abe",
    "ab,ab",
    "ab,ba",
    "abc,abc",
    "abc,bac",
    "abc,cba",
    "ab,bc",
    "ab,cb",
    "ba,bc",
    "ba,cb",
    "abcd,cd",
    "abcd,ab",
    "abcd,cdef",
    "abcd,cdef->feba",
    "abcd,efdc",
    "aab,bc->ac",
    "ab,bcc->ac",
    "aab,bcc->ac",
    "baa,bcc->ac",
    "aab,ccb->ac",
    "aab,fa,df,ecc->bde",
    "ecb,fef,bad,ed->ac",
    "bcf,bbb,fbf,fc->",
    "bb,ff,be->e",
    "bcb,bb,fc,fff->",
    "fbb,dfd,fc,fc->",
    "afd,ba,cc,dc->bf",
    "adb,bc,fa,cfc->d",
    "bbd,bda,fc,db->acf",
    "dba,ead,cad->bce",
    "aef,fbc,dca->bde",
    "abab->ba",
]


@pytest.mark.parametrize("subscripts", einsum_cases)
@pytest.mark.parametrize("density", [0.1, 1.0])
def test_einsum(subscripts, density):
    d = 4
    terms = subscripts.split("->")[0].split(",")
    arrays = [sparse.random((d,) * len(term), density=density) for term in terms]
    sparse_out = sparse.einsum(subscripts, *arrays)
    numpy_out = np.einsum(subscripts, *(s.todense() for s in arrays))

    if not numpy_out.shape:
        # scalar output
        assert np.allclose(numpy_out, sparse_out)
    else:
        # array output
        assert np.allclose(numpy_out, sparse_out.todense())


def test_einsum_input_fill_value():
    x = sparse.random(shape=(2,), density=0.5, format="coo", fill_value=2)
    with pytest.raises(ValueError):
        sparse.einsum("cba", x)


format_test_cases = [
    (("coo",), "coo"),
    (("dok",), "dok"),
    (("gcxs",), "gcxs"),
    (("dense",), "dense"),
    (("coo", "coo"), "coo"),
    (("dok", "coo"), "coo"),
    (("coo", "dok"), "coo"),
    (("coo", "dense"), "coo"),
    (("dense", "coo"), "coo"),
    (("dok", "dense"), "dok"),
    (("dense", "dok"), "dok"),
    (("gcxs", "dense"), "gcxs"),
    (("dense", "gcxs"), "gcxs"),
    (("dense", "dense"), "dense"),
    (("dense", "dok", "gcxs"), "coo"),
]


@pytest.mark.parametrize("formats,expected", format_test_cases)
def test_einsum_format(formats, expected):
    inputs = [
        np.random.randn(2, 2, 2)
        if format == "dense"
        else sparse.random((2, 2, 2), density=0.5, format=format)
        for format in formats
    ]
    if len(inputs) == 1:
        eq = "abc->bc"
    elif len(inputs) == 2:
        eq = "abc,cda->abd"
    elif len(inputs) == 3:
        eq = "abc,cad,dea->abe"

    out = sparse.einsum(eq, *inputs)
    assert {
        sparse.COO: "coo",
        sparse.DOK: "dok",
        sparse.GCXS: "gcxs",
        np.ndarray: "dense",
    }[out.__class__] == expected


def test_einsum_shape_check():
    x = sparse.random((2, 3, 4), density=0.5)
    with pytest.raises(ValueError):
        sparse.einsum("aab", x)
    y = sparse.random((2, 3, 4), density=0.5)
    with pytest.raises(ValueError):
        sparse.einsum("abc,acb", x, y)
