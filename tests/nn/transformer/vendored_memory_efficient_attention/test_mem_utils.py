from providence.nn.transformer.vendored_memory_effecient_attention.utils import dynamic_slice, map_pt

import torch as pt


class TestDynamicSlice:
    """Testing ``dynamic_slice``, a port of the JAX ``slice``"""

    def test_matches_1D(self):
        xs = pt.arange(0, 20, step=1)

        # param 1 and 2 are parallel, so the vvvv extras are ignored
        result = dynamic_slice(xs, [0], [2, 6, 5])
        expected = xs[:2]

        assert (result == expected).all()

    def test_matches_2D(self):
        xs = pt.arange(0, 40, step=1).reshape(-1, 4)

        # param 1 and 2 are parallel, so the vvvv extras are ignored
        result = dynamic_slice(xs, [0, 3], [6, 1])
        expected = pt.tensor([3, 7, 11, 15, 19, 23]).unsqueeze(-1)

        assert (expected == result).all(), "Should return the 3rd element of the first 6 rows"


class TestMapPt:
    """Testing for ``map_pt``, a PyTorch port of (a form of) JAX's ``map``"""

    def test_matches(self):
        # def f(x: pt.Tensor): return x.unsqueeze(-1)
        def f(x: pt.Tensor):
            return x.tile(1, 2)

        xs = pt.arange(0, 10, step=2)
        (stack_of_rows_of_two,) = map_pt(f, xs=xs)

        expected = pt.stack([xs, xs], dim=0).T

        assert (expected == stack_of_rows_of_two).all()
