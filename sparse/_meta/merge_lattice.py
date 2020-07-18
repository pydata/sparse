import operator
import collections.abc
import itertools
import typing


class MergeLattice:
    def __init__(
        self,
        args: typing.Union[
            int, collections.abc.Iterable[collections.abc.Iterable[int]]
        ],
    ):
        if not isinstance(args, collections.abc.Iterable):
            self._args: typing.Tuple[typing.Tuple[int, ...], ...] = (
                (operator.index(args),),
            )
        else:
            l = []
            for group in args:
                l.append(tuple(operator.index(arg) for arg in group))
            self._args: typing.Tuple[typing.Tuple[int, ...], ...] = tuple(l)

    @property
    def args(self) -> typing.Tuple[typing.Tuple[int, ...], ...]:
        return self._args

    def __eq__(self, other) -> bool:
        if not isinstance(other, MergeLattice):
            return NotImplemented

        return self.args == other.args

    def __and__(self, other) -> "MergeLattice":
        if not isinstance(other, MergeLattice):
            return NotImplemented
        it = (
            itertools.chain(g1, g2)
            for g1, g2 in itertools.product(self.args, other.args)
        )
        return self.merge_args(it)

    def __or__(self, other) -> "MergeLattice":
        if not isinstance(other, MergeLattice):
            return NotImplemented
        it = (
            itertools.chain(g1, g2)
            for g1, g2 in itertools.product(self.args, other.args)
        )
        it = itertools.chain(self.args, other.args, it)
        return self.merge_args(it)

    @staticmethod
    def merge_args(
        args: collections.abc.Iterable[collections.abc.Iterable[int]],
    ) -> "MergeLattice":
        out_set = set()
        out_args = []
        for group in args:
            current_args = []
            current_set = set()
            for arg in group:
                if arg not in current_set:
                    current_set.add(arg)
                    current_args.append(arg)
            current_args_tup = tuple(current_args)
            if current_args_tup not in out_set:
                out_set.add(current_args_tup)
                out_args.append(current_args_tup)
        return MergeLattice(out_args)
