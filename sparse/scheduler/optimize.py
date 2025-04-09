from .compiler import LogicCompiler
from .finch_logic import Aggregate, Alias, LogicNode, MapJoin, Plan, Produces, Query
from .rewrite_tools import Chain, PostOrderDFS, PostWalk, PreWalk, Rewrite


def optimize(prgm: LogicNode) -> LogicNode:
    # ...
    return propagate_map_queries(prgm)


def get_productions(root: LogicNode) -> LogicNode:
    for node in PostOrderDFS(root):
        if isinstance(node, Produces):
            return [arg for arg in PostOrderDFS(node) if isinstance(arg, Alias)]
    return []


def propagate_map_queries(root: LogicNode) -> LogicNode:
    def rule_agg_to_mapjoin(ex):
        match ex:
            case Aggregate(op, init, arg, ()):
                return MapJoin(op, (init, arg))

    root = Rewrite(PostWalk(rule_agg_to_mapjoin))(root)
    rets = get_productions(root)
    props = {}
    for node in PostOrderDFS(root):
        match node:
            case Query(a, MapJoin(op, args)) if a not in rets:
                props[a] = MapJoin(op, args)

    def rule_0(ex):
        return props.get(ex)

    def rule_1(ex):
        match ex:
            case Query(a, _) if a in props:
                return Plan(())

    def rule_2(ex):
        match ex:
            case Plan(args) if Plan(()) in args:
                return Plan(tuple(a for a in args if a != Plan(())))

    root = Rewrite(PreWalk(Chain([rule_0, rule_1])))(root)
    return Rewrite(PostWalk(rule_2))(root)


class DefaultLogicOptimizer:
    def __init__(self, ctx: LogicCompiler):
        self.ctx = ctx

    def __call__(self, prgm: LogicNode):
        prgm = optimize(prgm)
        return self.ctx(prgm)
