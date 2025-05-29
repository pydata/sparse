from .compiler import LogicCompiler
from .rewrite_tools import gensym


class LogicExecutor:
    def __init__(self, ctx, verbose=False):
        self.ctx: LogicCompiler = ctx
        self.codes = {}
        self.verbose = verbose

    def __call__(self, prgm):
        prgm_structure = prgm
        if prgm_structure not in self.codes:
            thunk = logic_executor_code(self.ctx, prgm)
            self.codes[prgm_structure] = eval(thunk), thunk

        f, code = self.codes[prgm_structure]
        if self.verbose:
            print(code)
        return f(prgm)


def logic_executor_code(ctx, prgm):
    # jc = JuliaContext()
    code = ctx(prgm)
    fname = gensym("compute")
    return f""":(function {fname}(prgm) \n {code} \n end)"""
