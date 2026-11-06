"""Sandboxed Python Calculation & Snippet Runner (`src/ai/tools/code_runner.py`).

Provides ``run_python_calculation(code: str)`` and ``SafeCalculationEngine``:
- Strict whitelist-only AST validator blocking non-whitelisted nodes, imports, dunders, and dangerous calls.
- Loop and iteration guards preventing unbounded execution DOS (prohibiting `while` loops, bounding `for` and comprehensions).
- Rich mathematical and scientific capabilities built directly into the execution environment:
  * Full math functions (trig, log, exp, roots, comb, perm, factorial, gcd, lcm, erf, etc.)
  * Full statistics suite (mean, median, mode, stdev, variance, quantiles, etc.)
  * High-precision Decimal and Fraction support
  * Numerical calculus primitives: `diff(func, x, h=1e-5)` and `integrate(func, a, b, n=1000)`
  * Safe date/time math (date, time, datetime, timedelta, timezone)
- Captures stdout, evaluates final expressions, and guarantees structured JSON output.
"""
from __future__ import annotations

import ast
import contextlib
import datetime
import decimal
import fractions
import io
import json
import math
import re
import statistics
from typing import Any, Callable, Dict, List, Set, Union

from .internet import _tool_decorator

# Whitelist of permissible AST node types. Everything outside this set is blocked.
_ALLOWED_AST_NODES: Set[type] = {
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.Pass,
    ast.Return,
    ast.Break,
    ast.Continue,
    ast.If,
    ast.For,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Starred,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Del,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.Invert,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.LShift,
    ast.RShift,
    ast.BitOr,
    ast.BitXor,
    ast.BitAnd,
    ast.MatMult,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
    ast.Call,
    ast.keyword,
    ast.IfExp,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.Lambda,
    ast.FormattedValue,
    ast.JoinedStr,
    ast.Import,
    ast.ImportFrom,
    ast.alias,
}

if hasattr(ast, "Index"):
    _ALLOWED_AST_NODES.add(getattr(ast, "Index"))

# Explicit rejection messages for hazardous node types
_EXPLICIT_FORBIDDEN_NODE_ERRORS: Dict[type, str] = {
    ast.While: "While loops are prohibited to prevent infinite loops.",
    ast.ClassDef: "Class definitions are prohibited.",
    ast.Yield: "Yield statements and generators are prohibited.",
    ast.YieldFrom: "Yield-from statements are prohibited.",
    ast.Global: "Global statements are prohibited.",
    ast.Nonlocal: "Nonlocal statements are prohibited.",
    ast.Try: "Try/except blocks are prohibited in calculation snippets.",
    ast.With: "With blocks are prohibited in calculation snippets.",
}

if hasattr(ast, "AsyncFunctionDef"):
    _EXPLICIT_FORBIDDEN_NODE_ERRORS[getattr(ast, "AsyncFunctionDef")] = "Async functions are prohibited."
if hasattr(ast, "AsyncFor"):
    _EXPLICIT_FORBIDDEN_NODE_ERRORS[getattr(ast, "AsyncFor")] = "Async for loops are prohibited."
if hasattr(ast, "AsyncWith"):
    _EXPLICIT_FORBIDDEN_NODE_ERRORS[getattr(ast, "AsyncWith")] = "Async with blocks are prohibited."
if hasattr(ast, "Await"):
    _EXPLICIT_FORBIDDEN_NODE_ERRORS[getattr(ast, "Await")] = "Await expressions are prohibited."

_BLOCKED_NAMES: Set[str] = {
    "_bounded_iter",
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "vars",
    "memoryview",
    "exit",
    "quit",
}

_ALLOWED_MODULES: Set[str] = {
    "math",
    "datetime",
    "statistics",
    "decimal",
    "fractions",
    "re",
    "json",
}


def _bounded_iter(iterable: Any, limit: int = 100_000) -> Any:
    """Safeguard iterator capping maximum steps to prevent loop iteration DOS."""
    count = 0
    for item in iterable:
        count += 1
        if count > limit:
            raise RuntimeError(f"Loop iteration limit ({limit}) exceeded.")
        yield item


class _LoopGuardTransformer(ast.NodeTransformer):
    """Instruments loops and comprehensions to enforce iteration boundaries."""

    def __init__(self, limit: int = 100_000) -> None:
        super().__init__()
        self.limit = limit

    def visit_For(self, node: ast.For) -> ast.For:
        self.generic_visit(node)
        node.iter = ast.Call(
            func=ast.Name(id="_bounded_iter", ctx=ast.Load()),
            args=[node.iter],
            keywords=[ast.keyword(arg="limit", value=ast.Constant(value=self.limit))],
        )
        return node

    def visit_comprehension(self, node: ast.comprehension) -> ast.comprehension:
        self.generic_visit(node)
        node.iter = ast.Call(
            func=ast.Name(id="_bounded_iter", ctx=ast.Load()),
            args=[node.iter],
            keywords=[ast.keyword(arg="limit", value=ast.Constant(value=self.limit))],
        )
        return node


def diff(func: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """Numerical derivative of func at x using central difference formula.

    f'(x) ≈ (f(x + h) - f(x - h)) / (2 * h)
    """
    x_val = float(x)
    h_val = float(h)
    if h_val == 0.0:
        raise ValueError("Step size h cannot be zero.")
    return (float(func(x_val + h_val)) - float(func(x_val - h_val))) / (2.0 * h_val)


def integrate(func: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Numerical definite integral of func from a to b using composite Simpson's 1/3 rule."""
    a_val = float(a)
    b_val = float(b)
    if a_val == b_val:
        return 0.0
    intervals = int(n)
    if intervals <= 0:
        raise ValueError("Number of intervals n must be positive.")
    if intervals > 50_000:
        intervals = 50_000
    if intervals % 2 != 0:
        intervals += 1

    h = (b_val - a_val) / intervals
    total = float(func(a_val)) + float(func(b_val))
    for i in range(1, intervals):
        x_i = a_val + i * h
        weight = 4.0 if i % 2 != 0 else 2.0
        total += weight * float(func(x_i))
    return total * (h / 3.0)


def _safe_import(
    name: str,
    globals: Any = None,
    locals: Any = None,
    fromlist: Any = (),
    level: int = 0,
) -> Any:
    root = (name or "").split(".")[0]
    module_map = {
        "math": math,
        "datetime": datetime,
        "statistics": statistics,
        "decimal": decimal,
        "fractions": fractions,
        "re": re,
        "json": json,
    }
    if root in module_map:
        return module_map[root]
    raise ImportError(f"Module '{name}' is not in the allowed sandbox list ({sorted(module_map)}).")


_SAFE_BUILTINS: Dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "chr": chr,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "__import__": _safe_import,
}


def _serialize_result(val: Any) -> Any:
    """Recursively normalize computation results for clean JSON serialization."""
    if isinstance(val, (int, float, str, bool, type(None))):
        return val
    if isinstance(val, (decimal.Decimal, fractions.Fraction)):
        return str(val)
    if isinstance(val, (datetime.date, datetime.datetime, datetime.time, datetime.timedelta)):
        return str(val)
    if isinstance(val, (list, tuple)):
        return [_serialize_result(x) for x in val]
    if isinstance(val, set):
        return [_serialize_result(x) for x in sorted(val, key=str)]
    if isinstance(val, dict):
        return {str(k): _serialize_result(v) for k, v in val.items()}
    return str(val)


class SafeCalculationEngine:
    """Deep, AST-validated execution engine for mathematical, statistical, and date calculations."""

    def __init__(
        self,
        max_loop_iterations: int = 100_000,
        allowed_modules: Set[str] | None = None,
    ) -> None:
        self.max_loop_iterations = max_loop_iterations
        self.allowed_modules = set(allowed_modules) if allowed_modules is not None else set(_ALLOWED_MODULES)

    def validate_ast(self, code_or_tree: Union[str, ast.AST]) -> ast.Module:
        """Strict whitelist-only AST validator rejecting dangerous constructs and dunder access."""
        if isinstance(code_or_tree, str):
            tree = ast.parse(code_or_tree, mode="exec")
        elif isinstance(code_or_tree, ast.Module):
            tree = code_or_tree
        else:
            raise ValueError("Expected Python code string or ast.Module.")

        for node in ast.walk(tree):
            node_type = type(node)
            if node_type in _EXPLICIT_FORBIDDEN_NODE_ERRORS:
                raise ValueError(_EXPLICIT_FORBIDDEN_NODE_ERRORS[node_type])
            if node_type not in _ALLOWED_AST_NODES:
                raise ValueError(f"AST node '{node_type.__name__}' is blocked for safety.")

            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    if root_mod not in self.allowed_modules:
                        raise ValueError(f"Import of module '{root_mod}' is blocked for safety.")
            elif isinstance(node, ast.ImportFrom):
                root_mod = (node.module or "").split(".")[0]
                if root_mod not in self.allowed_modules:
                    raise ValueError(f"Import from module '{root_mod}' is blocked for safety.")
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("_"):
                    raise ValueError(f"Access to private/dunder attribute '{node.attr}' is blocked.")
            elif isinstance(node, ast.Name):
                if node.id.startswith("_") and node.id != "_":
                    raise ValueError(f"Access or assignment to private/internal identifier '{node.id}' is blocked.")
                if node.id in _BLOCKED_NAMES:
                    raise ValueError(f"Access to identifier '{node.id}' is blocked for safety.")
            elif isinstance(node, ast.FunctionDef):
                if node.name.startswith("_"):
                    raise ValueError(f"Function name '{node.name}' cannot start with underscore.")
            elif isinstance(node, ast.arg):
                if node.arg.startswith("__"):
                    raise ValueError(f"Argument name '{node.arg}' cannot be a dunder.")

        return tree

    def get_environment(self) -> Dict[str, Any]:
        """Construct the rich mathematical and scientific execution scope."""
        env: Dict[str, Any] = {
            # Modules
            "math": math,
            "datetime": datetime,
            "statistics": statistics,
            "decimal": decimal,
            "fractions": fractions,
            "re": re,
            "json": json,
            # Decimal & Fraction
            "Decimal": decimal.Decimal,
            "Fraction": fractions.Fraction,
            # Datetime types
            "date": datetime.date,
            "time": datetime.time,
            "datetime": datetime.datetime,
            "timedelta": datetime.timedelta,
            "timezone": datetime.timezone,
            # Calculus
            "diff": diff,
            "integrate": integrate,
            # Internal iteration guard
            "_bounded_iter": _bounded_iter,
        }

        # Math functions & constants
        math_names = [
            "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
            "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
            "exp", "exp2", "expm1", "log", "log2", "log10", "log1p",
            "sqrt", "hypot", "floor", "ceil", "trunc",
            "comb", "perm", "factorial", "gcd", "lcm", "erf", "erfc", "gamma", "lgamma",
            "isclose", "isfinite", "isinf", "isnan", "degrees", "radians",
            "pi", "e", "tau", "inf", "nan",
        ]
        if hasattr(math, "cbrt"):
            math_names.append("cbrt")
        for name in math_names:
            if hasattr(math, name):
                env[name] = getattr(math, name)

        # Statistics functions
        stat_names = [
            "mean", "median", "median_low", "median_high", "median_grouped",
            "mode", "multimode", "stdev", "pstdev", "variance", "pvariance",
            "quantiles", "geometric_mean", "harmonic_mean", "fmean",
        ]
        for name in stat_names:
            if hasattr(statistics, name):
                env[name] = getattr(statistics, name)

        return env

    def execute(self, code: str) -> Dict[str, Any]:
        """Safely execute Python code string and return structured execution result."""
        snippet = (code or "").strip()
        if not snippet:
            return {"error": "Empty Python code snippet"}

        if snippet.startswith("```"):
            snippet = re.sub(r"^```(?:python)?\s*", "", snippet, flags=re.I)
            snippet = re.sub(r"\s*```$", "", snippet)
            snippet = snippet.strip()

        try:
            tree = self.validate_ast(snippet)
        except Exception as exc:  # noqa: BLE001
            return {
                "code": snippet,
                "error": f"Blocked or invalid Python code: {type(exc).__name__}: {exc}",
            }

        # Instrument loop limits
        transformer = _LoopGuardTransformer(limit=self.max_loop_iterations)
        guarded_tree = transformer.visit(tree)
        ast.fix_missing_locations(guarded_tree)

        env_locals = self.get_environment()
        safe_globals = {"__builtins__": _SAFE_BUILTINS, **env_locals}
        stdout_buf = io.StringIO()

        try:
            last_val: Any = None
            with contextlib.redirect_stdout(stdout_buf):
                if guarded_tree.body and isinstance(guarded_tree.body[-1], ast.Expr):
                    prefix_mod = ast.Module(body=guarded_tree.body[:-1], type_ignores=[])
                    if prefix_mod.body:
                        exec(compile(prefix_mod, "<sandbox>", "exec"), safe_globals, env_locals)
                    expr_ast = ast.Expression(body=guarded_tree.body[-1].value)
                    last_val = eval(compile(expr_ast, "<sandbox>", "eval"), safe_globals, env_locals)
                else:
                    exec(compile(guarded_tree, "<sandbox>", "exec"), safe_globals, env_locals)
                    if "result" in env_locals:
                        last_val = env_locals["result"]

            printed = stdout_buf.getvalue().strip()
            output_text = printed
            if last_val is not None:
                output_text = f"{printed}\n{last_val}".strip() if printed else str(last_val)

            serialized_result = _serialize_result(last_val) if last_val is not None else None
            return {
                "code": snippet,
                "output": output_text,
                "result": serialized_result,
                "runner": "ast-sandbox",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "code": snippet,
                "error": f"{type(exc).__name__}: {exc}",
            }


_DEFAULT_ENGINE = SafeCalculationEngine()


def _validate_safe_ast(code: str) -> ast.Module:
    """Backwards-compatibility alias for AST validation."""
    return _DEFAULT_ENGINE.validate_ast(code)


@_tool_decorator
def run_python_calculation(code: str) -> str:
    """Execute a safe Python calculation or data/date processing snippet (supports math, datetime, statistics, re, json)."""
    res = _DEFAULT_ENGINE.execute(code)
    return json.dumps(res, ensure_ascii=False)


CODE_TOOLS: List[Any] = [run_python_calculation]

CODE_TOOL_REGISTRY: Dict[str, Callable[..., str]] = {
    "run_python_calculation": run_python_calculation,
}

CODE_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_python_calculation",
            "description": "Run a sandboxed Python snippet for complex math, statistics, date/time differences, or list/JSON processing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code snippet to execute (e.g. 'import datetime; (datetime.date(2026, 12, 31) - datetime.date(2026, 11, 3)).days').",
                    }
                },
                "required": ["code"],
            },
        },
    }
]
