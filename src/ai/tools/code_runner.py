"""Sandboxed Python Calculation & Snippet Runner (`src/ai/tools/code_runner.py`).

Provides ``run_python_calculation(code: str)``:
- Validates the Python AST first to block dangerous modules/builtins (`subprocess`, `shutil`, `socket`,
  `urllib`, `http`, `ctypes`, `os.system`, `open`, `eval`, `exec`, `__import__`).
- Uses ``langchain_experimental.tools.PythonAstREPLTool`` when installed, and falls back to a safe
  local AST-verified execution environment with `math`, `datetime`, `statistics`, `re`, `json`,
  and captured `stdout`.
"""
from __future__ import annotations

import ast
import contextlib
import datetime
import io
import json
import math
import re
import statistics
from typing import Any, Callable, Dict, List, Set

from .internet import _tool_decorator

_FORBIDDEN_MODULES: Set[str] = {
    "subprocess",
    "shutil",
    "socket",
    "urllib",
    "http",
    "requests",
    "ftplib",
    "telnetlib",
    "ctypes",
    "multiprocessing",
    "threading",
    "pty",
    "fcntl",
    "resource",
    "signal",
    "sys",
    "os",
    "pathlib",
    " importlib",
    "importlib",
    "builtins",
}

_FORBIDDEN_CALLS: Set[str] = {
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
}


def _validate_safe_ast(code: str) -> ast.Module:
    """Parse ``code`` and reject any unsafe imports, dunder attribute access, or system calls."""
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod in _FORBIDDEN_MODULES:
                    raise ValueError(f"Import of module '{root_mod}' is blocked for safety.")
        elif isinstance(node, ast.ImportFrom):
            root_mod = (node.module or "").split(".")[0]
            if root_mod in _FORBIDDEN_MODULES:
                raise ValueError(f"Import from module '{root_mod}' is blocked for safety.")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                raise ValueError(f"Access to dunder attribute '{node.attr}' is blocked.")
        elif isinstance(node, ast.Name):
            if node.id.startswith("__") and node.id.endswith("__"):
                raise ValueError(f"Access to dunder identifier '{node.id}' is blocked.")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                raise ValueError(f"Call to '{node.func.id}()' is blocked for safety.")
    return tree


def _SafeImport(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    allowed = {"math": math, "datetime": datetime, "statistics": statistics, "re": re, "json": json}
    root = (name or "").split(".")[0]
    if root in allowed:
        return allowed[root]
    raise ImportError(f"Module '{name}' is not in the allowed sandbox list ({sorted(allowed)}).")


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
    "__import__": _SafeImport,
}


@_tool_decorator
def run_python_calculation(code: str) -> str:
    """Execute a safe Python calculation or data/date processing snippet (supports math, datetime, statistics, re, json)."""
    snippet = (code or "").strip()
    if not snippet:
        return json.dumps({"error": "Empty Python code snippet"})

    # Strip optional markdown code fences if the model wrapped its snippet
    if snippet.startswith("```"):
        snippet = re.sub(r"^```(?:python)?\s*", "", snippet, flags=re.I)
        snippet = re.sub(r"\s*```$", "", snippet)
        snippet = snippet.strip()

    try:
        tree = _validate_safe_ast(snippet)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Blocked or invalid Python code: {type(exc).__name__}: {exc}"})

    env_locals: Dict[str, Any] = {
        "math": math,
        "datetime": datetime,
        "statistics": statistics,
        "re": re,
        "json": json,
    }

    # 1. Try langchain_experimental PythonAstREPLTool if installed
    try:
        from langchain_experimental.tools import PythonAstREPLTool  # type: ignore

        repl = PythonAstREPLTool(locals=env_locals)
        out = str(repl.run(snippet)).strip()
        return json.dumps(
            {"code": snippet, "output": out, "runner": "PythonAstREPLTool"},
            ensure_ascii=False,
        )
    except Exception:  # noqa: BLE001 - fall back to built-in AST sandbox runner
        pass

    # 2. Built-in safe AST runner with stdout capture + last-expression evaluation
    stdout_buf = io.StringIO()
    safe_globals: Dict[str, Any] = {"__builtins__": _SAFE_BUILTINS, **env_locals}
    try:
        last_val: Any = None
        with contextlib.redirect_stdout(stdout_buf):
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                prefix_mod = ast.Module(body=tree.body[:-1], type_ignores=[])
                if prefix_mod.body:
                    exec(compile(prefix_mod, "<sandbox>", "exec"), safe_globals, env_locals)
                expr_ast = ast.Expression(body=tree.body[-1].value)
                last_val = eval(compile(expr_ast, "<sandbox>", "eval"), safe_globals, env_locals)
            else:
                exec(compile(tree, "<sandbox>", "exec"), safe_globals, env_locals)
                if "result" in env_locals:
                    last_val = env_locals["result"]

        printed = stdout_buf.getvalue().strip()
        output_text = printed
        if last_val is not None:
            output_text = f"{printed}\n{last_val}".strip() if printed else str(last_val)
        return json.dumps(
            {
                "code": snippet,
                "output": output_text,
                "result": last_val if isinstance(last_val, (int, float, str, bool, list, dict)) else str(last_val) if last_val is not None else None,
                "runner": "ast-sandbox",
            },
            ensure_ascii=False,
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {"code": snippet, "error": f"{type(exc).__name__}: {exc}"},
            ensure_ascii=False,
        )


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
