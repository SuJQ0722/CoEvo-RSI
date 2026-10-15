"""Stable Python normalization shared by semantic and execution-asset hashes.

Version 1 deliberately ignores docstrings/comments and class-level `name`
assignments (candidate identity). It does not resolve imports, choose assets,
or aggregate file digests: the two harness hash functions own those contracts.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator

NORMALIZATION_VERSION = 1


def _target_names(node: ast.AST) -> Iterator[str]:
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, (ast.Tuple, ast.List)):
        for item in node.elts:
            yield from _target_names(item)


class _NormalizeHarnessPython(ast.NodeTransformer):
    """Normalize executable Python while ignoring comments and docstrings."""

    @staticmethod
    def _strip(body: list[ast.stmt]) -> list[ast.stmt]:
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            return body[1:]
        return body

    def visit_Module(self, node: ast.Module) -> ast.AST:
        node.body = self._strip(node.body)
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.body = self._strip(node.body)
        node.body = [
            statement
            for statement in node.body
            if not (
                isinstance(statement, (ast.Assign, ast.AnnAssign))
                and any(
                    name == "name"
                    for target in (
                        statement.targets
                        if isinstance(statement, ast.Assign)
                        else [statement.target]
                    )
                    for name in _target_names(target)
                )
            )
        ]
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.body = self._strip(node.body)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.body = self._strip(node.body)
        return self.generic_visit(node)


def normalized_python(source: str, *, filename: str = "<unknown>") -> str:
    """Return the version-1 AST dump; syntax errors remain caller-visible."""
    tree = ast.parse(source, filename=filename)
    tree = _NormalizeHarnessPython().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, include_attributes=False)
