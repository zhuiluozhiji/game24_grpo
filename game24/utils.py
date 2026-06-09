"""Expression evaluator and utilities for the 24-Point Game.

Core functionality:
- Parse and evaluate arithmetic expressions safely
- Verify that an expression is a valid 24-point solution
- Extract numbers from expressions and validate they match input
"""

import re
import ast
import operator
from typing import List, Tuple, Optional


# Allowed operators
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


class ExpressionError(Exception):
    """Raised when an expression is invalid."""
    pass


def safe_eval(expr: str) -> float:
    """Safely evaluate an arithmetic expression using AST.

    Only allows: numbers, +, -, *, /, (), and whitespace.
    """
    # Normalize expression
    expr = expr.strip()
    if not expr:
        raise ExpressionError("Empty expression")

    # Remove all whitespace
    expr_clean = re.sub(r'\s+', '', expr)

    # Validate characters: only digits, +, -, *, /, (, ), .
    if not re.match(r'^[\d+\-*/().\s]+$', expr_clean):
        raise ExpressionError(f"Invalid characters in expression: {expr_clean}")

    # Validate balanced parentheses
    depth = 0
    for ch in expr_clean:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if depth < 0:
            raise ExpressionError("Unbalanced parentheses")
    if depth != 0:
        raise ExpressionError("Unbalanced parentheses")

    try:
        tree = ast.parse(expr_clean, mode='eval')
    except SyntaxError as e:
        raise ExpressionError(f"Syntax error: {e}")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type not in OPS:
                raise ExpressionError(f"Unsupported operator: {op_type}")
            return OPS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type not in OPS:
                raise ExpressionError(f"Unsupported operator: {op_type}")
            return OPS[op_type](operand)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ExpressionError(f"Invalid constant: {node.value}")
        else:
            raise ExpressionError(f"Unsupported AST node: {type(node)}")

    try:
        result = _eval(tree)
        return float(result)
    except ZeroDivisionError:
        raise ExpressionError("Division by zero")
    except ExpressionError:
        raise
    except Exception as e:
        raise ExpressionError(f"Evaluation error: {e}")


def extract_numbers(expr: str) -> List[int]:
    """Extract all integer numbers from an expression string.

    Returns numbers as they appear (including duplicates if used multiple times).
    """
    # Find all integer numbers
    numbers = re.findall(r'\d+', expr)
    return [int(n) for n in numbers]


def validate_solution(numbers: List[int], expression: str, target: int = 24, tolerance: float = 1e-6) -> Tuple[bool, str]:
    """Validate whether an expression is a valid 24-point solution.

    Checks:
    1. Expression evaluates to target (24) within tolerance
    2. Expression uses exactly the given numbers, each exactly once
    3. Only allowed operators are used

    Args:
        numbers: List of 4 input integers.
        expression: The arithmetic expression string.
        target: Target value (default 24).
        tolerance: Floating point tolerance for comparison.

    Returns:
        Tuple of (is_valid, reason_string).
    """
    # Check 1: Evaluate expression
    try:
        result = safe_eval(expression)
    except ExpressionError as e:
        return False, f"Expression error: {e}"

    if abs(result - target) > tolerance:
        return False, f"Result is {result}, expected {target}"

    # Check 2: Extract numbers from expression and validate
    expr_numbers = extract_numbers(expression)
    input_sorted = sorted(numbers)
    expr_sorted = sorted(expr_numbers)

    if expr_sorted != input_sorted:
        return False, (
            f"Numbers mismatch. "
            f"Expected (each once): {input_sorted}, "
            f"Got from expression: {expr_sorted}"
        )

    return True, "Valid solution"


def extract_answer_from_completion(completion: str) -> Optional[str]:
    """Extract the expression from model completion.

    Looks for content inside <answer>...</answer> tags.
    If not found, tries to find the last arithmetic expression.
    """
    # Try to extract from <answer> tags
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', completion, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        if re.search(r'\b(no\s*solution|no_solution|unsolvable|none)\b', answer, re.IGNORECASE):
            return None
        return answer

    # Fallback: look for arithmetic expression pattern
    # Match expressions containing numbers, operators, parentheses
    expr_pattern = re.findall(
        r'(?:[\d]+\s*[\+\-\*\/]\s*)+[\d]+|'
        r'\([^\)]+\)|'
        r'[\d]+\s*[\+\-\*\/]\s*\([^\)]+\)',
        completion
    )
    if expr_pattern:
        return expr_pattern[-1].strip()

    return None


def format_prompt(numbers: List[int]) -> str:
    """Create a standard prompt for the 24-point game.

    Uses Qwen chat format with R1-style think/answer template.
    """
    nums_str = ", ".join(str(n) for n in numbers)
    prompt = (
        f"Given the numbers [{nums_str}], use each number exactly once "
        f"with basic arithmetic operations (+, -, *, /) and parentheses "
        f"to make 24.\n\n"
        f"Think step by step inside <think>...</think> tags, "
        f"then provide your final expression inside <answer>...</answer> tags."
    )
    return prompt
