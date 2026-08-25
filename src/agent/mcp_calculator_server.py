from mcp.server.fastmcp import FastMCP

mcp = FastMCP("calculator-server")


@mcp.tool()
def calculator(a: float, b: float, op: str) -> str:
    """Evaluate a binary arithmetic operation. op is one of '+', '-', '*', '/'."""
    if op == "+":
        result = a + b
    elif op == "-":
        result = a - b
    elif op == "*":
        result = a * b
    elif op == "/":
        if b == 0:
            raise ValueError("division by zero")
        result = a / b
    else:
        raise ValueError(f"unsupported operator: {op}")

    return str(int(result)) if result == int(result) else str(result)


if __name__ == "__main__":
    mcp.run(transport="stdio")