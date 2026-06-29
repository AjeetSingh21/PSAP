"""Strip trailing NUL bytes from every .py file and verify compile()."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for root, _, files in os.walk(ROOT):
    for fn in files:
        if not fn.endswith(".py"):
            continue
        p = os.path.join(root, fn)
        with open(p, "rb") as f:
            data = f.read()
        cleaned = data.rstrip(b"\x00")
        if len(cleaned) != len(data):
            with open(p, "wb") as f:
                f.write(cleaned)
            print("cleaned", p, "removed", len(data) - len(cleaned), "nulls")
        try:
            compile(cleaned.decode("utf-8"), p, "exec")
        except SyntaxError as e:
            print("SYNTAX ERROR in", p, e, file=sys.stderr)
print("done")
