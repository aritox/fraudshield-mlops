# FraudShield development rules

## Mandatory Python environment

This project uses only Python 3.12 from the existing repository virtual environment:

.\.venv\Scripts\python.exe

Never use global Python installations.

Never run these as bare commands:

- python
- python3
- py
- pip
- pytest
- ruff

Always use the virtual-environment interpreter directly:

- .\.venv\Scripts\python.exe -m pip
- .\.venv\Scripts\python.exe -m pytest
- .\.venv\Scripts\python.exe -m ruff
- .\.venv\Scripts\python.exe -m fraudshield

Before running Python-related work, verify:

.\.venv\Scripts\python.exe -c "import sys; print(sys.version); print(sys.executable)"

The result must be Python 3.12 and the executable must be inside this project's
.venv directory.

If the interpreter is not Python 3.12 from .venv, stop and report the issue.
Do not fall back to Python 3.14.
Do not install packages globally.
