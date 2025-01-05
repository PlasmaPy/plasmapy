"""Script and utilities to launch the plasma calculator."""

__all__ = ["main"]

import argparse
import pathlib
import shlex
import subprocess

_description = """
Plasma calculator is a tool that opens a page in a web browser for
interactive calculation of plasma parameters.

This tool is currently in the prototype stage and is expected to change in
the future. Please raise an issue at the following link to provide suggestions
and feedback: https://github.com/PlasmaPy/PlasmaPy/issues/new
"""


def main() -> None:
    """
    Stub function for command line tool that launches the plasma calculator notebook.
    """
    parser = argparse.ArgumentParser(description=_description)
    parser.add_argument(
        "--port", type=int, default=8866, help="Port to run the notebook"
    )
    parser.add_argument(
        "--dark", action="store_true", help="Turn on dark mode, reduces eye strain"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open the browser"
    )

    module_path = pathlib.Path(__file__).parent.absolute()
    notebook_path = module_path / "plasma_calculator.ipynb"
    favicon_path = module_path / "favicon.ico"

    args = parser.parse_args()
    theme = "dark" if args.dark else "light"
    no_browser = "--no-browser" if args.no_browser else ""

    command = [
        "voila",
        notebook_path,
        f"--port={args.port}",
        f"--theme={theme}",
        f"--VoilaConfiguration.file_whitelist={favicon_path}",
    ]
    if no_browser:
        command.append(no_browser)

    try:
        subprocess.call(command)  # noqa: S603
    except KeyboardInterrupt:
        # We'll need to switch from print() to using logging library
        print("Stopping calculator! Bye")  # noqa: T201
