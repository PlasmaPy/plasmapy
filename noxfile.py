"""
Nox is an automation tool used by PlasmaPy to run tests, build
documentation, and perform other checks. Nox sessions are defined in
noxfile.py.

Running `nox` without arguments will run tests with the version of
Python that `nox` is installed under, skipping slow tests. To invoke a
nox session, enter the top-level directory of this repository and run
`nox -s "<session>"`, where <session> is replaced with the name of the
session. To list available sessions, run `nox -l`.

The tests can be run with the following options:

* "all": run all tests
* "skipslow": run tests, except tests decorated with `@pytest.mark.slow`
* "cov": run all tests with code coverage checks
* "lowest-direct" : run all tests with lowest version of direct dependencies

Doctests are run only for the most recent versions of Python and
PlasmaPy dependencies, and not when code coverage checks are performed.
Some of the checks require the most recent supported version of Python
to be installed.
"""

# Documentation: https://nox.thea.codes

import os
import pathlib
import re
import sys

import nox

# SPEC 0 indicates that scientific Python packages should support
# versions of Python that have been released in the last 3 years, or
# equivalently the most three recently released versions of Python.
# The minimum version of Python should be incremented immediately
# following the first release after October of each year.

supported_python_versions: tuple[str, ...] = ("3.11", "3.12", "3.13")
supported_operating_systems: tuple[str, ...] = ("linux", "macos", "windows")

maxpython = max(supported_python_versions)
minpython = min(supported_python_versions)

current_python = f"{sys.version_info.major}.{sys.version_info.minor}"

# The documentation should be built always using the same version of
# Python. Read the Docs enables new version of Python a few months
# after an October release of Python, so docpython ≠ maxpython in that
# period. After updating docpython, run `nox -s requirements` and update
# GitHub workflows for CI and weekly tests.

docpython = "3.13"

current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
nox.options.sessions: list[str] = [f"tests-{current_python}(skipslow)"]

nox.options.default_venv_backend = "uv|virtualenv"

uv_sync = ("uv", "sync", "--no-progress", "--frozen")

running_on_ci = os.getenv("CI")
running_on_rtd = os.environ.get("READTHEDOCS") == "True"

uv_requirement = "uv >= 0.6.1"


@nox.session
def requirements(session) -> None:
    """
    Regenerate the uv.lock file for running tests and building the
    documentation.
    """
    session.install(uv_requirement)

    # If it becomes possible to exclude the current project when using
    # `uv lock`, we should do so here. That would allow us to add a Nox
    # session to validate the requirements back in.
    session.run("uv", "lock", "--upgrade", "--no-progress")


@nox.session
def validate_requirements(session: nox.Session) -> None:
    """
    Verify that the requirements in :file:`uv.lock` are compatible
    with the requirements in `pyproject.toml`.
    """
    session.install(uv_requirement)
    session.log(
        "🛡 If this check fails, regenerate the pinned requirements in "
        "`uv.lock` with `nox -s requirements`."
    )

    # Generate the cache without updating uv.lock by syncing the
    # current environment. If there ends up being a `--dry-run` option
    # for `uv sync`, we could probably use it here.

    session.run("uv", "sync", "--frozen", "--all-extras", "--no-progress")

    # Verify that uv.lock will be unchanged. Using --offline makes it
    # so that only the information from the cache is used.

    session.run("uv", "lock", "--check", "--offline", "--no-progress")


pytest_command: tuple[str, ...] = (
    "pytest",
    "--pyargs",
    "--durations=5",
    "--tb=short",
    "-n=auto",
    "--dist=loadfile",
)

with_doctests: tuple[str, ...] = ("--doctest-modules", "--doctest-continue-on-failure")

with_coverage: tuple[str, ...] = (
    "--cov=plasmapy",
    "--cov-report=xml",
    "--cov-config=pyproject.toml",
    "--cov-append",
    "--cov-report",
    "xml:coverage.xml",
)

skipslow: tuple[str, ...] = ("-m", "not slow")

test_specifiers: list = [
    nox.param("run all tests", id="all"),
    nox.param("skip slow tests", id="skipslow"),
    nox.param("with code coverage", id="cov"),
    nox.param("lowest-direct", id="lowest-direct"),
]


@nox.session(python=supported_python_versions)
@nox.parametrize("test_specifier", test_specifiers)
def tests(session: nox.Session, test_specifier: nox._parametrize.Param) -> None:
    """Run tests with pytest."""

    session.install(uv_requirement)

    options: list[str] = []

    if test_specifier == "skip slow tests":
        options += skipslow

    if test_specifier == "with code coverage":
        options += with_coverage

    # Doctests are only run with the most recent versions of Python and
    # other dependencies because there may be subtle differences in the
    # output between different versions of Python, NumPy, and Astropy.
    if session.python == maxpython and test_specifier not in {"lowest-direct", "cov"}:
        options += with_doctests

    if gh_token := os.getenv("GH_TOKEN"):
        session.env["GH_TOKEN"] = gh_token

    match test_specifier:
        case "lowest-direct":
            session.install(".[tests]", "--resolution=lowest-direct")
        case _:
            # From https://nox.thea.codes/en/stable/cookbook.html#using-a-lockfile
            session.run_install(
                *uv_sync,
                "--extra=tests",
                env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
            )

    session.run(*pytest_command, *options, *session.posargs)


@nox.session(python=maxpython)
@nox.parametrize(
    ["repository"],
    [
        nox.param("numpy", id="numpy"),
        nox.param("https://github.com/astropy/astropy", id="astropy"),
        nox.param("https://github.com/pydata/xarray", id="xarray"),
        nox.param("https://github.com/lmfit/lmfit-py", id="lmfit"),
        nox.param("https://github.com/pandas-dev/pandas", id="pandas"),
    ],
)
def run_tests_with_dev_version_of(session: nox.Session, repository: str) -> None:
    """
    Run tests against the development branch of a dependency.

    Running this session helps us catch problems resulting from breaking
    changes in an upstream dependency before its official release.
    """

    session.install(uv_requirement)

    if repository == "numpy":
        # From: https://numpy.org/doc/1.26/dev/depending_on_numpy.html
        session.run_install(
            "uv",
            "pip",
            "install",
            "-U",
            "--pre",
            "--only-binary",
            ":all:",
            "-i",
            "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple",
            "numpy",
        )
    else:
        session.install(f"git+{repository}")

    session.install(".[tests]")
    session.run(*pytest_command, *session.posargs)


if running_on_rtd:
    rtd_output_path = pathlib.Path(os.environ.get("READTHEDOCS_OUTPUT")) / "html"
    rtd_output_path.mkdir(parents=True, exist_ok=True)
    doc_build_dir = str(rtd_output_path)
else:
    doc_build_dir = "docs/_build/html"

sphinx_base_command: list[str] = [
    "sphinx-build",
    "docs/",
    doc_build_dir,
    "--nitpicky",
    "--keep-going",
]

if not running_on_rtd:
    sphinx_base_command.extend(
        [
            "--fail-on-warning",
            "--quiet",
        ]
    )

build_html: tuple[str, ...] = ("--builder", "html")
check_hyperlinks: tuple[str, ...] = ("--builder", "linkcheck")

doc_troubleshooting_message = """

📘 Tips for troubleshooting common documentation build failures are in
PlasmaPy's documentation guide at:

🔗 https://docs.plasmapy.org/en/latest/contributing/doc_guide.html#troubleshooting
"""


@nox.session(python=docpython)
def docs(session: nox.Session) -> None:
    """
    Build documentation with Sphinx.

    This session may require installation of pandoc and graphviz.
    """

    session.run(*sphinx_base_command, *build_html, *session.posargs)

    if running_on_ci:
        session.log(doc_troubleshooting_message)

    session.install(uv_requirement)

    session.run_install(
        *uv_sync,
        "--extra=docs",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run(*sphinx_base_command, *build_html, *session.posargs)

    landing_page = pathlib.Path(doc_build_dir) / "index.html"

    if landing_page.exists():
        session.log(f"The documentation may be previewed at {landing_page}")
    else:
        session.error(f"Documentation preview landing page not found: {landing_page}")


@nox.session(python=docpython)
@nox.parametrize(
    ["site", "repository"],
    [
        nox.param("github", "sphinx-doc/sphinx", id="sphinx"),
        nox.param("github", "readthedocs/sphinx_rtd_theme", id="sphinx_rtd_theme"),
        nox.param("github", "spatialaudio/nbsphinx", id="nbsphinx"),
    ],
)
def build_docs_with_dev_version_of(
    session: nox.Session, site: str, repository: str
) -> None:
    """
    Build documentation against the development branch of a dependency.

    The purpose of this session is to catch bugs and breaking changes
    so that they can be fixed or updated earlier rather than later.
    """
    session.install(f"git+https://{site}.com/{repository}", ".[docs]")
    session.run(*sphinx_base_command, *build_html, *session.posargs)


LINKCHECK_TROUBLESHOOTING = """
The Sphinx configuration variables `linkcheck_ignore` and
`linkcheck_allowed_redirects` in `docs/conf.py` can be used to specify
hyperlink patterns to be ignored along with allowed redirects. For more
information, see:

🔗 https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-linkcheck_ignore
🔗 https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-linkcheck_allowed_redirects

These variables are in the form of Python regular expressions:

🔗 https://docs.python.org/3/howto/regex.html
"""


@nox.session(python=docpython)
def linkcheck(session: nox.Session) -> None:
    """Check hyperlinks in documentation."""
    if running_on_ci:
        session.log(LINKCHECK_TROUBLESHOOTING)
    session.install(uv_requirement)
    session.run_install(
        *uv_sync,
        "--extra=docs",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run(*sphinx_base_command, *check_hyperlinks, *session.posargs)


MYPY_TROUBLESHOOTING = """
🛡 To learn more about type hints, check out mypy's cheat sheet at:
  https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html

For more details about specific mypy errors, go to:
🔗 https://mypy.readthedocs.io/en/stable/error_codes.html

🪧 Especially difficult errors can be ignored with an inline comment of
the form: `# type: ignore[error]`, where `error` is replaced with the
mypy error code. Please use sparingly!

🛠 To automatically add type hints for common patterns, run:
  nox -s 'autotyping(safe)'
"""


@nox.session(python=maxpython)
def mypy(session: nox.Session) -> None:
    """Perform static type checking."""

    session.install(uv_requirement)
    session.run_install(
        *uv_sync,
        "--extra=tests",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )

    if running_on_ci:
        session.log(MYPY_TROUBLESHOOTING)

    MYPY_COMMAND: tuple[str, ...] = (
        "mypy",
        ".",
        "--install-types",
        "--non-interactive",
        "--show-error-context",
        "--show-error-code-links",
        "--pretty",
    )

    session.run(*MYPY_COMMAND, *session.posargs)


@nox.session(name="import")
def try_import(session: nox.Session) -> None:
    """Install PlasmaPy and import it."""
    session.install(".")
    session.run("python", "-c", "import plasmapy", *session.posargs)


@nox.session
def build(session: nox.Session) -> None:
    """Build & verify the source distribution and wheel."""
    session.install("twine", "build")
    build_command = ("python", "-m", "build")
    session.run(*build_command, "--sdist")
    session.run(*build_command, "--wheel")
    session.run("twine", "check", "dist/*", *session.posargs)


AUTOTYPING_SAFE: tuple[str, ...] = (
    "--none-return",
    "--scalar-return",
    "--annotate-magics",
)
AUTOTYPING_RISKY: tuple[str, ...] = (
    *AUTOTYPING_SAFE,
    "--bool-param",
    "--int-param",
    "--float-param",
    "--str-param",
    "--bytes-param",
    "--annotate-imprecise-magics",
)


@nox.session
@nox.parametrize("draft", [nox.param(False, id="draft"), nox.param(True, id="final")])
def changelog(session: nox.Session, final: str) -> None:
    """
    Build the changelog with towncrier.

     - 'final': build the combined changelog for the release, and delete
       the individual changelog entries in `changelog`.
     - 'draft': print the draft changelog to standard output, without
       writing to files

    When executing this session, provide the version of the release, as
    in this example:

       nox -s 'changelog(final)' -- 2024.7.0
    """

    if len(session.posargs) != 1:
        raise TypeError(
            "Please provide the version of PlasmaPy to be released "
            "(i.e., `nox -s changelog -- 2024.9.0`"
        )

    source_directory = pathlib.Path("./changelog")

    extraneous_files = source_directory.glob("changelog/*[0-9]*.*.rst?*")
    if final and extraneous_files:
        session.error(
            "Please delete the following extraneous files before "
            "proceeding, as the presence of these files may cause "
            f"towncrier errors: {extraneous_files}"
        )

    version = session.posargs[0]

    year_pattern = r"(202[4-9]|20[3-9][0-9]|2[1-9][0-9]{2}|[3-9][0-9]{3,})"
    month_pattern = r"(1[0-2]|[1-9])"
    patch_pattern = r"(0?[0-9]|[1-9][0-9])"
    version_pattern = rf"^{year_pattern}\.{month_pattern}\.{patch_pattern}$"

    if not re.match(version_pattern, version):
        raise ValueError(
            "Please provide a version of the form YYYY.M.PATCH, where "
            "YYYY is the year past 2024, M is the one or two digit month, "
            "and PATCH is a non-negative integer."
        )

    session.install(".", "towncrier")

    options = ("--yes",) if final else ("--draft", "--keep")

    session.run(
        "towncrier",
        "build",
        "--config",
        "pyproject.toml",
        "--dir",
        ".",
        "--version",
        version,
        *options,
    )

    if final:
        original_file = pathlib.Path("./CHANGELOG.rst")
        destination = pathlib.Path(f"./docs/changelog/{version}.rst")
        original_file.rename(destination)


@nox.session
@nox.parametrize(
    "options",
    [
        nox.param(AUTOTYPING_SAFE, id="safe"),
        nox.param(AUTOTYPING_RISKY, id="aggressive"),
    ],
)
def autotyping(session: nox.Session, options: tuple[str, ...]) -> None:
    """
    Automatically add type hints with autotyping.

    The `safe` option generates very few incorrect type hints, and can
    be used in CI. The `aggressive` option may add type hints that are
    incorrect, so please perform a careful code review when using this
    option.

    To check specific files, pass them after a `--`, such as:

        nox -s 'autotyping(safe)' -- noxfile.py
    """
    session.install(".[tests,docs]", "autotyping", "typing_extensions")
    DEFAULT_PATHS = ("src", "tests", "tools", "*.py", ".github", "docs/*.py")
    paths = session.posargs or DEFAULT_PATHS
    session.run("python", "-m", "autotyping", *options, *paths)


@nox.session
def monkeytype(session: nox.Session) -> None:
    """
    Add type hints to a module based on variable types from running pytest.

    Examples
    --------
    nox -s monkeytype -- plasmapy.particles.atomic
    """

    if not session.posargs:
        session.error(
            "Please add at least one module using a command like: "
            "`nox -s monkeytype -- plasmapy.particles.atomic`"
        )

    session.install(".[tests]")
    session.install("MonkeyType", "pytest-monkeytype", "pre-commit")

    database = pathlib.Path("./monkeytype.sqlite3")

    if not database.exists():
        session.log(f"File {database.absolute()} not found. Running MonkeyType.")
        session.run("pytest", f"--monkeytype-output={database.absolute()}")
    else:
        session.log(f"File {database.absolute()} found.")

    for module in session.posargs:
        session.run("monkeytype", "apply", module)

    session.run("pre-commit", "run", "ruff", "--all-files")
    session.run("pre-commit", "run", "ruff-format", "--all-files")

    session.log("Please inspect newly added type hints for correctness.")
    session.log("Check new type hints with `nox -s mypy`.")


@nox.session
def cff(session: nox.Session) -> None:
    """Validate CITATION.cff against the metadata standard."""
    session.install("cffconvert")
    session.run("cffconvert", "--validate", *session.posargs)


@nox.session
def manifest(session: nox.Session) -> None:
    """
    Check for missing files in MANIFEST.in.

    When run outside of CI, this check may report files that were
    locally created but not included in version control. These false
    positives can be ignored by adding file patterns and paths to
    `ignore` under `[tool.check-manifest]` in `pyproject.toml`.
    """
    session.install("check-manifest")
    session.run("check-manifest", *session.posargs)


@nox.session
def lint(session: nox.Session) -> None:
    """Run all pre-commit hooks on all files."""
    session.install("pre-commit")
    session.run(
        "pre-commit",
        "run",
        "--all-files",
        "--show-diff-on-failure",
        *session.posargs,
    )
