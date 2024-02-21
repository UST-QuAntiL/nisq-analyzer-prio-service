from os import environ
from os import execvpe as replace_process
from pathlib import Path
from shlex import join
from typing import cast

from dotenv import load_dotenv, set_key
from invoke import task, Context
from invoke.runners import Result

load_dotenv(".flaskenv")
load_dotenv(".env")


# FIXME change this name after renaming the flask template package!
MODULE_NAME = "flask_template"


@task
def doc(c, format_="html", all_=False, color=True):
    """Build the documentation.

    Args:
        c (Context): task context
        format_ (str, optional): the format to build. Defaults to "html".
        all (bool, optional): build all files new. Defaults to False.
        color (bool, optional): color output. Defaults to True.
    """
    cmd = ["sphinx-build", "-b", format_]
    if all_:
        cmd.append("-a")
    if color:
        cmd.append("--color")
    else:
        cmd.append("--no-color")
    cmd += [".", "_build"]
    with c.cd(str(Path("./docs"))):
        c.run(join(cmd), echo=True)


@task
def browse_doc(c):
    """Open the documentation in the browser.

    Args:
        c (Context): task context
    """
    index_path = Path("./docs/_build/html/index.html")
    if not index_path.exists():
        doc(c)

    print(f"Open: file://{index_path.resolve()}")
    import webbrowser

    webbrowser.open_new_tab(str(index_path.resolve()))


@task
def doc_index(c, filter_=""):
    """Search the index of referencable sphinx targets in the documentation.

    Args:
        c (Context): task context
        filter_ (str, optional): an optional filter string. Defaults to "".
    """
    inv_path = Path("./docs/_build/html/objects.inv")
    if not inv_path.exists():
        doc(c)

    if filter_:
        filter_ = filter_.lower()

    with c.cd(str(Path("./docs"))):
        output: Result = c.run(
            join(["python", "-m", "sphinx.ext.intersphinx", "_build/html/objects.inv"]),
            echo=True,
            hide="stdout",
        )
        print(
            "".join(
                l
                for l in output.stdout.splitlines(True)
                if (l and not l[0].isspace()) or (not filter_) or (filter_ in l.lower())
            ),
        )


@task()
def update_dependencies(c):
    """Update dependencies that are derived from the pyproject.toml dependencies (e.g. doc dependencies).

    Args:
        c (Context): task context
    """
    c.run(
        join(
            [
                "poetry",
                "export",
                "--dev",
                "--format",
                "requirements.txt",
                "--output",
                str(Path("./docs/requirements.txt")),
            ]
        ),
        echo=True,
        hide="err",
        warn=True,
    )


@task
def start_broker(c, port=None):
    """Start a redis broker container with docker or podman.

    Resuses an existing container if the environment variable REDIS_CONTAINER_ID is set.
    The reused container ignores the port option!
    Sets the environemnt variable in the .env file if a new container is created.

    Redis port is optionally read from REDIS_PORT environment variable. Use the
    ``reset-broker`` task to remove the old container to create a new container
    with a different port.

    To use podman instead of docker set the DOCKER_CMD environment variable to "podman".

    Args:
        c (Context): task context
        port (str, optional): outside port for connections to redis. Defaults to "6379".
    """
    c = cast(Context, c)
    docker_cmd = environ.get("DOCKER_CMD", "docker")
    container_id = environ.get("REDIS_CONTAINER_ID", None)

    if container_id:
        res: Result = c.run(join([docker_cmd, "restart", container_id]), echo=True)
        if res.failed:
            print(f"Failed to start container with id {container_id}.")
        return

    if not port:
        port = environ.get("REDIS_PORT", "6379")
    c.run(join([docker_cmd, "run", "-d", "-p", f"{port}:6379", "redis"]), echo=True)
    result: Result = c.run(join([docker_cmd, "ps", "-q", "--latest"]), hide=True)
    result_container_id = result.stdout.strip()
    dot_env_path = Path(".env")
    if not dot_env_path.exists():
        dot_env_path.touch()
    set_key(dot_env_path, "REDIS_CONTAINER_ID", result_container_id)


@task
def stop_broker(c):
    """Stop the previously started redis broker container with docker or podman.

    Discovers the container id from the environment variable REDIS_CONTAINER_ID.
    If the variable is not set ``--latest`` is used (this assumes that the latest
    created container is the broker!).

    To use podman instead of docker set the DOCKER_CMD environment variable to "podman".

    Args:
        c (Context): task context
    """
    c = cast(Context, c)
    docker_cmd = environ.get("DOCKER_CMD", "docker")
    container_id = environ.get("REDIS_CONTAINER_ID", "--latest")
    c.run(join([docker_cmd, "stop", container_id]))


@task
def worker(
    c,
    pool="threads",
    concurrency=2,
    dev=False,
    log_level="INFO",
    periodic_scheduler=False,
):
    """Run the celery worker, optionally starting the redis broker.

    Args:
        c (Context): task context
        pool (str, optional): the executor pool to use for celery workers (defaults to "solo" for development on linux and windows)
        concurrency (int, optional): the number of concurrent workers (defaults to 1 for development)
        dev (bool, optional): If true the redis docker container will be started before the worker and stopped after the workers finished. Defaults to False.
        log_level (str, optional): The log level of the celery logger in the worker (DEBUG|INFO|WARNING|ERROR|CRITICAL|FATAL). Defaults to "INFO".
        periodic_scheduler (bool, optional): If true a celery beat scheduler will be started alongside the worker. This is needed for periodic tasks. Should only be set to True for one worker otherwise the periodic tasks get executed too often (see readme file).
    """
    CELERY_WORKER = "qhana_plugin_runner.celery_worker:CELERY"

    if dev:
        start_broker(c)
    c = cast(Context, c)
    cmd = [
        "celery",
        "--app",
        CELERY_WORKER,
        "worker",
        f"--pool={pool}",
        "--concurrency",
        str(concurrency),
        "--loglevel",
        log_level.upper(),
        "-E",
    ]

    if periodic_scheduler:
        cmd += ["-B"]

    if dev:
        c.run(join(cmd), echo=True)
        stop_broker(c)
    else:
        # if not in dev mode completely replace the current process with the started process
        print(join(cmd))
        replace_process(cmd[0], cmd, environ)
