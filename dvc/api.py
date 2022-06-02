import os
from collections import Counter
from contextlib import _GeneratorContextManager as GCM
from typing import Dict, Iterable, Optional, Union

from funcy import first, reraise

from dvc.exceptions import OutputNotFoundError, PathMissingError
from dvc.repo import Repo


def get_url(path, repo=None, rev=None, remote=None):
    """
    Returns the URL to the storage location of a data file or directory tracked
    in a DVC repo. For Git repos, HEAD is used unless a rev argument is
    supplied. The default remote is tried unless a remote argument is supplied.

    Raises OutputNotFoundError if the file is not tracked by DVC.

    NOTE: This function does not check for the actual existence of the file or
    directory in the remote storage.
    """
    with Repo.open(repo, rev=rev, subrepos=True, uninitialized=True) as _repo:
        fs_path = _repo.dvcfs.from_os_path(path)
        with reraise(FileNotFoundError, PathMissingError(path, repo)):
            info = _repo.dvcfs.info(fs_path)

        dvc_info = info.get("dvc_info")
        if not dvc_info:
            raise OutputNotFoundError(path, repo)

        dvc_repo = info["repo"]
        md5 = dvc_info["md5"]

        return dvc_repo.cloud.get_url_for(remote, checksum=md5)


def open(  # noqa, pylint: disable=redefined-builtin
    path, repo=None, rev=None, remote=None, mode="r", encoding=None
):
    """
    Open file in the supplied path tracked in a repo (both DVC projects and
    plain Git repos are supported). For Git repos, HEAD is used unless a rev
    argument is supplied. The default remote is tried unless a remote argument
    is supplied. It may only be used as a context manager:

        with dvc.api.open(
                'path/to/file',
                repo='https://example.com/url/to/repo'
                ) as fd:
            # ... Handle file object fd
    """
    args = (path,)
    kwargs = {
        "repo": repo,
        "remote": remote,
        "rev": rev,
        "mode": mode,
        "encoding": encoding,
    }
    return _OpenContextManager(_open, args, kwargs)


class _OpenContextManager(GCM):
    def __init__(
        self, func, args, kwds
    ):  # pylint: disable=super-init-not-called
        self.gen = func(*args, **kwds)
        self.func, self.args, self.kwds = func, args, kwds

    def __getattr__(self, name):
        raise AttributeError(
            "dvc.api.open() should be used in a with statement."
        )


def _open(path, repo=None, rev=None, remote=None, mode="r", encoding=None):
    with Repo.open(repo, rev=rev, subrepos=True, uninitialized=True) as _repo:
        with _repo.open_by_relpath(
            path, remote=remote, mode=mode, encoding=encoding
        ) as fd:
            yield fd


def read(path, repo=None, rev=None, remote=None, mode="r", encoding=None):
    """
    Returns the contents of a tracked file (by DVC or Git). For Git repos, HEAD
    is used unless a rev argument is supplied. The default remote is tried
    unless a remote argument is supplied.
    """
    with open(
        path, repo=repo, rev=rev, remote=remote, mode=mode, encoding=encoding
    ) as fd:
        return fd.read()


def get_params(
    *targets: str,
    repo: Optional[str] = None,
    revs: Optional[Union[str, Iterable[str]]] = None,
    deps: bool = False,
    stages: Optional[Union[str, Iterable[str]]] = None,
) -> Dict:
    """Get parameters tracked in `repo`.

    Args:
        *targets: Names of the parameter files to retrieve parameters from.
            If no `targets` are provided, all parameter files will be used.
        repo (str, optional): location of the DVC project.
            Defaults to the current project (found by walking up from the
            current working directory tree).
            It can be a URL or a file system path.
            Both HTTP and SSH protocols are supported for online Git repos
            (e.g. [user@]server:project.git).
        revs (Union[str, Iterable[str]], optionak): Any `Git revision`_ such as
            a branch or tag name, a commit hash or a dvc experiment name.
            Defaults to HEAD.
            If `repo` is not a Git repo, this option is ignored.
        deps (bool, optional): Whether to retrieve only parameters that are
            stage dependencies or not.
            Defaults to `False`.
        stages: (Union[str, Iterable[str]], optional): Name of the stages to
            retrieve parameters from.
            Defaults to `None`.
            If no stages are provided, all parameters from all stages will be
            retrieved.
    Returns:
        Dict: Processed parameters.
            If no `revs` were provided, parameters will be available as
            top-level keys.
            If `revs` were provided, the top-level keys will be those revs.
            See Examples below.

    Examples:

        - No args.

        Will use the current project as `repo` and retrieve all parameters,
        for all stages, for the current revision.

        >>> import json
        >>> import dvc.api
        >>> params = dvc.api.get_params()
        >>> print(json.dumps(params, indent=4))
        {
            "prepare": {
                "split": 0.2,
                "seed": 20170428
            },
            "featurize": {
                "max_features": 10000,
                "ngrams": 2
            },
            "train": {
                "seed": 20170428,
                "n_est": 50,
                "min_split": 0.01
            }
        }

        - Filter with `stages`.

        >>> import json
        >>> import dvc.api
        >>> params = dvc.api.get_params(stages="prepare")
        >>> print(json.dumps(params, indent=4))
        {
            "prepare": {
                "split": 0.2,
                "seed": 20170428
            }
        }

        - Git URL as `repo`.

        >>> import json
        >>> import dvc.api
        >>> params = dvc.api.get_params(
        ...     repo="https://github.com/iterative/demo-fashion-mnist")
        {
            "train": {
                "batch_size": 128,
                "hidden_units": 64,
                "dropout": 0.4,
                "num_epochs": 10,
                "lr": 0.001,
                "conv_activation": "relu"
            }
        }

        - Multiple `revs`.

        >>> import json
        >>> import dvc.api
        >>> params = dvc.api.get_params(
        ...     repo="https://github.com/iterative/demo-fashion-mnist",
        ...     revs=["main", "low-lr-experiment"])
        >>> print(json.dumps(params, indent=4))
        {
            "main": {
                "train": {
                    "batch_size": 128,
                    "hidden_units": 64,
                    "dropout": 0.4,
                    "num_epochs": 10,
                    "lr": 0.001,
                    "conv_activation": "relu"
                }
            },
            "low-lr-experiment": {
                "train": {
                    "batch_size": 128,
                    "hidden_units": 64,
                    "dropout": 0.4,
                    "num_epochs": 10,
                    "lr": 0.001,
                    "conv_activation": "relu"
                }
            }
        }

    """

    if isinstance(revs, str):
        revs = [revs]
    if isinstance(stages, str):
        stages = [stages]

    def _onerror_raise(result: Dict, exception: Exception, *args, **kwargs):
        raise exception

    def _postprocess(params):
        processed = {}
        for rev, rev_data in params.items():
            processed[rev] = {}

            counts = Counter()
            for file_data in rev_data["data"].values():
                for k in file_data["data"]:
                    counts[k] += 1

            for file_name, file_data in rev_data["data"].items():
                to_merge = {
                    (k if counts[k] == 1 else f"{file_name}:{k}"): v
                    for k, v in file_data["data"].items()
                }
                processed[rev] = {**processed[rev], **to_merge}

        if first(processed) == "":
            return processed[""]

        if "workspace" in processed:
            del processed["workspace"]

        return processed

    with Repo.open(repo) as _repo:
        params = _repo.params.show(
            revs=revs,
            targets=targets,
            deps=deps,
            onerror=_onerror_raise,
            stages=stages,
        )

    return _postprocess(params)


def make_checkpoint():
    """
    Signal DVC to create a checkpoint experiment.

    If the current process is being run from DVC, this function will block
    until DVC has finished creating the checkpoint. Otherwise, this function
    will return immediately.
    """
    import builtins
    from time import sleep

    from dvc.env import DVC_CHECKPOINT, DVC_ROOT
    from dvc.stage.monitor import CheckpointTask

    if os.getenv(DVC_CHECKPOINT) is None:
        return

    root_dir = os.getenv(DVC_ROOT, Repo.find_root())
    signal_file = os.path.join(
        root_dir, Repo.DVC_DIR, "tmp", CheckpointTask.SIGNAL_FILE
    )

    with builtins.open(signal_file, "w", encoding="utf-8") as fobj:
        # NOTE: force flushing/writing empty file to disk, otherwise when
        # run in certain contexts (pytest) file may not actually be written
        fobj.write("")
        fobj.flush()
        os.fsync(fobj.fileno())
    while os.path.exists(signal_file):
        sleep(0.1)
