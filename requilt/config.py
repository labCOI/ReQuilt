"""Load the shared YAML configuration used by every notebook."""

from pathlib import Path

import yaml

# Repository root = the folder that contains this package.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


class Config(dict):
    """A dict that also allows attribute access: ``cfg.train.epochs``."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as err:
            raise AttributeError(key) from err


def _wrap(value):
    if isinstance(value, dict):
        return Config({k: _wrap(v) for k, v in value.items()})
    return value


def load_config(path=DEFAULT_CONFIG):
    """Read the YAML config and turn every entry of ``paths`` into an absolute Path.

    Relative paths are resolved against the repository root, so notebooks
    work no matter which folder they are launched from.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = _wrap(yaml.safe_load(f))

    for key, value in cfg.paths.items():
        p = Path(value).expanduser()
        cfg.paths[key] = p if p.is_absolute() else REPO_ROOT / p

    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg
