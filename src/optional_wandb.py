import sys


class _MissingWandb:
    _warned = False

    def _warn(self):
        if self._warned:
            return
        print(
            "Warning: wandb is not installed or could not be imported; "
            "continuing without Weights & Biases logging.",
            file=sys.stderr,
        )
        self._warned = True

    def init(self, *args, **kwargs):
        self._warn()
        return None

    def log(self, *args, **kwargs):
        self._warn()

    def finish(self, *args, **kwargs):
        self._warn()

    def __getattr__(self, name):
        self._warn()

        def _noop(*args, **kwargs):
            return None

        return _noop


def _load_wandb():
    try:
        import wandb

        return wandb
    except Exception:
        missing_wandb = _MissingWandb()
        missing_wandb._warn()
        return missing_wandb


wandb = _load_wandb()
