"""Experiment tracking with MLflow and Weights & Biases behind one interface.

Tracking must never break a run: a backend that is not installed, has no
credentials or raises is dropped with a warning and the experiment goes on.

* MLflow logs to a local ``mlruns/`` store, no account needed.
* W&B logs *online* to wandb.ai when ``WANDB_API_KEY`` is found (environment
  or Colab secret) and falls back to offline otherwise; an offline run is
  uploaded later with ``wandb sync``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Protocol, Sequence

logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "lora-metadata-extraction"


class TrackingBackend(Protocol):
    """What a tracking backend must be able to do."""

    name: str

    def start_run(self, run_name: str, parameters: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    def log_artifact(self, path: Path) -> None: ...
    def end_run(self) -> None: ...


class MlflowBackend:
    """Local MLflow file store."""

    name = "mlflow"

    def __init__(self, project: str) -> None:
        """Point MLflow at a local file store and select the experiment."""
        # Recent MLflow versions refuse the plain file store unless opted in;
        # it needs no database driver, which is what a notebook run wants.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import mlflow

        self._mlflow = mlflow
        mlflow.set_tracking_uri(f"file:{Path('mlruns').resolve().as_posix()}")
        mlflow.set_experiment(project)

    def start_run(self, run_name: str, parameters: dict[str, Any]) -> None:
        """Open a run and log its parameters.

        Args:
            run_name: Display name of the run.
            parameters: Run parameters; each value is logged as a string cut to 250 characters.
        """
        self._mlflow.start_run(run_name=run_name)
        self._mlflow.log_params({key: str(value)[:250] for key, value in parameters.items()})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log numeric metrics, optionally at a training step.

        Args:
            metrics: Metric name to value.
            step: Training step the values belong to; None lets MLflow use its default step.
        """
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: Path) -> None:
        """Attach a file to the current run.

        Args:
            path: Local file to store with the run's artifacts.
        """
        self._mlflow.log_artifact(str(path))

    def end_run(self) -> None:
        """Close the current run."""
        self._mlflow.end_run()

    def list_runs(self) -> Any:
        """All runs of the experiment as a DataFrame (params and metrics as columns)."""
        return self._mlflow.search_runs()


def find_wandb_api_key() -> str | None:
    """Return the W&B API key from the environment or from Colab secrets.

    The key is never written in the notebook: on Colab it is stored once under
    the key icon (*Secrets*), named ``WANDB_API_KEY``, with notebook access on.
    """
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    try:
        from google.colab import userdata

        key = userdata.get("WANDB_API_KEY")
    except Exception:  # noqa: BLE001 - not on Colab, secret missing or access not granted
        return None
    if key:
        os.environ["WANDB_API_KEY"] = key
    return key or None


class WandbBackend:
    """Weights & Biases: online on wandb.ai with an API key, offline without."""

    name = "wandb"

    def __init__(self, project: str) -> None:
        """Choose online or offline mode from the availability of an API key."""
        import wandb

        self._wandb = wandb
        self._project = project
        self._run: Any = None
        # One group per experiment: search trials and the final model sit side by side on wandb.ai.
        self._group = wandb.util.generate_id()
        self.mode = "online" if find_wandb_api_key() else "offline"
        self.project_url: str | None = None
        os.environ["WANDB_MODE"] = self.mode
        if self.mode == "offline":
            os.environ.setdefault("WANDB_SILENT", "true")  # online, W&B prints each run's URL

    def start_run(self, run_name: str, parameters: dict[str, Any]) -> None:
        """Open a run in the project, grouped with the other runs of this session.

        Args:
            run_name: Display name of the run.
            parameters: Run configuration, stored as the W&B config. Its ``stage`` entry, when
                present, becomes the job type; otherwise the job type is ``run``.
        """
        self._run = self._wandb.init(
            project=self._project,
            name=run_name,
            group=self._group,
            job_type=str(parameters.get("stage", "run")),
            config=parameters,
            reinit=True,
        )
        if self.mode == "online":
            self.project_url = self._run.get_project_url()

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log numeric metrics; the training step is logged as a value.

        Args:
            metrics: Metric name to value.
            step: Training step, logged under ``training_step``; None omits it.
        """
        # W&B requires increasing steps within a run; keep ours as a logged value instead.
        self._wandb.log({**metrics, **({"training_step": step} if step is not None else {})})

    def log_artifact(self, path: Path) -> None:
        """Upload a file with the current run.

        Args:
            path: Local file to upload immediately.
        """
        self._wandb.save(str(path), policy="now")

    def end_run(self) -> None:
        """Close the current run, if any."""
        if self._run is not None:
            self._run.finish()
            self._run = None


class ExperimentTracker:
    """Fan-out to every backend that could be created; failures are isolated."""

    def __init__(self, backends: Sequence[TrackingBackend]) -> None:
        """Keep the backends that could be created."""
        self.backends = list(backends)

    @property
    def active_backends(self) -> list[str]:
        """Names of the backends still in use."""
        return [backend.name for backend in self.backends]

    @property
    def backend_destinations(self) -> dict[str, str]:
        """Where each active backend sends its runs."""
        destinations = {}
        for backend in self.backends:
            if isinstance(backend, WandbBackend):
                online = backend.project_url or "wandb.ai"
                destinations[backend.name] = online if backend.mode == "online" else "offline (./wandb, `wandb sync` to upload)"
            elif isinstance(backend, MlflowBackend):
                destinations[backend.name] = "local file store (./mlruns)"
            else:
                destinations[backend.name] = "custom"
        return destinations

    def _call(self, method: str, *args: Any) -> None:
        """Call ``method`` on every backend; a backend that raises is dropped."""
        for backend in list(self.backends):
            try:
                getattr(backend, method)(*args)
            except Exception as error:  # noqa: BLE001 - tracking is best effort
                logger.warning("Tracking backend %s disabled (%s failed: %s)", backend.name, method, error)
                self.backends.remove(backend)

    def start_run(self, run_name: str, parameters: dict[str, Any]) -> None:
        """Open a run on every backend.

        Args:
            run_name: Display name of the run.
            parameters: Run parameters, passed unchanged to each backend.
        """
        self._call("start_run", run_name, parameters)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log the numeric entries of ``metrics`` on every backend.

        Args:
            metrics: Metric name to value. Only ``int`` and ``float`` values are logged, converted
                to float; other entries are ignored, and nothing is logged when none is numeric.
            step: Training step the values belong to, or None.
        """
        numeric = {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
        if numeric:
            self._call("log_metrics", numeric, step)

    def log_loss_history(self, loss_history: Sequence[dict[str, Any]]) -> None:
        """Log per-step training and validation losses recorded by the trainer.

        Args:
            loss_history: Trainer log entries. ``loss``, ``eval_loss`` and ``learning_rate`` are
                logged as ``train_loss``, ``validation_loss`` and ``learning_rate`` at the entry's
                ``step`` (0 when absent); entries with none of these keys are skipped.
        """
        for entry in loss_history:
            metrics = {
                name: entry[key]
                for key, name in (("loss", "train_loss"), ("eval_loss", "validation_loss"), ("learning_rate", "learning_rate"))
                if key in entry
            }
            if metrics:
                self.log_metrics(metrics, step=int(entry.get("step", 0)))

    def log_artifact(self, path: Path | str) -> None:
        """Attach ``path`` to the run on every backend, if the file exists.

        Args:
            path: Local file to attach; a path that does not exist is ignored.
        """
        if Path(path).exists():
            self._call("log_artifact", Path(path))

    def end_run(self) -> None:
        """Close the run on every backend."""
        self._call("end_run")

    def list_mlflow_runs(self) -> Any:
        """MLflow run table, or ``None`` when MLflow is not active."""
        for backend in self.backends:
            if isinstance(backend, MlflowBackend):
                return backend.list_runs()
        return None


def create_experiment_tracker(
    backend_names: Sequence[str] = ("mlflow", "wandb"), project: str = DEFAULT_PROJECT
) -> ExperimentTracker:
    """Build a tracker with every requested backend that is available.

    Args:
        backend_names: Backends to create, among ``mlflow`` and ``wandb``. A name that is
            unknown or whose backend cannot be created is skipped with a warning.
        project: MLflow experiment name and W&B project name.

    Returns:
        A tracker over the backends that could be created, possibly none.
    """
    factories = {"mlflow": MlflowBackend, "wandb": WandbBackend}
    backends: list[TrackingBackend] = []
    for name in backend_names:
        try:
            backends.append(factories[name](project))
        except Exception as error:  # noqa: BLE001 - not installed, no credentials...
            logger.warning("Tracking backend %s unavailable: %s", name, error)
    return ExperimentTracker(backends)
