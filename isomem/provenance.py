"""Provenance failure raised when a checkpoint does not describe the run `results.csv` records."""


class CheckpointProvenanceError(RuntimeError):
    """The checkpoint does not describe the model `results.csv` says was trained."""
