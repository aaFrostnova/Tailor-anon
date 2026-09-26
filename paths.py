"""Single place for the data roots the pipeline reads.

Every path in this release is written relative to one of the roots below. They
are placeholders: set the environment variables, or edit the defaults, to point
at your own copies of the measurement artifacts and model checkpoints.

    TAILOR_PROJECT    source tree and fragment checkpoints
    TAILOR_WORKSPACE  measurement campaigns, per-image cells, frozen inputs
    TAILOR_ASSETS     third-party model weights (diffusion, VAE, watermarks)
"""
import os
from pathlib import Path

PROJECT = Path(os.environ.get('TAILOR_PROJECT', '/data/tailor/project'))
WORKSPACE = Path(os.environ.get('TAILOR_WORKSPACE', '/data/tailor/workspace'))
ASSETS = Path(os.environ.get('TAILOR_ASSETS', '/data/tailor/assets'))
INPUTS = Path(__file__).resolve().parent / 'inputs'
