"""MSBD5018 Individual Project — Evaluation of Language Models."""
import os
from pathlib import Path

# Model isolation: pin HuggingFace's cache to THIS project's folder
# (Self_Project/.hf_cache) so models/datasets never mix with other projects'
# shared ~/.cache/huggingface. Importing the `src` package sets this up before
# any transformers/datasets call, so it works regardless of how a script is run.
_PROJECT = Path(__file__).resolve().parents[1]  # src/ -> Self_Project
os.environ["HF_HOME"] = str(_PROJECT / ".hf_cache")
