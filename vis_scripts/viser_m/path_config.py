from __future__ import annotations



from pathlib import Path
import os


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

SMPL_DATA_PATH = "../../prep"

REPO_ROOT = Path(SMPL_DATA_PATH)# _repo_root()
REPO_SMPL_DATA_DIR = REPO_ROOT / "data" / "smpl"


PREP_DATA_DIR = REPO_ROOT / "prep" / "data"
PREP_SMPL_DATA_DIR = PREP_DATA_DIR / "smpl"


REPO_ROOT = _repo_root()
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_INIT_DIR = RESULTS_DIR / "init"
RESULTS_OUTPUT_DIR = RESULTS_DIR / "output"

# Output buckets
SCENE_OUTPUT_DIR = RESULTS_OUTPUT_DIR / "scene"
AGENT_OUTPUT_DIR = RESULTS_OUTPUT_DIR / "agent"

# VSLAM-related data
VSLAM_ROOT = RESULTS_INIT_DIR / "vslam"
VSLAM_MEGACAM_DIR = VSLAM_ROOT / "megacam"
VSLAM_PRIORS_DIR = VSLAM_ROOT / "raw_mega_priors"

# Contacts and auxiliary annotations
CONTACTS_ROOT = RESULTS_INIT_DIR / "contacts"

# Flow / mask caches (assumed to be staged inside results/init)
FLOWS_COVISIBILITY_DIR = RESULTS_INIT_DIR / "flows"
DYN_MASK_DIR = RESULTS_INIT_DIR / "dyn_mask"

# SMPL / GVHMR assets
GVHMR_ROOT = REPO_ROOT / "prep" / "HMR"
GVHMR_BODY_MODELS_DIR = GVHMR_ROOT / "inputs" / "checkpoints" / "body_models"
GVHMR_UTILS_DIR = GVHMR_ROOT / "hmr4d" / "utils"

DEFAULT_SMPL_DATA_DIR = RESULTS_INIT_DIR / "smpl"
LEGACY_SMPL_DATA_DIR = GVHMR_BODY_MODELS_DIR / "smpl"


def resolve_smpl_data_dir() -> Path:
    """Allow overriding SMPL assets via env var while preferring staged data."""
    env_override = os.getenv("SMPL_DATA_PATH")
    if env_override:
        return Path(env_override)

    for candidate in (
        DEFAULT_SMPL_DATA_DIR,
        PREP_SMPL_DATA_DIR,
        REPO_SMPL_DATA_DIR,
        LEGACY_SMPL_DATA_DIR,
    ):
        if candidate.exists():
            return candidate

    return LEGACY_SMPL_DATA_DIR


SMPL_DATA_DIR = resolve_smpl_data_dir()


def resolve_body_segments_dir() -> Path:
    """Locate the body segment JSON files even when staged outside the SMPL dir."""
    env_override = os.getenv("BODY_SEGMENTS_PATH")
    if env_override:
        return Path(env_override)

    candidates = (
        SMPL_DATA_DIR / "body_segments",
        SMPL_DATA_DIR.parent / "body_segments",
        PREP_DATA_DIR / "body_segments",
        REPO_SMPL_DATA_DIR / "body_segments",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[-1]


BODY_SEGMENTS_DIR = resolve_body_segments_dir()
CONTACT_IDS_PATH = BODY_SEGMENTS_DIR.parent / "CONTACT_IDS_SMPL.pt"
CONTACT_CACHE_DIR = BODY_SEGMENTS_DIR.parent / "cache_contact"

# GVHMR demo outputs (hmr4d_results.pt etc.) are staged under results/init/hmr.
HMR_RESULTS_ROOT = RESULTS_INIT_DIR / "hmr"

# TRAM-specific outputs follow the same convention; populate if/when needed.
TRAM_RESULTS_ROOT = RESULTS_INIT_DIR / "tram"