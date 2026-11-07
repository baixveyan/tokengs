"""GeometryGS: spatial Gaussian tokens with a reusable TokenGS scene backbone.

The package reuses the local TokenGS research copy (training loop, data, rendering, evaluation) without modifying it.
Registers the legacy `tokengs_sfm`, new `geometrygs_spatial`, and the SfM-aware data provider.
"""

import os
import sys
from pathlib import Path

GEOMETRYGS_ROOT = Path(__file__).resolve().parents[1]
_default_tokengs = GEOMETRYGS_ROOT.parent / "TokenGS"
if not _default_tokengs.exists() and (GEOMETRYGS_ROOT.parent / "tokengs").exists():
    _default_tokengs = GEOMETRYGS_ROOT.parent / "tokengs"
TOKENGS_ROOT = Path(os.environ.get("TOKENGS_ROOT", _default_tokengs)).resolve()
for _p in (TOKENGS_ROOT, TOKENGS_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from geometrygs import data as _data  # noqa: E402,F401  (patches split_data / random_reflect / Provider)
from geometrygs.model import SfmTokenGS  # noqa: E402
from tokengs.models import model_registry  # noqa: E402

model_registry["tokengs_sfm"] = SfmTokenGS

from geometrygs.spatial_model import SpatialTokenGS  # noqa: E402

model_registry["geometrygs_spatial"] = SpatialTokenGS
