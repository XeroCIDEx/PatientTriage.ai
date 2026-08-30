"""
version.py
==========
Single source of truth for version strings, so model outputs and audit
log entries always agree on what produced them (spec Section 19/40:
every prediction and audit entry should carry model_version and
preprocessing_version).

Bump MODEL2_SELECTION_VERSION whenever model2_selection.py is re-run and
a different candidate wins — the comparison result and selection date
should be recorded in models/model2_candidate_comparison.json alongside
this string.
"""

MODEL_VERSION = "patienttriage-prototype-v0.2"
PREPROCESSING_VERSION = "preprocessing-v0.2-age-aware"
MODEL2_SELECTION = "MLP_2layer (evidence-based, see models/model2_candidate_comparison.json)"
