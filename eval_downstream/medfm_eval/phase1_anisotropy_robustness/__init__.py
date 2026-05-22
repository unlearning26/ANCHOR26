# Phase 1: Spacing/Anisotropy Robustness Evaluation
# Part of the 3D Medical Foundation Model Evaluation Framework

"""
Phase 1 evaluates whether 3D medical foundation model representations
are invariant to voxel spacing and anisotropy variations.

Primary tasks:
- representation_geometry: CKA and embedding geometry across spacing bins
- spacing_readout: anisotropy prediction from frozen embeddings
- semantic_readout: label probing within spacing regimes
- cross_bin_semantic_transfer: train on one spacing regime, test on another
- controlled_spacing_perturbation: same-volume resampling intervention

Legacy aliases retained for compatibility:
- Track A -> representation_geometry
- Track B -> spacing_readout / semantic_readout
- Setting A -> observational_bin_analysis
- Setting B -> controlled_spacing_perturbation
"""

__version__ = "0.1.0"
