#!/usr/bin/env python3
"""BIMS-LEGAL configuration: single entry for legal consultation memory runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class BIMSLegalConfig:
    """Canonical switches for BIMS-LEGAL experiments (paper-facing names)."""

    name: str = "BIMS-LEGAL"
    # Index
    use_pq: bool = False          # O1: FlatIP when False
    # Episodic Soft O2
    session_expand: bool = True
    session_coherence_beta: float = 0.98
    session_first_rerank: bool = True
    # Must stay False for fair Soft O2 (no hard score copy path elsewhere)
    hard_score_copy: bool = False
    # Optional pathways
    exact_match_boost: float = 0.0
    enable_temporal: bool = True
    enable_associative: bool = True

    def apply(self, mgr: Any) -> None:
        mgr.vector_store.use_pq = self.use_pq
        mgr._session_expand = self.session_expand
        mgr._session_coherence = float(self.session_coherence_beta)
        mgr._session_first_rerank = self.session_first_rerank
        mgr._exact_match_boost = float(self.exact_match_boost)
        # temporal/assoc toggles depend on manager flags if present
        if hasattr(mgr, "_disable_temporal"):
            mgr._disable_temporal = not self.enable_temporal
        if hasattr(mgr, "_disable_associative"):
            mgr._disable_associative = not self.enable_associative

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Named ablations for the completeness matrix
ABLATIONS = {
    "bims_legal_full": BIMSLegalConfig(),
    "w_o_o2": BIMSLegalConfig(session_expand=False, session_first_rerank=False),
    "w_o_o1_pq": BIMSLegalConfig(use_pq=True),
    "beta_050": BIMSLegalConfig(session_coherence_beta=0.50),
    "beta_090": BIMSLegalConfig(session_coherence_beta=0.90),
    "beta_098": BIMSLegalConfig(session_coherence_beta=0.98),
    "beta_100": BIMSLegalConfig(session_coherence_beta=1.00),
}


BASELINES_REQUIRED = [
    "dense_flat",
    "bims_legal_full",
    "parent_hydrate",
    "session_max",
    "joint_qa",
    "bm25_turn",
    "bm25_joint",
    "shuffled_sid",
]
