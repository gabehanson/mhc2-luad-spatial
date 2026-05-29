"""
ceiba — spatial transcriptomics methods
https://github.com/gabehanson/mhc2-luad-spatial
"""

__version__ = "0.1.0"
__author__ = "Gabriel Hanson"

from .stats_utils import (
    run_stats,
    run_mixed_effects,
    ciita_expr_by_s100p_strata_per_sample,
    ciita_expr_cell_level_tests,
)
