"""
Accessors for frequently used data within an :class:`~anndata.AnnData`.
"""

# expose the API
from ._counts import get_count_data as count_data
from ._meta import get_meta as meta