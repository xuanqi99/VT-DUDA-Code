"""Dataset utilities included with the MCC source subset.

The trainer's optional image-list mode handles Office-31, Office-Home, and
VisDA-2017 without the unbundled upstream dataset modules.
"""

from .imagelist import ImageList

__all__ = ['ImageList']
