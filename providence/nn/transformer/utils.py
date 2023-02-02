"""
Utilities for the ProvidenceTransformer and related types

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import List

import torch as pt
from torchtyping import TensorType

def make_bit_masking_tensor(lengths: List[int], mask_offset=0) -> TensorType["max(lengths)", "len(lengths)"]:
    """
    An implementation of multiplicative masking, which will directly mask-out values which aren't actually part of the sequence.
    This is just the sequence-length mask.
    NOTE: We may still need another mask for the auto-regressive nature of the Transformer, but because of how the timesteps are implemented, we've actually seen similar behavior without it.
    """
    # convert the list of lengths into a matrix of dimension (max(lengths) x len(lengths))
    lengths_to_mask = pt.arange(max(lengths)).unsqueeze(0).T.tile(len(lengths))
    # apply the offset to make reference lengths
    to_mask_lengths = pt.tensor(lengths) - mask_offset
    # convert to bits that are the desired points to consider
    mask = lengths_to_mask < to_mask_lengths
    return mask