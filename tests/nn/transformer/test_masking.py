"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import pytest
import torch as pt

from providence.nn.transformer.utils import make_bit_masking_tensor


@pytest.mark.transformers
class TestMasking:
    """This test articulates the desired results of masking, as described in `test_no_offset()`.
    Effectively, we want to be able to mask on the outer-most dimension.
    The print statements depict the Tensors' lifecycle, through input to the ideal output post-masking."""

    def test_no_offset(self):
        """
        I have a 3-D tensor (sequence, device, features) and a length definition of each sequence,
        with everything shorter than the longeset `sequence`.

        Masking:
            I want my mask to be the stake of the padded squares that are (sequence-length x sequence length)
            squares, broadcast via hadamard product against the input feature tensor.
        Thus, after "masking", the masked tensor will have the last time step for all inputs zero'd out.
        And the mask will be applied as a tensor of (sequence_i - 1, device)
        """
        longest_sequence = 5
        n_devices = 3
        n_features = 2

        tensor_in = pt.randn(longest_sequence, n_devices, n_features)
        lengths = [5, 1, 3]

        # if I can reproduce the padding operation as a mathematical operation, I've definitely got a good mask
        # NOTE: this is the torch-implemented padding that should be much faster than our manual padding (even if we MIGHT have branch prediction)
        desired_mask = pt.nn.utils.rnn.pad_sequence([
            pt.ones(length_) for length_ in lengths
        ], batch_first=False, padding_value=0)

        print(f"Shapes: {tensor_in.shape = } {desired_mask.shape = }")
        print(f"{desired_mask.shape = }")
        print(f"{desired_mask.unsqueeze(2).shape = }")
        masked = (tensor_in * desired_mask.unsqueeze(2))

        assert masked.shape == tensor_in.shape, "Masking should be consistent in shape"
        
        # we should expect all cells that aren't up to the longest-sequence-index to be zero
        expected_zero_count = sum([(longest_sequence - curr_length) * tensor_in.size(2) for curr_length in lengths])
        zero_count = (pt.zeros_like(tensor_in) == masked).sum().sum().item()
        assert zero_count == expected_zero_count, "Should have zeroed out the elements that weren't supposed to be zeros"

    def test_one_index__future_mask(self):
        """
        This test outlines how one would going about a mask in the game-dev sense, of every element having a corresponding bit to indicate
        whether or not to care about that datum in some matrix.
        This is not what goes on internal to the Transformer, but is the easiest to make sure it is correct.
        """
        longest_sequence = 5
        n_devices = 3
        n_features = 2

        tensor_in = pt.randn(longest_sequence, n_devices, n_features)
        print(f"{tensor_in = }")
        lengths = [5, 2, 3]


        # need a faster way to do this, but masked_select and masked_fill definitely aren't it; they don't deal with indices.h
        # follow-up: this is fastest on CPU. Testing on GPU...
        # NOTE(stephen): on a GPU using a dynamically instantiated pt.tile()'d pt.arange()'s and comparing to the offset
        # is about 1000x faster... So we're taking the logical
        # for i, mask_len in enumerate(masked_lengths):
        #     keep_bits[i][mask_len:] = 0
        #
        # desired_mask = pt.nn.utils.rnn.pad_sequence(keep_bits, batch_first=False, padding_value=0).contiguous()
        # and using torch to implement the masking.
        mask_offset = 1
        desired_mask = make_bit_masking_tensor(lengths, mask_offset)

        print(f"Shapes: {tensor_in.shape = } {desired_mask.shape = }")
        print(f"{desired_mask.shape = }")
        print(f"{desired_mask.unsqueeze(2).shape = }")
        masked = (tensor_in * desired_mask.unsqueeze(2))
        print(f"{masked = }")

        assert masked.shape == tensor_in.shape, "Masking should be consistent in shape"
        masked_lengths = [length_ - mask_offset for length_ in lengths]

        # we should expect all cells that aren't up to the longest-sequence-index to be zero
        expected_zero_count = sum([(longest_sequence - curr_length) * tensor_in.size(2) for curr_length in masked_lengths])
        zero_count = (pt.zeros_like(tensor_in) == masked).sum().sum().item()
        assert zero_count == expected_zero_count, "Should have zeroed out the elements that weren't supposed to be zeros"
