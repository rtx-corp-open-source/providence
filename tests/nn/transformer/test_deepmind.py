"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import pytest
import torch as pt
from jaxtyping import Float

import providence.nn.transformer.deepmind as dm_module
from providence.nn.transformer.deepmind import AttentionHead
from providence.nn.transformer.deepmind import CausalSelfAttention
from providence.nn.transformer.deepmind import CSAConfig
from providence.nn.transformer.deepmind import Decoder
from providence.nn.transformer.deepmind import EDTransformer
from providence.nn.transformer.deepmind import Encoder
from providence.nn.transformer.deepmind import ETransformer
from providence.nn.transformer.deepmind import MaskMode
from providence.nn.transformer.deepmind import MHAttention

# Tensor typing
batch = None
time = None
features = None


def reverse_tensor(examples: pt.Tensor):
    idx_tens = pt.LongTensor(list(range(examples.size(0) - 1, -1, -1))).view(examples.size(0), 1, 1)
    reversed = examples.scatter(0, idx_tens, examples)
    return reversed


class TestDMTransformerComponents:
    class TestAttentionHead:
        def test_init(self):
            n_test_features = 4
            AttentionHead(
                n_test_features,
                n_test_features,
                8,
                mask=MaskMode.unmasked,
                inner_att_dim=6,
            )

        def test_forward(self):
            n_test_features = 4
            n_test_samples = 3
            n_time_steps = 8
            test_head = AttentionHead(
                n_test_features,
                n_test_features,
                att_out_dim=11,
                mask=MaskMode.unmasked,
                inner_att_dim=6,
            )

            test_examples: Float[
                pt.Tensor,
                "batch time features",
            ] = pt.randn(n_test_samples, n_time_steps, n_test_features)
            test_example = test_examples[0]

            # dm_module._DEBUG = False
            attention_out = test_head.forward(test_examples, test_examples)
            assert attention_out.size() == pt.Size([n_test_samples, n_time_steps, test_head.att_out_dim])

            test_head.forward(test_example, test_example)
            test_head.forward(test_example.unsqueeze(0), test_example.unsqueeze(0))
            assert True, "We should be able to get this far..."

    class TestMultiheadAttention:
        def test_init(self):
            n_test_features = 3
            MHAttention(
                2,
                n_test_features,
                n_test_features,
                16,
                mask_mode=MaskMode.backward_only,
                att_inner_dim=6,
            )

        def test_forward_default_initialization(self):
            n_test_features = 5
            n_test_samples = 3
            n_time_steps = 8
            test_head = MHAttention(
                2,
                n_test_features,
                n_test_features,
                att_out_dim=11,
                mask_mode=MaskMode.unmasked,
                att_inner_dim=6,
            )

            test_examples: Float[
                pt.Tensor,
                "batch time features",
            ] = pt.randn(n_test_samples, n_time_steps, n_test_features)
            test_example = test_examples[0]

            dm_module._DEBUG = False
            attention_out = test_head.forward(test_examples, test_examples)
            assert attention_out.size() == pt.Size([n_test_samples, n_time_steps, test_head.output_dim])

            test_head.forward(test_example, test_example)
            test_head.forward(test_example.unsqueeze(0), test_example.unsqueeze(0))
            assert True, "We should be able to get this far..."

        @pytest.mark.parametrize("n_heads", [2, 3, 4, 5, 6])
        def test_forward__specific_output_dim__iterating_n_heads(self, n_heads: int):
            n_test_features = 5
            n_test_samples = 3
            n_time_steps = 8
            test_head = MHAttention(
                n_heads,
                n_test_features,
                n_test_features,
                att_out_dim=11,
                mask_mode=MaskMode.unmasked,
                att_inner_dim=6,
                output_dim=2,
            )

            test_examples: Float[
                pt.Tensor,
                "batch time features",
            ] = pt.randn(n_test_samples, n_time_steps, n_test_features)
            test_example = test_examples[0]

            dm_module._DEBUG = False
            attention_out = test_head.forward(test_examples, test_examples)
            assert test_head.output_dim is not None
            assert attention_out.size() == pt.Size([n_test_samples, n_time_steps, test_head.output_dim])

            test_head.forward(test_example, test_example)
            test_head.forward(test_example.unsqueeze(0), test_example.unsqueeze(0))
            assert True, "We should be able to get this far..."

    class TestCausalSelfAttention:
        def test_init(self):
            config = CSAConfig(block_size=512, n_head=2, n_embd=64)
            CausalSelfAttention(config)

        @pytest.mark.parametrize("n_heads", [2, 3, 4, 5, 6])
        def test_forward__specific_output_dim__iterating_n_heads(self, n_heads: int):
            n_test_features = 5
            n_test_samples = 3
            n_time_steps = 8
            test_head = CausalSelfAttention(CSAConfig(block_size=n_time_steps, n_head=n_heads, n_embd=60))

            test_examples: Float[
                pt.Tensor,
                "batch time features",
            ] = pt.randn(n_test_samples, n_time_steps, n_test_features)
            # (S, T, F) * (embedd, F) -> (S, T, embed)
            test_examples = pt.nn.functional.linear(
                test_examples, pt.randn(test_head.n_embd, n_test_features)
            )  # pt.randn(test_head.n_embd, n_test_features) @ test_examples

            dm_module._DEBUG = False
            attention_out = test_head.forward(test_examples)
            assert attention_out.size() == pt.Size([n_test_samples, n_time_steps, test_head.output_dim])

            test_example = test_examples[0]
            test_head.forward(test_example.unsqueeze(0))
            assert True, "We should be able to get this far..."

    class TestDecoder:
        def test_init(self):
            n_test_features = 5
            n_heads = 2
            Decoder(n_heads, n_test_features, att_inner_dim=6)

        @pytest.mark.parametrize("n_heads", [2, 3, 4, 5, 6])
        def test_forward(self, n_heads):
            n_test_features = 5
            decoder = Decoder(n_heads, n_test_features, att_inner_dim=6)

            n_test_samples = 3
            n_time_steps = 8
            examples = pt.randn(n_test_samples, n_time_steps, n_test_features)
            reversed = reverse_tensor(examples)
            decoder(examples, reversed)

    class TestEncoder:
        def test_init(self):
            n_test_features = 5
            n_heads = 2
            Encoder(n_heads, n_test_features, mask_mode=MaskMode.unmasked, att_inner_dim=6)

        @pytest.mark.parametrize("n_heads", [2, 3, 4, 5, 6])
        def test_forward(self, n_heads):
            n_test_features = 5
            encoder = Encoder(
                n_heads,
                n_test_features,
                mask_mode=MaskMode.backward_only,
                att_inner_dim=11,
            )

            n_test_samples = 3
            n_time_steps = 8
            examples = pt.randn(n_test_samples, n_time_steps, n_test_features)
            encoder(examples)

    class TestEncoderDecoderTransformer:
        def test_init(self):
            n_test_features = 5
            n_heads = 2
            EDTransformer(
                n_heads,
                2,
                n_test_features,
                20,
                mask_mode=MaskMode.unmasked,
                max_seq_len=10,
                att_inner_dim=6,
            )

        @pytest.mark.parametrize("n_heads", [2, 3, 4, 5, 6])
        def test_forward(self, n_heads):
            n_test_features = 5
            enc_dec = EDTransformer(
                n_heads,
                2,
                n_test_features,
                20,
                mask_mode=MaskMode.unmasked,
                max_seq_len=10,
                att_inner_dim=60,
            )

            n_test_samples = 3
            n_time_steps = 8
            examples = pt.randn(n_test_samples, n_time_steps, n_test_features)
            lengths = pt.LongTensor([7, 1, 6])
            sequence_encoded = enc_dec(examples, examples, lengths, lengths)
            return sequence_encoded

        @pytest.mark.parametrize("n_heads", [2, 3, 4, 5, 6])
        def test_backward(self, n_heads):
            pt.autograd.set_detect_anomaly(True)
            loss_fn = pt.nn.CrossEntropyLoss()
            sequence_encoded = self.test_forward(n_heads)

            correct = pt.randn_like(sequence_encoded).softmax(dim=1).detach()
            loss_fn(sequence_encoded, correct).backward()

    class TestEncoderOnlyTransformer:
        def test_init(self):
            n_test_features = 5
            n_heads = 2
            ETransformer(
                n_heads,
                2,
                n_test_features,
                20,
                mask_mode=MaskMode.unmasked,
                max_seq_len=10,
                att_inner_dim=6,
            )

        @pytest.mark.parametrize("n_heads", [2, 3, 4, 5, 6])
        def test_forward(self, n_heads):
            n_test_features = 5
            encoder = ETransformer(
                n_heads,
                2,
                n_test_features,
                20,
                mask_mode=MaskMode.unmasked,
                max_seq_len=10,
                att_inner_dim=6,
            )

            n_test_samples = 3
            n_time_steps = 8
            examples = pt.randn(n_test_samples, n_time_steps, n_test_features)
            lengths = pt.LongTensor([7, 1, 6])
            sequence_encoded = encoder(examples, lengths)
            return sequence_encoded

        @pytest.mark.parametrize("n_heads", [2, 3, 4, 5, 6])
        def test_backward(self, n_heads):
            pt.autograd.set_detect_anomaly(True)
            loss_fn = pt.nn.CrossEntropyLoss()
            sequence_encoded = self.test_forward(n_heads)

            correct = pt.randn_like(sequence_encoded).softmax(dim=1).detach()
            loss_fn(sequence_encoded, correct).backward()
