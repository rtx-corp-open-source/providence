import logging

import pytest
import torch
from providence.model.blocks.transformer import (BasicAttention, DecoderBlock,
                                                 EncoderBlock,
                                                 MultiheadedAttention,
                                                 Residual, Transformer)
from torch.nn import Linear
from torch.nn.modules.activation import Softmax
from torch.nn.modules.loss import MSELoss

log = logging.getLogger(__name__)


@pytest.mark.transformers
class TestResidualConnections:
    def test_activation_agreement(self):
        """White-box test on the expected behavior of a default Residual connection"""
        l1 = Linear(10, 10)
        l2 = Linear(10, 10)

        in_ = torch.rand(5, 10)

        expected = l2(l1(in_) + in_) + in_
        result = Residual(l1, l2)(in_)
        assert torch.allclose(result, expected), "Residual module should add to every activation"


def construct_attention_mask(attention_input_shape: torch.Size) -> torch.Tensor:
    cum_sum = torch.cumsum(torch.eye(attention_input_shape[0]), dim=1)
    log.debug("cum_sum= %s", cum_sum)
    log.debug("cum_sum.T= %s", cum_sum.T)
    return cum_sum.T


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


@pytest.mark.transformers
class TestAttentionMechanisms:
    def test_basic_attention_init(self):

        test_dim = 10

        attention = BasicAttention(test_dim)

        assert attention.embed_dim == test_dim, "Attention should encapsulate its dimensionality argument"
        assert list(attention.modules()), "Attention should have child modules"

    def test_basic_attention_with_mask_3D(self):
        log.info(f"{'=' * 20}Testing 3-D masking {'=' * 20}")

        # names=("time_steps", "batch_size", "features")
        test_examples_3D = torch.randn(5, 2, 7)  # something small, for interpretability

        time_steps = test_examples_3D.shape[0]
        batch_size = test_examples_3D.shape[1]
        embedding_dimension = test_examples_3D.shape[-1]
        attention = BasicAttention(embedding_dimension)

        # test_mask = construct_attention_mask(torch.Size([time_steps]))
        test_mask = generate_square_subsequent_mask(batch_size)

        result = attention(test_examples_3D, test_examples_3D, test_examples_3D, mask=test_mask)
        assert result.shape[-1] == embedding_dimension, "Shouldn't have changed dimensionality"
        assert result.shape[0] == time_steps, "Shouldn't have generated extra rows"

        result_wo_mask = attention(
            test_examples_3D,
            test_examples_3D,
            test_examples_3D,
        )
        log.info("Result with mask minus result w/o mask\n%s", result_wo_mask - result)
        assert result.shape == result_wo_mask.shape

    def test_multiheaded_attention_init(self):
        test_heads = torch.randint(2, 10, (1,)).item()
        test_dim = torch.randint(8, 256, (1,)).item()
        attention = MultiheadedAttention(test_heads, test_dim)

        assert attention.n_heads == test_heads, "MHA should encapsulate number of attention heads"
        assert attention.embed_dim == test_dim, "MHA should encapsulate width of the embedding dimension"
        assert attention.attention_concat_dim == test_dim * test_heads, "MHA should encapsulate the scale for merging multiple heads"
        assert list(attention.modules()), "MHA should have multiple child modules"

    def test_multiheaded_attention_init_validation(self):
        # attention head validation
        with pytest.raises(ValueError, match=".*more than one.*") as ex:
            MultiheadedAttention(n_heads=1, embed_dim=128)
        with pytest.raises(ValueError, match=".*integer.*") as ex:
            MultiheadedAttention(n_heads=1.5, embed_dim=128)

        # embedding validation
        with pytest.raises(ValueError, match=".*integer.*") as ex:
            MultiheadedAttention(n_heads=2, embed_dim=12.3)
        with pytest.raises(ValueError, match="integer") as ex:
            MultiheadedAttention(n_heads=5, embed_dim=0.1)


@pytest.mark.transformers
class TestEncoderBlock:
    def test_activation(self):

        examples = torch.randn(2, 5, 40)
        test_d_model = examples.shape[-1]
        encoder = EncoderBlock(model_dimension=test_d_model)

        results = encoder(examples)
        assert results.shape == examples.shape, "Encoding shouldn't change dimensionality"


@pytest.mark.transformers
class TestDecoderBlock:
    @pytest.fixture
    def sample_data(self):
        return torch.randn(5, 10, 20)

    @pytest.fixture
    def sample_memory(self, sample_data):
        return Softmax(dim=-1)(sample_data)

    def test_init(self):
        """Test to make sure the object construction goes smoothly. Parameters should be set and the module list should be non-empty"""
        test_dim = 24

        decoder = DecoderBlock(model_dimension=test_dim)

        assert test_dim == decoder.d_model, "model dimension should match the input"
        assert list(decoder.modules()), "The DecoderBlock should have child modules"

    def test_activation(self, sample_data, sample_memory):

        decoder = DecoderBlock(model_dimension=sample_data.shape[-1])

        result = decoder(sample_data, sample_memory)

        assert result.shape == sample_data.shape, "Decoding shouldn't change the shape"


@pytest.mark.transformers
class TestEndToEndTraining:
    @pytest.fixture
    def training_examples(self) -> torch.Tensor:
        return torch.randn(5, 10, 30)

    @pytest.fixture
    def training_batches(self, training_examples):
        return [torch.randn_like(training_examples) for _ in range(10)]

    def test_shallow_endtoend_singlepass(self, training_examples: torch.Tensor):
        """Show that you can execute a forward and backward pass"""
        test_model_dim = training_examples.shape[-1]

        encoder = EncoderBlock(model_dimension=test_model_dim, feed_forward_internal_dimension=64)
        decoder = DecoderBlock(model_dimension=test_model_dim, feed_forward_internal_dimension=64)

        # naked forward and backward pass
        memory = encoder(training_examples)
        prediction = decoder(training_examples, memory)

        assert prediction.shape == training_examples.shape

        loss = MSELoss()(prediction, training_examples)
        loss.backward()

        assert True, "We should be able to do a backward pass"

    def test_full_scale_endtoend_singlepass(self, training_examples: torch.Tensor) -> torch.Tensor:
        """Show that you can get a similar inference"""
        test_model_dim = training_examples.shape[-1]

        model = Transformer(model_dimension=test_model_dim, feed_forward_internal_dimension=64)

        prediction = model(training_examples)

        assert prediction.shape == training_examples.shape, "We should be able to do inference"

        loss = MSELoss()(prediction, training_examples)
        loss.backward()

        assert True, "We should be able to do a backward pass"
