"""
Unit tests for the Transformers (and their components) of the Providence project

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import logging
from typing import Literal
from providence.nn.transformer.memory_efficient import MemoryEfficientMHA

import pytest
import torch as pt
from torchtyping import TensorType
from providence.nn.transformer import BasicAttention, DecoderBlock, EncoderBlock, MultiheadedAttention, PositionalEncoding, ReferenceProvidenceTransformer, Residual, Transformer, make_bit_masking_tensor, set_attention_axis
from torch.nn import Linear
from torch.nn.modules.activation import Softmax
from torch.nn.modules.loss import MSELoss
from providence.nn.transformer.deepmind import MaskMode, ProvidenceBertTransformer

from providence.nn.transformer.transformer import MultiheadedAttention2, MultiheadedAttention3
from providence.training import unpack_label_and_censor

log = logging.getLogger(__name__)


def construct_attention_mask(attention_input_shape: pt.Size) -> pt.Tensor:
    cum_sum = pt.cumsum(pt.eye(attention_input_shape[0]), dim=1)
    log.debug("cum_sum= %s", cum_sum)
    log.debug("cum_sum.T= %s", cum_sum.T)
    return cum_sum.T


def generate_square_subsequent_mask(sz: int) -> pt.Tensor:
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = pt.tril(pt.ones(sz, sz, dtype=pt.float))
    mask = mask.masked_fill(mask == 0, float("-inf"))
    return mask


def randint(floor: int, ceil: int) -> int:
    "return an int between floor and ceil (inclusive)"
    return pt.randint(floor, ceil + 1, (1, )).item()

@pytest.mark.transformers
class TestTransformerComponents:
    @pytest.mark.transformers
    class TestResidualConnections:
        def test_activation_agreement(self):
            """White-box test on the expected behavior of a default Residual connection"""
            l1 = Linear(10, 10)
            l2 = Linear(10, 10)

            in_ = pt.rand(5, 10)

            expected = l2(l1(in_) + in_) + in_
            result = Residual(l1, l2)(in_)
            assert pt.allclose(result, expected), "Residual module should add to every activation"

    @pytest.mark.transformers
    class TestAttentionMechanisms:
        def test_basic_attention_init(self):

            test_dim = 10

            attention = BasicAttention(test_dim)

            assert attention.embed_dim == test_dim, "Attention should encapsulate its dimensionality argument"
            assert list(attention.modules()), "Attention should have child modules"

        def test_basic_attention_with_mask_3D(self):
            print(f"{'=' * 20}Testing 3-D masking {'=' * 20}")

            # names=("time_steps", "batch_size", "features")
            test_examples_3D = pt.randn(5, 2, 7)  # something small, for interpretability

            time_steps, batch_size, embedding_dimension = test_examples_3D.shape

            attention = BasicAttention(embedding_dimension)

            test_mask = generate_square_subsequent_mask(batch_size)

            result = attention(test_examples_3D, test_examples_3D, test_examples_3D, mask=test_mask)
            assert result.shape[-1] == embedding_dimension, "Shouldn't have changed dimensionality"
            assert result.shape[0] == time_steps, "Shouldn't have generated extra rows"

            result_wo_mask = attention(
                test_examples_3D,
                test_examples_3D,
                test_examples_3D,
            )
            print("Result with mask minus result w/o mask\n%s", result_wo_mask - result)
            assert result.shape == result_wo_mask.shape
            assert (result.abs_().sum(dim=0).sum(dim=0) <= result_wo_mask.abs_().sum(dim=0).sum(dim=0)).all()

        def test_basic_attention_with_mask_3D__feature_axis(self):
            """Same as the above, but changes the attention axis"""
            test_examples_3D = pt.randn(5, 2, 7)

            time_steps, batch_size, embedding_dimension = test_examples_3D.shape

            attention = BasicAttention(embedding_dimension, attention_axis="feature")

            test_mask = generate_square_subsequent_mask(batch_size)

            result = attention(test_examples_3D, test_examples_3D, test_examples_3D, mask=test_mask)
            assert result.shape[-1] == embedding_dimension, "Shouldn't have changed dimensionality"
            assert result.shape[0] == time_steps, "Shouldn't have generated extra rows"

            result_wo_mask = attention(
                test_examples_3D,
                test_examples_3D,
                test_examples_3D,
            )
            assert result.shape == result_wo_mask.shape
            assert (result.abs_().sum(dim=0).sum(dim=0) <= result_wo_mask.abs_().sum(dim=0).sum(dim=0)).all()

        def test_multiheaded_attention_init(self):
            test_heads = randint(2, 9)
            test_dim = randint(8, 256)
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

        def test_multiheaded_attention_with_mask_3D__temporal_axis(self):
            test_examples_3D = pt.randn(5, 2, 7)

            time_steps, batch_size, embedding_dimension = test_examples_3D.shape

            attention = MultiheadedAttention(n_heads=2, embed_dim=embedding_dimension, attention_axis="temporal")

            test_mask = generate_square_subsequent_mask(batch_size)

            result = attention(test_examples_3D, test_examples_3D, test_examples_3D, mask=test_mask)
            assert result.shape[-1] == embedding_dimension, "Shouldn't have changed dimensionality"
            assert result.shape[0] == time_steps, "Shouldn't have generated extra rows"

            print(result)

            result_wo_mask = attention(
                test_examples_3D,
                test_examples_3D,
                test_examples_3D,
            )

            print(result_wo_mask)
            assert result.shape == result_wo_mask.shape
            assert (result.abs_().sum(dim=0).sum(dim=0) <= result_wo_mask.abs_().sum(dim=0).sum(dim=0)).all()

        def test_multiheaded_attention_with_mask_3D__feature_axis(self):
            test_examples_3D = pt.randn(5, 2, 7)

            time_steps, batch_size, embedding_dimension = test_examples_3D.shape

            attention = MultiheadedAttention(n_heads=2, embed_dim=embedding_dimension, attention_axis="feature")

            test_mask = generate_square_subsequent_mask(batch_size)

            result = attention(test_examples_3D, test_examples_3D, test_examples_3D, mask=test_mask)
            assert result.shape[-1] == embedding_dimension, "Shouldn't have changed dimensionality"
            assert result.shape[0] == time_steps, "Shouldn't have generated extra rows"

            result_wo_mask = attention(
                test_examples_3D,
                test_examples_3D,
                test_examples_3D,
            )
            assert result.shape == result_wo_mask.shape
            assert (result.abs_().sum(dim=0).sum(dim=0) <= result_wo_mask.abs_().sum(dim=0).sum(dim=0)).all()

    @pytest.mark.transformers
    class TestEncoderBlock:
        def test_activation(self):

            examples = pt.randn(2, 5, 40)
            test_d_model = examples.shape[-1]
            encoder = EncoderBlock(model_dimension=test_d_model)

            results = encoder(examples)
            assert results.shape == examples.shape, "Encoding shouldn't change dimensionality"

    @pytest.mark.transformers
    class TestPositionalEncoding:
        def test_activation(self):
            examples = pt.randn(2, 5, 40)
            test_d_model = examples.shape[-1]
            encoder = PositionalEncoding(d_model=test_d_model, max_len=20)

            results = encoder(examples)
            assert results.shape == examples.shape, "Encoding shouldn't change dimensionality"

    @pytest.mark.transformers
    class TestDecoderBlock:
        @pytest.fixture
        def sample_data(self):
            return pt.randn(5, 10, 20)

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
class TestTransformer:
    @pytest.fixture
    def default_instance(self):
        return Transformer()

    @pytest.fixture
    def small_instance(self):
        return Transformer(
            model_dimension=4,
            feed_forward_internal_dimension=20,
            num_heads=2,
            num_layers=1,
            positional_encoding_dim=200
        )

    @pytest.fixture
    def small_examples(self) -> TensorType["time", "devices", "features"]:
        return pt.randn(5, 2, 4)

    def attention_axis_name_to_dim(self, axis_name: Literal["feature", "temporal"]) -> int:
        if axis_name == "temporal":
            return 0
        elif axis_name == "feature":
            return -1
        else:
            raise ValueError(f"{axis_name = } is not a valid attention axis")

    def test_set_attention_axis__changes_axis(self, small_instance: Transformer):
        assert small_instance.attention_axis == "temporal"
        expected_axis = "feature"
        set_attention_axis(small_instance, expected_axis)
        assert small_instance.attention_axis == "feature"

        reassigned_modules = [m for m in small_instance.modules() if hasattr(m, "attention_axis")]
        print("Module names:", [m._get_name() for m in reassigned_modules])
        assert all(
            map(lambda m: m.attention_axis == expected_axis, reassigned_modules)
        ), "All modules should have the new attention axis"

        reassigned_attention_heads = [h for h in reassigned_modules if isinstance(h, BasicAttention)]
        assert all(
            map(
                lambda m: m.attention_dim == self.attention_axis_name_to_dim(expected_axis), reassigned_attention_heads
            )
        ), "All modules should have the new attention axis"

    def test_activation(self, small_instance: Transformer, small_examples: pt.Tensor):
        output = small_instance(small_examples)

        assert output.size() == small_examples.size(), "should be able to activate"

    def test_activation__with_mask(self, small_instance: Transformer, small_examples: pt.Tensor):
        """NOTE: this is how the masking should be done for the transformer in the temporal case"""
        test_lengths = 5, 2  # the two devices in the generated examples will only have these lengths, and we only want to consider those
        # apply mask to the example to make it look like the data that we produce in batches
        # TODO: mask everything like this, rather than with the lopsided matrix in the generate_square_subsequent_mask()
        mask_ = make_bit_masking_tensor(test_lengths, 0).unsqueeze(2)  # give an extra dimension so broadcasting works
        small_examples *= mask_

        print("small_examples")
        print(small_examples)

        output = small_instance(small_examples, encoder_mask=mask_)
        print(output)
        assert output.size() == small_examples.size(), "should be able to activate"


@pytest.mark.transformers
class TestEndToEndTraining:
    @pytest.fixture
    def training_examples(self) -> pt.Tensor:
        return pt.randn(5, 10, 30)

    @pytest.fixture
    def training_batches(self, training_examples):
        return [pt.randn_like(training_examples) for _ in range(10)]

    def test_shallow_endtoend_singlepass(self, training_examples: pt.Tensor):
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

    def test_full_scale_endtoend_singlepass(self, training_examples: pt.Tensor) -> pt.Tensor:
        """Show that you can get a similar inference"""
        test_model_dim = training_examples.shape[-1]

        model = Transformer(model_dimension=test_model_dim, feed_forward_internal_dimension=64)

        prediction = model(training_examples)

        assert prediction.shape == training_examples.shape, "We should be able to do inference"

        loss = MSELoss()(prediction, training_examples)
        loss.backward()

        assert True, "We should be able to do a backward pass"

    def test_full_scale_endtoend_singlepass_with_positional_encoding(self, training_examples: pt.Tensor) -> pt.Tensor:
        """Show that you can get a similar inference"""
        test_model_dim = training_examples.shape[-1]

        model = Transformer(
            model_dimension=test_model_dim, feed_forward_internal_dimension=64, positional_encoding_dim=200
        )

        prediction = model(training_examples)

        assert prediction.shape == training_examples.shape, "We should be able to do inference"

        loss = MSELoss()(prediction, training_examples)
        loss.backward()

        assert True, "We should be able to do a backward pass"


@pytest.mark.transformers
class TestAlternateMultiheadedAttention:
    @pytest.fixture
    def feature_dim(self) -> int:
        return 7

    @pytest.fixture
    def training_examples(self, feature_dim: int) -> pt.Tensor:
        time_steps = 2
        n_devices = 3
        return pt.randn(time_steps, n_devices, feature_dim)

    def MHA_from_Basic(self, base: BasicAttention) -> MultiheadedAttention2:
        mha = MultiheadedAttention2(n_heads=1, embed_dim=base.embed_dim)
        mha.key_projection.load_state_dict(base.k_proj.state_dict())
        mha.value_projection.load_state_dict(base.v_proj.state_dict())
        mha.query_projection.load_state_dict(base.q_proj.state_dict())
        return mha

    def test__same_attention_distribution__linear(self, training_examples: pt.Tensor, feature_dim: int):
        "Testing the linearity of the MHA against the BasicAttention"
        ba = BasicAttention(feature_dim)
        x = training_examples
        mha = self.MHA_from_Basic(ba)
        baseline = ba(x, x, x)
        result = mha(x, x, x)
        assert pt.allclose(baseline, result)

    def MHA_NEW_from_MHA_old(self, old: MultiheadedAttention) -> MultiheadedAttention2:
        mha = MultiheadedAttention2(n_heads=old.n_heads, embed_dim=old.embed_dim)
        mha.key_projection.weight = pt.nn.parameter.Parameter(
            pt.cat([ah.k_proj.weight for ah in old.attention_heads], )
        )
        mha.value_projection.weight = pt.nn.parameter.Parameter(
            pt.cat([ah.v_proj.weight for ah in old.attention_heads], )
        )
        mha.query_projection.weight = pt.nn.parameter.Parameter(
            pt.cat([ah.q_proj.weight for ah in old.attention_heads], )
        )
        # NOTE: I don't remember why I hard-coded this. Make the model less flexible...
        # maybe to make the matmul easier to pull off, and add bias in (later) manually
        if old.attention_heads[0].k_proj.bias:
            mha.key_projection.bias = pt.nn.parameter.Parameter(
                pt.cat([ah.k_proj.bias for ah in old.attention_heads], )
            )
            mha.value_projection.bias = pt.nn.parameter.Parameter(
                pt.cat([ah.v_proj.bias for ah in old.attention_heads], )
            )
            mha.query_projection.bias = pt.nn.parameter.Parameter(
                pt.cat([ah.q_proj.bias for ah in old.attention_heads], )
            )
        return mha

    def test__same_attention_distribution__multiple_heads(self, training_examples: pt.Tensor, feature_dim: int):
        "Testing whether the attention head is a linear map. Basically"
        mha = MultiheadedAttention(2, feature_dim)
        mha_new = self.MHA_NEW_from_MHA_old(mha)

        mha_new.key_projection.weight.shape, mha.attention_heads[0].k_proj.weight.shape
        # mha_new.key_projection.bias.shape, mha.attention_heads[0].k_proj.bias.shape

        mask_to_look_like_collate_outputs = make_bit_masking_tensor([2, 1, 2]).unsqueeze(2)
        x = training_examples * mask_to_look_like_collate_outputs

        baseline = mha(x, x, x, mask_to_look_like_collate_outputs)
        result = mha_new(x, x, x, mask_to_look_like_collate_outputs)

        assert baseline.shape == result.shape, "fail to match output dimensions"
        # assert pt.allclose(baseline, result)

    def test__same_attention_distribution__multiple_heads_abstracted(
        self, training_examples: pt.Tensor, feature_dim: int
    ):
        "Testing whether the attention head is a linear map. Basically"
        mha = MultiheadedAttention2(2, feature_dim)
        mha_new = MultiheadedAttention3(2, feature_dim)

        mask_to_look_like_collate_outputs = make_bit_masking_tensor([2, 1, 2]).unsqueeze(2)
        x = training_examples * mask_to_look_like_collate_outputs

        result2 = mha(x, x, x, mask_to_look_like_collate_outputs)
        result3 = mha_new(x, x, x, mask_to_look_like_collate_outputs)

        baseline = MultiheadedAttention(2, feature_dim)(x, x, x)

        assert baseline.shape == result2.shape == result3.shape, "fail to match output dimensions"

    def test__mha2__is_dropin_replacement(self, training_examples: pt.Tensor, feature_dim: int):
        transformer = Transformer(
            feature_dim,
            feed_forward_internal_dimension=8,
            num_layers=1,
            num_heads=2,
            t_attention=MultiheadedAttention2
        )
        longest = training_examples.shape[0]
        result = transformer(training_examples, make_bit_masking_tensor([longest, longest - 1, longest]).unsqueeze(2))

        assert result.shape == training_examples.shape, "Prediction output is misshapen"

    def test__mha3__is_dropin_replacement(self, training_examples: pt.Tensor, feature_dim: int):
        transformer = Transformer(
            feature_dim,
            feed_forward_internal_dimension=8,
            num_layers=1,
            num_heads=2,
            t_attention=MultiheadedAttention3
        )
        longest = training_examples.shape[0]
        result = transformer(training_examples, make_bit_masking_tensor([longest, longest - 1, longest]).unsqueeze(2))

        assert result.shape == training_examples.shape, "Prediction output is misshapen"

    def test__effcient_mha__is_dropin_replacement(self, training_examples: pt.Tensor, feature_dim: int):
        import torch as pt
        from providence.nn.transformer.vendored_memory_effecient_attention import efficient_dot_product_attention

        # Random Data (batch dimensions are not necessary)
        # using non-power-of-2 numbers to stress the point
        timesteps = 500
        entities = 30
        embedding_dim = 71
        with pt.set_grad_enabled(True):
            query = pt.rand(timesteps, entities, embedding_dim).to(pt.float32)
            key = pt.rand(timesteps, entities, embedding_dim).to(pt.float32)
            value = pt.rand(timesteps, entities, embedding_dim).to(pt.float32)
            # for casual tasks, ...
            mask = pt.rand(entities, timesteps, timesteps) > 0.5
            bias = pt.rand(entities, timesteps, 1).to(pt.float32) / 100

        out = efficient_dot_product_attention(query, key, value, mask, bias)
        assert out.shape == query.shape, "Attention should not change the shape, only highlight what is of concern"

        transformer = Transformer(
            feature_dim,
            feed_forward_internal_dimension=8,
            num_layers=1,
            num_heads=2,
            t_attention=MemoryEfficientMHA,
        )
        result = transformer(training_examples)

        assert result.shape == training_examples.shape, "Prediction output is misshapen"


class TestReferenceTransformer:
    @classmethod
    def helper_small_model(cls, feature_dimension: int):

        return ReferenceProvidenceTransformer(
            model_dimension=feature_dimension, hidden_size=12, n_attention_heads=2, positional_encoding_dimension=10
        )

    def test_initialization_and_inference(self):
        examples = pt.randn(3, 2, 12)  # feature dim must be divisible by n_heads parameter (default: 4)
        model = ReferenceProvidenceTransformer(model_dimension=examples.size(2))

        model.forward(examples, [3, 1])

    def test_masking_inference(self):
        examples = pt.randn(4, 2, 4)
        model = self.helper_small_model(examples.size(2))

        example_lengths = [3, 4]
        # NOTE: uncomment to see the progression toward diagnosing the needs of a mask (not well explained in the papers)
        # mask = pt.triu(pt.ones(3, 3)).to(bool)
        # mask_offset = (mask.to(int) - pt.eye(3)).to(bool)

        with pt.no_grad():
            print("--" * 40)
            print("\nresult =")
            model.forward(examples, example_lengths, encoder_mask=None, decoder_mask=None)

            # print("--"*40)
            # print("\nresult_enc_mask =")
            # model.forward(examples, example_lengths, encoder_mask=mask, decoder_mask=None)

            # print("--"*40)
            # print("\nresult_dec_mask =")
            # model.forward(examples, example_lengths, encoder_mask=None, decoder_mask=mask)

            # print("--"*40)
            # print("\nresult_full_mask =")
            # model.forward(examples, example_lengths, encoder_mask=mask, decoder_mask=mask)

            # print("--"*40)
            # print("\nresult_offset_mask =")
            # model.forward(examples, example_lengths, encoder_mask=mask, decoder_mask=mask_offset)

            print("--" * 40)
            print("\nresult_dec_offset_mask =")
            model.forward(examples, example_lengths, encoder_mask=None, decoder_mask=True)


# TODO: parameterize this test to a BERT model and Encoder-Decoder-based model, because it's the *exact* same and copy-paste tests suck.
class TestDeepMindTransformer_BERT:
    import providence.nn.transformer.deepmind as dm

    @classmethod
    def helper_small_model(cls, feature_dimension: int):

        return ProvidenceBertTransformer(
            n_heads=2,
            n_layers=2,
            n_features_in=feature_dimension,
            n_embedding=10,
            mask_mode=MaskMode.backward_only,
            max_seq_len=4,
            att_inner_dim=7,  # small prime for debugging
            ff_dim=9,  # odd number for debugging
            final_dim=11,  # still small prime for debugging
        )

    def test_initialization_and_inference(self):
        examples = pt.randn(3, 2, 12)
        model = self.helper_small_model(examples.size(2))
        print(f"{model.default = }")

        out = model.forward(examples, pt.tensor([3, 1]))
        assert out is not None, "Inference failed to return valid result."

    def test_backward_pass(self):
        n_test_features = 6
        model = self.helper_small_model(n_test_features)
        # remember Providence examples are (sequence, item or entity, feature)
        # rather than (batch, time step, feature).
        # A simple transpose will move you between the two, but that's computationally heavy some times.
        from providence.distributions import Weibull
        from providence.loss import DiscreteWeibullLoss

        self.dm._DEBUG = False
        examples = pt.randn(model.transformer.max_seq_len, 2, 6)
        targets = pt.tensor(
            [
                [[3, 4], [1, 1]],
                [[2, 3], [1, 1]],
                [[1, 2], [1, 1]],
                [[0, 1], [0, 1]],
            ]
        )  # targets.size() == Size(4, 2, 2)
        print(f"{targets.size() = }")

        example_lengths = pt.tensor([3, model.transformer.max_seq_len], dtype=pt.long)

        alpha_tens, beta_tens = model.forward(examples, example_lengths)

        print(f"{alpha_tens.size() = }")

        y, censor = unpack_label_and_censor(targets)

        loss_fn = DiscreteWeibullLoss()
        loss = loss_fn(Weibull.Params(alpha_tens, beta_tens), y, censor, example_lengths)
        loss.backward()

    def setup_inference_test(self):
        n_test_features = 6
        model = self.helper_small_model(n_test_features)
        # remember Providence examples are (sequence, item or entity, feature)
        # rather than (batch, time step, feature).
        # A simple transpose will move you between the two, but that's computationally heavy some times.
        self.dm._DEBUG = False

        examples = pt.randn(model.transformer.max_seq_len, 2, 6)
        return model, examples

    def test_inference(self):
        model, examples = self.setup_inference_test()

        example_lengths = pt.tensor([3, model.transformer.max_seq_len])

        with pt.no_grad():
            model.forward(examples, example_lengths)

    def test_no_grad_respected(self):
        from typing import Tuple
        model, examples = self.setup_inference_test()

        example_lengths = pt.tensor([3, model.transformer.max_seq_len])

        with pt.no_grad():
            result: Tuple[pt.Tensor, pt.Tensor] = model.forward(examples, example_lengths)

        assert result[0].grad is None, "no_grad() isn't respected properly"
