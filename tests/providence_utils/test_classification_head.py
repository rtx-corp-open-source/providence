"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import torch as pt

from providence.nn import ProvidenceGRU
from providence.nn import ProvidenceTransformer
from providence.nn.rnn import providence_rnn_infer
from providence.nn.transformer.transformer import MultiheadedAttention2
from providence.nn.transformer.transformer import MultiheadedAttention3
from providence.nn.transformer.utils import make_bit_masking_tensor
from providence_utils.model_output_heads import ClassificationHead


def test_classification_classes_determines_output_shape():
    m = ClassificationHead(n_classes=4)
    examples = pt.randn(5, 11)

    output = m(examples)

    assert output.size(-1) == m.n_classes


def test_classification_probability_axioms_hold():
    "Check that the first two axioms from {https://en.wikipedia.org/wiki/Probability_axioms} hold"
    m = ClassificationHead(n_classes=4)
    examples = pt.randn(1, 11)

    output = m(examples)[0]

    assert (output >= 0).all(), "Probabilities should all be non-negative"
    assert pt.allclose(pt.tensor(1.0), output.sum()), "Probabilities should sum to 1"


def test_classification_probability_axioms_hold__many_examples():
    "Check that the first two axioms from {https://en.wikipedia.org/wiki/Probability_axioms} hold"
    m = ClassificationHead(n_classes=4)
    examples = pt.randn(5, 11)

    output = m(examples)

    assert_quasiprobability_axioms(output)

    true = pt.tensor([0, 0, 1.0, 0]).tile(examples.size(0), 1)  # go from Size((4,)) to Size((5, 4))
    pt.nn.CrossEntropyLoss()(output, true).backward()
    assert True, "Should be able to do a backward pass"


def assert_quasiprobability_axioms(output):
    assert (output >= 0).all().all(), "Probabilities should all be non-negative"
    assert pt.allclose(pt.tensor(1.0), output.sum(dim=-1)), "Probabilities should sum to 1"


def test_demo_classification_head_in_simple_model():
    example_features = 11
    m = pt.nn.Sequential(pt.nn.Linear(example_features, 5), ClassificationHead(n_classes=3))

    n_examples = 10
    examples = pt.randn(n_examples, example_features)

    output = m(examples)

    assert_quasiprobability_axioms(output)


def helper_test_binary_labels(examples, n_classes: int):
    true = pt.zeros(n_classes)
    index = pt.randint(n_classes, (1,))
    true[index] = 1
    true = true.tile(examples.size(1), 1)
    return true


class TestRawOutputBasedClassifier:
    def test_demo_classification_head_in_providence_rnn_model(self):
        example_features = 3
        m = ProvidenceGRU(example_features, hidden_size=10, num_layers=1)

        x_lengths = sorted([12, 5, 6, 9, 3], reverse=True)  # randomly generated.
        examples = pt.randn(max(x_lengths), 5, example_features)  # max time steps, devices, features

        alpha_beta_tuple = m(examples, x_lengths)
        rnn_outputs = pt.cat(alpha_beta_tuple, dim=-1)

        classifier = ClassificationHead(n_classes=4)

        predictions = classifier(rnn_outputs)
        assert predictions.size() == pt.Size(examples.shape[1:2] + (classifier.n_classes,))
        assert_quasiprobability_axioms(predictions)

    def test_demo_classification_head_in_providence_transformer_model(self):
        example_features = 3
        m = ProvidenceTransformer(
            example_features,
            hidden_size=10,
            n_layers=1,
            n_attention_heads=1,
            t_attention=MultiheadedAttention3,
        )

        x_lengths = [12, 5, 6, 9, 3]  # randomly generated.
        examples = pt.randn(max(x_lengths), 5, example_features)  # max time steps, devices, features

        alpha_beta_tuple = m(examples, x_lengths, encoder_mask=True)
        transformer_outputs = pt.cat(alpha_beta_tuple, dim=-1)

        classifier = ClassificationHead(n_classes=4)

        predictions = classifier(transformer_outputs)
        assert predictions.size() == pt.Size(examples.shape[1:2] + (classifier.n_classes,))
        assert_quasiprobability_axioms(predictions)


class TestEmbeddingBasedClassifier:
    def test_demo_classification_head_in_providence_rnn_model(self):
        example_features = 3
        m = ProvidenceGRU(example_features, hidden_size=10, num_layers=1)

        x_lengths = [12, 5, 6, 9, 3]  # randomly generated.
        examples = pt.randn(max(x_lengths), 5, example_features)  # max time steps, devices, features

        rnn_outputs = providence_rnn_infer(m.rnn, examples, x_lengths)

        classifier = ClassificationHead(n_classes=4)

        predictions = classifier(rnn_outputs)
        assert predictions.size() == pt.Size(examples.shape[1:2] + (classifier.n_classes,))
        assert_quasiprobability_axioms(predictions)

        true = helper_test_binary_labels(examples, classifier.n_classes)
        losses = pt.nn.CrossEntropyLoss()(predictions, true)
        losses.backward()
        assert True, "Should be able to perform a backwards pass"

    def test_demo_classification_head_in_providence_transformer_model(self):
        example_features = 3
        m = ProvidenceTransformer(
            example_features,
            hidden_size=10,
            n_layers=1,
            n_attention_heads=1,
            t_attention=MultiheadedAttention2,
        )

        x_lengths = [12, 5, 6, 9, 3]  # randomly generated.
        examples = pt.randn(max(x_lengths), 5, example_features)  # max time steps, devices, features

        encoder_mask = make_bit_masking_tensor(x_lengths, mask_offset=0).unsqueeze(2)
        decoder_mask = make_bit_masking_tensor(x_lengths, mask_offset=1).unsqueeze(2)
        transformer_output_embedding = m.transformer(examples, encoder_mask, decoder_mask)

        classifier = ClassificationHead(n_classes=4)

        predictions = classifier(transformer_output_embedding)
        assert predictions.size() == pt.Size(examples.shape[1:2] + (classifier.n_classes,))
        assert_quasiprobability_axioms(predictions)

        true = helper_test_binary_labels(examples, classifier.n_classes)
        losses = pt.nn.CrossEntropyLoss()(predictions, true)
        losses.backward()
        assert True, "Should be able to perform a backwards pass"
