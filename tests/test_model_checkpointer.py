"""
Test the ModelCheckpointer is working as desired
Not leveraging the pytest stuff because it just couples us to the framework. Everything here runs regardless.
Pytest only affords leaving the methods outside of a TestUnit class (which is a win, purely from a syntactic noise perspective)

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from pathlib import Path

from torch import nn

from providence.training import EpochLosses
from providence.utils import configure_logger_in_dir
from providence_utils.callbacks import ModelCheckpointer

logger = configure_logger_in_dir("./tests", logger_name="model-checkpointer-test-001")

tiny_model = nn.Linear(2, 1)
output_dir = Path(".tmp")
output_dir.mkdir(exist_ok=True)


def clean_up():
    for child in output_dir.iterdir():
        child.unlink(missing_ok=True)


def use_callback(cb: ModelCheckpointer, n_epochs: int):
    for inc in range(n_epochs):
        fake_loss = 100 - inc
        cb.after_epoch(inc, tiny_model, ..., EpochLosses(0, fake_loss), ...)  # type: ignore[arg-type]


def test_keep_nothing():
    clean_up()
    keep_old_p = 0
    num_to_expect = keep_old_p + 1
    checkpointer = ModelCheckpointer(output_dir, "val_loss", logger, keep_old=keep_old_p)
    use_callback(checkpointer, 10)

    assert (
        len(list(output_dir.iterdir())) == num_to_expect
    ), f"Should have {num_to_expect} files on disk for keep_old={keep_old_p}"


def test_keep_small_number():
    clean_up()
    keep_old_p = 5
    num_to_expect = keep_old_p + 1
    checkpointer = ModelCheckpointer(output_dir, "val_loss", logger, keep_old=keep_old_p)
    use_callback(checkpointer, 10)

    assert (
        len(list(output_dir.iterdir())) == num_to_expect
    ), f"Should have {num_to_expect} files on disk for keep_old={keep_old_p}"


def test_keep_semantic_demonstration():
    """Only test worthy of a comment, meant to show that we keep only up to the allowable number of epochs
    i.e. if there's only one file per epoch (the assumption of all of these tests) then the maximum number
    of output files (the assertion) is the min(epochs, keep_old + 1)
    """
    clean_up()
    keep_old_p = 30
    n_test_epochs = 10
    num_to_expect = min(n_test_epochs, keep_old_p + 1)
    checkpointer = ModelCheckpointer(output_dir, "val_loss", logger, keep_old=keep_old_p)
    use_callback(checkpointer, n_test_epochs)

    assert (
        len(list(output_dir.iterdir())) == num_to_expect
    ), f"Should have {num_to_expect} files on disk for keep_old={keep_old_p}"


def test_keep_everything():
    clean_up()
    keep_old_p = True
    n_test_epochs = 20
    num_to_expect = n_test_epochs
    checkpointer = ModelCheckpointer(output_dir, "val_loss", logger, keep_old=keep_old_p)
    use_callback(checkpointer, n_test_epochs)

    assert (
        len(list(output_dir.iterdir())) == num_to_expect
    ), f"Should have {num_to_expect} files on disk for keep_old={keep_old_p}"


def test_keep_only_best():
    clean_up()
    keep_old_p = False
    num_to_expect = 1
    checkpointer = ModelCheckpointer(output_dir, "val_loss", logger, keep_old=keep_old_p)
    use_callback(checkpointer, 10)

    assert (
        len(list(output_dir.iterdir())) == num_to_expect
    ), f"Should have {num_to_expect} files on disk for keep_old={keep_old_p}"
