"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from providence.training import EpochLosses
from providence.type_utils import patch
from providence_utils.callbacks import Callback
from providence_utils.callbacks import check_before_epoch
from providence_utils.callbacks import DeferrableEarlyStopping
from providence_utils.callbacks import EarlyStopping
from providence_utils.callbacks import EmergencyBrake


def tick_cb_epoch_only_losses(es_cb, epoch_losses: EpochLosses):
    es_cb.after_epoch(..., ..., ..., epoch_losses, ...)


class TestEarlyStopping:
    def test__track_train__should_trigger_callback(self):
        test_patience = 5
        es_cb = EarlyStopping(patience=test_patience, track="train_loss")
        es_cb.after_epoch(..., ..., ..., EpochLosses(10, 2), ...)  # set best found
        for _ in range(test_patience + 1):
            epoch_losses = EpochLosses(11, 2)
            tick_cb_epoch_only_losses(es_cb, epoch_losses)

        should_early_terminate, message = check_before_epoch([es_cb])
        assert should_early_terminate, "EarlyStopping cb isn't working correctly"


@patch
def __truediv__(self: EpochLosses, other: float) -> EpochLosses:
    return EpochLosses(self.training / other, self.validation / other)


@patch
def __sub__(self: EpochLosses, other: float) -> EpochLosses:
    return EpochLosses(self.training - other, self.validation - other)


class TestEmergencyBrake:
    def test__single_epoch__failure(self):
        cb = EmergencyBrake(1, 10)

        # this is called at the end of the epoch
        test_losses = EpochLosses(100, 100)
        cb.after_epoch(1, None, None, test_losses, None)

        # at the top of the next epoch
        termination = cb.before_epoch()
        assert termination.terminate, "Should want to terminate"
        assert termination.message.endswith(EmergencyBrake.format_message(test_losses, cb.requisite))

    def test__single_epoch__success(self):
        cb = EmergencyBrake(1, 10)

        # this is called at the end of the epoch
        test_losses = EpochLosses(10, 9)
        cb.after_epoch(1, None, None, test_losses, None)

        # at the top of the next epoch
        termination = cb.before_epoch()
        assert not termination.terminate, "Shouldn't want to terminate because we had one loss under the treshhold"

    def test__multi_epoch__failure(self):
        cb = EmergencyBrake(10, 10)

        # this is called at the end of the epoch
        for epoch in range(10):
            test_losses = EpochLosses(100, 100) - epoch
            cb.after_epoch(epoch, None, None, test_losses, None)

        # at the top of the next epoch
        termination = cb.before_epoch()
        assert termination.terminate, "Should want to terminate"
        assert termination.message.endswith(EmergencyBrake.format_message(test_losses, cb.requisite))

    def test__multi_epoch__success(self):
        cb = EmergencyBrake(10, 10)

        # this is called at the end of the epoch
        for epoch in range(1, 11):
            test_losses = EpochLosses(10, 10) / epoch
            cb.after_epoch(epoch, None, None, test_losses, None)

        # at the top of the next epoch
        termination = cb.before_epoch()
        assert not termination.terminate, "Shouldn't want to terminate"


class TestDeferrableEarlyStopping:
    def test_core_usage(self):
        from random import randint

        test_patience = randint(1, 15)
        test_defer = 5

        cb = DeferrableEarlyStopping(track="train_loss", patience=test_patience, defer_until=test_defer)
        "expected mehavior: I can tick this thing 5 times before it cares to track anything"
        cb_as_list = [cb]
        for i in range(test_defer):
            should_stop, error_message = check_before_epoch(cb_as_list)
            cb.after_epoch(
                epoch=i,
                losses=EpochLosses(i * 10, None),
                model=...,
                optimizer=...,
                dls=...,
            )
            assert not should_stop, "DeferrableEarlyStopping raised stop signal early: when testing defer"

        assert not len(cb.most_recent), "Shouldn't have accumulated anything."

        cb.verbose = True
        for i in range(test_patience):
            should_stop, error_message = check_before_epoch(cb_as_list)
            assert not should_stop, "DeferrableEarlyStopping raised stop signal early: when testing patience"

            cb.after_epoch(
                epoch=...,
                losses=EpochLosses(i * 10, None),
                model=...,
                optimizer=...,
                dls=...,
            )

        assert cb.best == 0, "Should have seen train_loss=0 as the best (so far)"
        assert len(cb.most_recent) == test_patience - 1, "Should have reset with the lowest loss seen"

    def test_fallback__(self):
        # TODO: when defer_until=0, it's just an EarlyStopping callback
        ...


class TestInteractions:
    def test_ebrake_deferrable_conflict__ebrake_wins(self):
        "As the name suggests, test the potential conflict between the DeferrableEarlyStopping and EmergencyBrake callbacks."
        ebrake = EmergencyBrake(2, 50.0)
        deferrable = DeferrableEarlyStopping("train", patience=1, defer_until=3)
        cbs: list[Callback] = [ebrake, deferrable]

        should_terminate, _ = check_before_epoch(cbs)
        assert not should_terminate, "Termination is premature"
        for _ in range(2):
            for cb in cbs:
                cb.after_epoch(
                    losses=EpochLosses(training=50, validation=50),
                    epoch=...,
                    model=...,
                    optimizer=...,
                    dls=...,
                )

        assert len(deferrable.most_recent) == 0, "Shouldn't have accumulated any 'recents' yet."
        assert deferrable._invocation_count == 2

        should_terminate, _ = check_before_epoch(cbs)
        assert should_terminate, "Ebrake should be displeased with the losses it has seen"

    def test_ebrake_deferrable_conflict__deferrable_wins(self):
        "Ebrake loses because its requisite loss is so high"
        ebrake = EmergencyBrake(3, 50.0)
        deferrable = DeferrableEarlyStopping("train", patience=1, defer_until=0)
        cbs: list[Callback] = [ebrake, deferrable]

        should_terminate, _ = check_before_epoch(cbs)
        assert not should_terminate, "Termination is premature"

        # set the "best" in deferrable
        should_terminate, error_message = check_before_epoch(cbs)
        for cb in cbs:
            tick_cb_epoch_only_losses(cb, EpochLosses(1, 0))

        assert not should_terminate, "Something went wrong with the state management"

        # two epochs of: first is allowed, second trips the termination
        for _ in range(past_patience := deferrable.patience + 1):
            should_terminate, error_message = check_before_epoch(cbs)
            for cb in cbs:
                tick_cb_epoch_only_losses(cb, EpochLosses(2, 0))

        # we do the check at the start of the epoch.
        should_terminate, error_message = check_before_epoch(cbs)
        assert should_terminate, f"Should terminate because we had {past_patience} epochs with higher-than-best values"
        assert f"Best = {deferrable.best}" in error_message, "Error message not constructed correctly"

    def test_ebrake_deferrable_conflict__ebrake_narrow_win(self):
        "Ebrake win because its requisite loss is lower, and Deferrable is all about relative / trend - rather than absolute values"
        ebrake = EmergencyBrake(check_at=3, requisite_loss=50.0)
        deferrable = DeferrableEarlyStopping("train", patience=3, defer_until=0)
        cbs: list[Callback] = [ebrake, deferrable]

        losses = [EpochLosses(l, 50) for l in [250, 100, 50]]
        for el in losses:
            should_terminate, _ = check_before_epoch(cbs)
            for cb in cbs:
                tick_cb_epoch_only_losses(cb, el)

        assert (
            not should_terminate
        ), f"Haven't check the callbacks again, even though the state change in {type(ebrake).__name__} warrants it"

        should_terminate, error_message = check_before_epoch(cbs)
        assert (
            should_terminate
        ), f"should terminate because the Ebrake wasn't satisfied, per its check at epoch {ebrake.check_at}"
        assert "failed to descend below requisite" in error_message
