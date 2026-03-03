# Copyright (c) Meta Platforms, Inc. and affiliates.


class TestTrainerContext:
    def test_iteration(self):
        import ncclx_trainer_context

        # default iteration is -1
        assert ncclx_trainer_context.getIteration() == -1
        ncclx_trainer_context.setIteration(100)
        assert ncclx_trainer_context.getIteration() == 100
