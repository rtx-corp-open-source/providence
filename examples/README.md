# Providence Examples and User Guide

This directory is designed to show you how to achieve some task with the Providence Library, with a few different perspectives on the task
- How do I do ...?
- Examples as API Reference

## "How to..." or "How do I ..."

### Reproduce the research

on the [Backblaze Dataset](./01_1_high_level_api.py)

or an example of making a [small tweak to batch size](./01_2_high_level_api_with_overrides.py)


### Create a new architecture
There are two paths to it:

1. Here's an [example](./01_3_high_level_api_custom_model.py) of tweaking *our* architectures
2. Alternatively, if you comply with being a PyTorch RNN, you can follow this [example](./02_mid_level_api_custom_model.py)
    - More is explained in the documentation of `providence.nn.rnn.ProvidenceRNN`

### Make a custom training loop

You don't need to dive too deep.
Pealing back just one layer of the API, but keeping everything else the same let's 

1. Review `custom_training()` in [this example](./03_custom_training_via_high_and_mid_level_api.py)

2. Or see how you might build a callback facility around your model training in [this example](./04_callbacks_via_mid_level_api.py)

### Use a custom loss functions

1. If you want to ingest our paper's discrete Weibull parameters, there is an example of a custom loss function
    - in use in [05_custom_loss_with_low_level_api.py](./05_custom_loss_with_low_level_api.py)
      - the [objectified version](./05_custom_loss_with_low_level_objects_api.py)
2. If you want to use your own learn how we go about it
    - dig into the implementation of [`discrete_weibull_rmse()`](../rewrite/loss.py)

### Use a new distribution

Follow our implementation in [distributions.py](../rewrite/distributions.py)
1. Implement a parameter type that inherits from `NamedTuple`
2. Implement a Distribution that solves the problems outlined in the `SurvivalAnalysisDistribution`
3. Make sure your distribution inherits from `SurvivalAnalysisDistribution` (mostly so your editor helps you out)


## Glance-able Reference

Examples were written to progressive dig into the API layers.

### High-Level API

<!-- - 1.1: four lines to reproduce the losses from our research runs (but not all the visualizations)
- 1.2: the above with overrides of the training batch size and epoch count
- 1.3: use custom, instantiated model with the high-level training and data APIs (see the `ProvidenceTransformer` in action)
- 1.4: replacing our optimizers with one of your preference
- 1.5: smart using our high-level metrics -->


1. four lines to reproduce the losses from our research runs
   1. including only a simple visualization ([1.1][1_1])
   1. the above with overrides of the training batch size and epoch count ([1.2][1_2_overrides])
   3. use custom, instantiated model with the high-level training and data APIs i.e. see the `ProvidenceTransformer` in action ([1.3][1_3_custom_model])
   4. replacing our optimizers with one of your preference ([1.4][1_4_optimizers])
   5. smart using our high-level metrics ([1.5][1_5_metrics])

### Mid-Level API

2. Extension that adheres to our API naming convention, with a bidirectional LSTM (**NOT** THOROUGHLY TESTED) ([2][]])
3. Multi-pass training at the epoch-level, using the mid-level training APIs ([3][])
4. A simple callback API, which we used in some experimentation to explore the space ([4][])

### Low-Level API / detailed usage

<!-- - 5.1: a custom loss function that sums RMSE with the Weibull negative loglikelihood
- 5.2: the above, with the PyTorch OOP Loss function convention
- 6: reimplementation of the training that Martinsson specified in his 2017 thesis that inspired this work. -->

5. a custom loss function that sums RMSE with the Weibull negative loglikelihood
    1. using a python _function_ ([5.1 functional][5_1_loss_func])
    2. using a Pytorch _OOP_ ([5.2 objectified][5_2_loss_object])
6. reimplementation of the training that Martinsson specified in his 2017 thesis that inspired this work. ([6][])
7. If you want to see the full training of a single model, look at [example 7][7]

<!-- Example links -->
[1_1]: ./01_1_high_level_api.py
[1_2_overrides]: ./01_2_high_level_api_with_overrides.py
[1_3_custom_model]: ./01_3_high_level_api_custom_model.py
[1_4_optimizers]: ./01_4_high_level_api_with_custom_optimizer.py
[1_5_metrics]: ./01_5_high_level_api_metrics.py

[2]: ./02_mid_level_api_custom_model.py
[3]: ./03_custom_training_via_high_and_mid_level_api.py
[4]: ./04_callbacks_via_mid_level_api.py

[5_1_loss_func]: ./05_1_custom_loss_with_low_level_api.py
[5_2_loss_object]: ./05_2_custom_loss_with_low_level_objects_api.py
[6]: ./06_martinsson_experiment_4_2.py
[7]: ./07_full_model_evaluation.py