<!-- 
**Raytheon Technologies proprietary**
Export controlled - see license file
-->
# Providence: a library and framework for predicting time-to-event with neural networks

## What is Providence?
Providence is a library designed **for researchers to easily reproduce and extend *our* research**.
Named for the definition
> noun: quality or state of making provision for the future ([source](https://www.merriam-webster.com/dictionary/providence))

this library allows the simple and swift training of time-to-event prediction models written in [PyTorch](https://github.com/pytorch/pytorch).
It was developed in the context of aerospace engineering and remaining useful-life (RUL) prediction (see *NASA CMAPSS Turbofan* dataset and our paper for more.).
For more general and abstract reading, Michael Betancourt covers the foundations of Surival Analysis / Time-to-Event Modeling [here](https://betanalpha.github.io/assets/case_studies/survival_modeling.html)

## Installation

TODO: with Luke

```sh
pip install rtx-edx-providence
```

Requires: Python 3.7+, PyTorch 1.8.0+

## Features

- Train deep sequential models that predict (discretized) Weibull distributions, _at every time step_
- One-line initialization of a `DataLoader` on our data i.e. PHM Challenge '08 NASA Turbofan CMAPSS or Backblaze Harddrive SMART Stats
    - Swap that one-liner to get just the `torch` `Dataset`
- Visualize Weibull curves
    - For every time step
    - For every device in fleet
- Plug-and-play design, so you can take and use what you need
- Uses [`torchtyping`](https://github.com/patrick-kidger/torchtyping)

## Use Cases

- Predicting Remaining Useful Life (RUL) of arbitrary entities that can be represented as a time series dataset.
- Predicting probabilistic distribution of Time-to-Event (TTE) of arbitrary entities, as a nuanced take on the above.
- (With additional effort) Generating Watchlists to augment the decision-making of fleet maintenance Engineer, Owners, and PHM SMEs.
- (With substantial additional effort) Training deep learning models to learn representaions of time series datasets.

## Permitted Scope

- General code for the auto-regressive, time series-to-probability distribution prediction according to [our paper](https://ieeexplore.ieee.org/document/9843469).
- `ProvidenceDataset` implementations for open- / freely-available (with permissive licensing) datasets.
- Adaptations of open-source algorithms to blend with the existing codebase.

## Unpermitted Scope

- No algorithms with logic specific to any product or program, be it predictive or data pipeline-related
  - To be abundantly clear: no code related to the data tables from a given program or non-open data set should find its way into this codebase.
- [Adapters](https://en.wikipedia.org/wiki/Adapter_pattern) to a specific program, product, system, fleet, etc.
  - Exception: potential scrubs that will make data compliant with a certain compliance regime. Sharing automation is a good thing.

The following are either unsupported or antithetical to this library's function.
- Alternative methods for inference against time series data
  - Explanation: This library is **not** to be the one-stop-shop for *all* time series inference methods. The work here is that which pertains to predicting
    a sequence of (survival analytic-friendly) probability distributions, and adapting such to the tasks outlined in the use cases.
- Naive inference (i.e. sans pre-processing and shape transformation) of data that is arranged by shape other than `(timestep, entity, feature)`
- See "Why you shouldn't use Providence" below

## Location
Location
Code: https://github.devops.utc.com/Type-C-Lite/RTXDS-providence

Documentation, examples, contributing: https://providence.rtxai.com and this repository's `examples/`

## Functional Example of Library Features

We provide a simple example to demonstrate the dual design ethos of `providence`.
> Minimize SLOCs to produce a meaningful experiment. Provide multiple granularities of API coarseness to ease use.

```python
# 4 lines of code to train one of our best models :) 
model = BackblazeTransformer()
optimizer = BackblazeTransformerOptimizer(model)
backblaze_dls = BackblazeDataLoaders(quarter=BackblazeQuarter._2019_Q4, batch_size=optimizer.batch_size)
losses = BackblazeTraining(model, optimizer, backblaze_dls)
```

Surely, that can't be it? You are correct. We have more to offer. Perhaps you want to get a sense of the wall-clock time to train a model.

```python

def time_training_full_example(n_laps: int = 5, random_seed: int = 1234):
    experiment_start_timestamp = now_dt_string()
    print(f"Experiment started at {experiment_start_timestamp}")

    model = paper_reproductions.BackblazeTransformer()
    model.device = use_gpu_if_available()
    model_optim_init = paper_reproductions.BackblazeTransformerOptimizer
    dataloaders = BackblazeDataLoaders(quarter=BackblazeQuarter._2019_Q4, batch_size=64, random_seed=random_seed)

    # run training.
    training_run_metrics = []
    for training_lap in range(n_laps):
        print(f"{training_lap = }")
        set_seed(random_seed)  # global set seeds just before model initialization.
        model.reset_parameters()
        model_optim = model_optim_init(model)

        start_time = perf_counter()
        paper_reproductions.BackblazeTraining(model, model_optim, dataloaders) # one-line training
        training_time = perf_counter() - start_time
        training_run_metrics.append(training_time)

```
See the rest of the full example [here](./examples/example_training_fewest_lines.py).

For additional concrete demonstrations and best practices, see our [examples and HOWTO's](./examples/).

For more on the design and history of Providence, review our [DESIGN document](./DESIGN.md).

### Why you should use Providence
- You want to play with - or extend - the idea of predicting a sequence of time-to-event probability distributions

- You have time series data, that you want to work with device-by-device using `Dataset`s that make sense
- You have a model that you think could be a suited for this problem, but you don't want to set up all the software architecture

- You are fond of reproducible research with minimal boiler-plate

- You read our paper and want to see how our code performs.

### Why you shouldn't use Providence
- If you want to do Timeseries deep learning from scratch.

- If you don't value the premise of our work, nor the general research direction.

- If you want step-by-step instructions to deploy into production. That is not our focus. 

- If you discretized Weibull predictions, per time steps, across multiple sequential architectures, but it has to be in [Tensorflow](https://github.com/tensorflow/tensorflow) or [JAX](https://github.com/google/jax)


## What to do if you just want to _use_ Providence, rather than doing experimentation?

You probably want to predict RULs and the get the associated probabilities.
Providence makes this easy, because our RUL prediction is *the time step where the majority of the probability density is found*.
- For the Weibull distribution, this is the `mode` prediction.

As a code example, assumes
- your model exists (in full) at `path/to/my/model.pt`
- you can use the `ProvidenceDataset` documentation to initialize your model.

```python
model: Union[ProvidencModule, nn.Module] = torch.load("path/to/my/model.pt")
new_devices: DataFrame = ... # initialized with new devices

ds = ProvidenceDataset(new_devices, feature_columns=..., temporal_indicator='tte column')

calculator = MetricsCalculator(model, Weibull, ds)

# outputs from the model for every device, for every timestep in the dataset
outputs_df = calculator.outputs_per_device().assign(prob_fail_in_1_step=lambda df: Weibull.pdf(Weibull.Params(
    torch.from_numpy(outputs_df["alpha"].to_numpy()),
    torch.from_numpy(outputs_df["beta"].to_numpy())),
    torch.from_numpy(outputs_df["mode"].to_numpy())), # probability at the RUL predicted.
)))

# effectively the "next time step" failure predictions
predictions_for_each_device = (
    outputs_df.sort_values(by=["id", "tte"], ascending=[True, False])
    [["id", "tte", "censor", "mode", "prob_fail_in_1_step"]]
    .groupby("id")
    .tail(1)
    .rename({"mode": "RUL"}, axis="columns")
)
```

To streamline this inference, you would want to look into the internals of `MetricsCalculator.outputs_per_device()` and see how we do the per-timestep computation of the mode and probability of that mode.
There are many ancillary statistics that are being computed that you don't need to care about.

### What you can expect in performance



For Providence Recurrent architectures on the NASA '08 PHM challenge datasets
- This is the NASA Aggregate dataset, including all FD001-FD004 devices
- statistics are taken across the top 5 (GRU, LSTM) or top 20 (Vanilla) training runs
    - To generate a similar table, use `providence.paper_reproductions.GeneralMetrics(...)`
- Format:
    - $\mu \pm \sigma$ or $mean \pm stddev$
    - all values are rounded to the thousandths place


|model   | MSE  | MFE  | SMAPE  | SMPE |
|---|----|---|---|---|
|Vanilla  | $2047.170 \pm 95.064$  | $-0.414 \pm 3.796$ | $0.135 \pm 0.007$ |  $0.005 \pm 0.015$ |
|GRU|  $1746.090 \pm 156.038$  |  $-1.878 \pm 9.934$  |  $0.123 \pm 0.009$ | $0.012 \pm 0.041$  |
|LSTM| $3779.240 \pm 578.048$  |  $2.992 \pm 22.772$ | $0.210 \pm 0.023$  | $-0.012 \pm 0.111$  |

<!-- 
source table

Vanilla

|      |      MSE |       MFE |      SMAPE |        SMPE |   NaN_count |
|:-----|---------:|----------:|-----------:|------------:|------------:|
| mean | 2047.17  | -0.413529 | 0.135115   |  0.00513113 |           0 |
| std  |   95.064 |  3.79612  | 0.00608509 |  0.0147332  |           0 |
| 25%  | 1946.49  | -2.92355  | 0.130285   | -0.00799967 |           0 |
| 75%  | 2118.12  |  2.68616  | 0.138      |  0.0152686  |           0 |

GRU
|      |      MSE |      MFE |      SMAPE |        SMPE |   NaN_count |
|:-----|---------:|---------:|-----------:|------------:|------------:|
| mean | 1746.09  | -1.87795 | 0.123231   |  0.0122705  |           0 |
| std  |  156.038 |  9.93427 | 0.00956984 |  0.0407989  |           0 |
| 25%  | 1665.59  | -5.65637 | 0.118582   | -0.00889792 |           0 |
| 75%  | 1816.83  |  3.34285 | 0.128196   |  0.0185875  |           0 |

LSTM
|      |      MSE |      MFE |     SMAPE |       SMPE |   NaN_count |
|:-----|---------:|---------:|----------:|-----------:|------------:|
| mean | 3779.24  |  2.99916 | 0.210117  | -0.0119779 |           0 |
| std  |  578.048 | 22.7718  | 0.0229488 |  0.110702  |           0 |
| 25%  | 3238.84  | -0.86747 | 0.195059  | -0.0794543 |           0 |
| 75%  | 4315.77  | 16.2356  | 0.219626  |  0.0167791 |           0 |
 -->


# Citing Providence

TODO: with Luke, once IEEE is posted and Bibtex can be downloaded

# Compliance

## License and Export Control
The License and Required Marking (LICENSE_AND_REQUIRED_MARKING.rst) file provides the license and export control levels of the repository.

The Control Plan (CONTROL_PLAN.rst) contains requriements for participant compliance with Global Trade policy in this environment.

Unpublished data within this repository can be used by RTX users only. See the License and Required Marking (LICENSE_AND_REQUIRED_MARKING.rst) for more details and information on distribution.


[Alternate Means of Compliance (AMOC) Paper](https://devops.utc.com/-/media/Project/Corp/Corp-Intranet/Mattermost/DevOps-site2/Files/AMOC225_whitepaper_v1_2021-05-13.pdf?rev=cd7bf410327f42d1bff28fad89960b12&hash=0C423C181E00D0732AF0FA4F7061670C)_

## Restrictions
**All users must understand and comply with the license and control plan files from the previous section.**

The following is a general list of **RESTRICTED** technology and software (via code, issues, PRs, etc). This is not a comprehensive list: the user is responsible for ensuring compliance with the control plan, and consulting their IP and GT points of contact when questions arise.

- Technical data related to:
  - A specific product with technology higher than 9E991
  - A military engine program
  - Material processing and/or parameters
  - Composites
  - Encryption
  - Real time signal processing
  - Electronic Engine Control Systems
- Any third party material, components, or code.
  - Any incorporation of third party code (e.g., open source, proprietary code) must **strictly** be by reference, and **must not be** included in the code base.
  - This restriction includes both in its complete and partial forms. **NO EXTERNAL CODE SHOULD EVER BE COPIED INTO A REPOSITORY.**
  - Consult IP legal for third party and/or open source license guidelines.
- Any personal data

The following guidelines **MUST** be adhered to:
- Do not contribute work performed under or relating to a Government contract to these packages.
- If an invention disclosure has been submitted, follow the instructions of the patent advisory board.
- If an invention disclosure is in progress or contemplated in the future, contact IP legal before contributing relevant information.

## Escape Handling
If any of the above limitations are violated:

1. Follow your business unit's standard procedure for reporting incidents
2. Contact the repo maintainers and/or the RTX GitHub admins

## Points of Contact (for Compliance Purposes)
- RTX: Joe Calogero (joseph.calogero@rtx.com)
- PWA: Chris Ruoti (Christopher.ruoti@prattwhitney.com)
- PWC: TBD