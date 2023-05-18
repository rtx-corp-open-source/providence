<!-- 
**Raytheon Technologies proprietary**
Export controlled - see license file
-->
# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2023-05-16
### Changed
- Implemented `mypy` static type checking and flake8 , initial run of black.
- Setup pre-commit.
- Used `jaxtyping` to better support `mypy` static type checking.
  - `providence.distribution` Distributions instantiation is now required to comply with `mypy`.

### Added
- `.pre-commit-config.yaml`
- `pyproject.toml`
- `mypy.ini`

### Notes
Additional fixes to produce the internal Jenkins pipeline

## [1.0.post1.dev7] - 2023-01-23
### Added
- `providence/nn/transformer/memory_efficient.py` for a memory-efficient attention implementation that has a complex interface for the sake of memory efficiency
- `CausalSelfAttention` from Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT/blob/e0c689cf38478eea9416757cec5f834620983862/model.py#L25) added
  to `providence/nn/transformer/deepmind.py`.
- `Weibull3` to `providence/distributions.py`: an implementation of the Weibull distribution, with a translation parameter.
- Old analysis notebooks
  - `notebooks/Experiment-Analysis-*` are for assessing the data needs of Providence, so far as censoring vs failure portion is concerned
  - `Watchlists_and_Usage_Notes` outlines how one might generate a watchlist for an arbitrary set of entities that need inference i.e. probability of event experiment on an arbitrary time scale.

### Changed
- `paper_reproductions::GeneralMetrics(...)` now takes a `dist_t` that is an implementation of `SurvivalAnalysisDistribution`
- Implementation of the DeepMind transformers - `ProvidenceDeepMindTransformer` and `ProvidenceBertTransformer` - to leverage the new `CausalSelfAttention`.

### Fixed
- Implementation of the DeepMind Transformers. They are now preferable to the `ProvidenceTransformer` for both flexibility of parameter inputs and performance.
- Clarified documentation for several things throughout the project.

### Notes
This is a partial build.
While the library is still usable and functional for standard experimentation, the additions and advancements above have not

## [1.0.post1.dev6] - 2022-11-03
### Added
- `Decoder` as "primary sequence" decoder
- Standard / original encoder-decoder Transformer, as `EDTransformer`
- Many supporting unit tests

### Fixed
- Layer normed used in the `ETransformer` no longer sums with the previous context i.e. actually normalizing rather than decorating.

### Notes
- We have an upcoming change to introduce dropout and layer norm parameters to the models in that file.
  - Given the simplicity of the implementation, it's just a matter of plumbing those parameters
- The unit tests need reasonable refactoring, so the code isn't so (nearly) copy-paste
  - There are meaningful differences that can manifest, but it is apparent that the current abstract around the tests is imposing
    more limitation than is necessary.

## [1.0.post1.dev5] - 2022-10-26
### Added
- `providence/nn/transformer/deepmind.py` components and BERT-based Encoder-only Transformer in `ETransformer`
  - Correspending tests in `tests/rewrite/nn/test_deepmind.py` and `tests/models/blocks/test_transformer.py`

### Changed
- `tests/models/test_models.py::TestProvidenceTransformer` to include test on the BERTish model.
- build version in `databricks-pip-init.sh`

### Removed
- `tests/rewrite` submodule. Tests under that module have been moved where appropriate.
  - There was a lot of redundancy that could be handled with file merging or renaming; there was relic of the old
    codebase's hierarchy (i.e. `models/blocks`);

## [1.0.post1.dev4] - 2022-09-07
### Changed
- `providence_utils/fastai_layers.py::Flatten` no longer wraps with an additional `Tensor`.

## [1.0.post1.dev3] - 2022-08-31
### Added
- `providence_utils/ts_transformer.py`, the Time Series Transformer from the original paper
- `providence_utils/fastcore_meta.py` and `fastai_torch_core.py` to fix instantiation of TST from TSAI.

### Fixed
- Import of `functional as F` to be from `torch.nn` rather than `torch`, proper.

## [1.0.post1.dev2] - 2022-08-30
### Added
- `providence_utils/tst/TST.py`, the Time Series Transformer from the TSAI
  - Added other immplementations that are not priority, just for reference, comparison, etc.
- `fastai_{layers|utils}.py` to facilitate running the TST stuff from TSAI correctly.


### Changed
- Normalization for NASA dataset to be per-device normalization on the training set.
- `ReferenceProvidenceTransformer` and `ProvidenceTransformer` have their masking updated to match the implications in
  the TSAI codebase i.e. no encoder masking, one-step decoder masking.
  - Implementations differ for the way inference is done internally.

## [1.0.post1.dev1] - 2022-08-23
### Changed
- `ReferenceProvidenceTransformer` masked inference behavior (backed by unit tests).

## [1.0.post1.dev0] - 2022-08
### Changed
- Replace the idiom of `df_view[field] = ...` with `df_view.assign()`, which correctly return a new DataFrame
  (or Series if `isinstance(df_view, pd.Series) == True`)
- `EmergencyBrake` semantics were not clear, so unit tests were added and documentation clarified.

### Fixed
- `DeferrableEarlyStopping` as it wasn't behaving correctly due to the use of truthy `not value` rather than `value is not None`
  - Mitigated future bugs by delegating the early-stopping behavior to `EarlyStopping`, and just decorating that behavior (which
    was the spirit of this class in the first place.)
- Unit tests
  - Fixed the non-expressiveness of tests in `test_metrics_calculator.py`
  - Some tests were coupled to the upstream PyTorch typing on `simple_weibull_model` and `simple_providence_ds`; that has been removed.
    - Now tests do the upcasting if they are concerned with the typing potentially conflicting.

## [1.0rc7.dev5] - 2022-07-12
### Changed
- Version number to be compliant with PEP 440 (which is mandated by published tools such as pip)
  - More can be read on this scheme here: https://peps.python.org/pep-0440/

## [1.0.0rc7e] - 2022-07-08
### Added
- This changelog
- New table in `README.md` to show model performance for the RNN models on the NASA PHM '08 dataset
- Documentation of the shared functionality between `ProvidenceDataset`, `adapters.compute_tte` and `adapters.assemble_trainable_by_entity_id`
- Call to `GeneralMetrics` in example `07_full_model_evaluation.py`
- `providence_utils.mlflow` to capture **many** of the redundancies in the Databricks notebooks
- `scripts/reproduction_benchmark.py` to allow easy reproduction.
- Descriptive documentation of `providence.utils.set_seed()` to explain the challenge of reproducibility
    - Implementation of `providence.utils.set_seed(...)` also given a safe assignment of `cudnn.benchmark = False`.
### Changed
- Extracted a new parameter for `compute_tte()` in `tte_name: str`
- Updated the version of the build in `databricks-pip-init.sh`
