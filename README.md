# MIMIC Benchmark in PyTorch

MIMIC Benchmark reimplemented in PyTorch. For examples and documentation checkout:  https://torchmimic.readthedocs.io/en/latest/

## Installation

To install, run:

```
pip install torchmimic
```

## Data

This library contains PyTorch datasets that will load your existing MIMIC Benchmark data. To use those datasets, just import the data you want from `torchmimic.data`:

```
from torchmimic.data import DecompensationDataset
from torchmimic.data import IHMDataset
from torchmimic.data import LOSDataset
from torchmimic.data import PhenotypingDataset
from torchmimic.data import MultitaskDataset
```

## Models

We've included some of the models used in the benchmark paper for reproduction and comparison with your own methods. Current models include:

- Standard LSTM

You can access these models from `torchmimic.models`:

```
from torchmimic.models import StandardLSTM
```

## Benchmarks

Each of the benchmarks can be found in `torchmimic.benchmarks`:

```
from torchmimic.benchmarks import DecompensationBenchmark
from torchmimic.benchmarks import IHMBenchmark
from torchmimic.benchmarks import LOSBenchmark
from torchmimic.benchmarks import PhenotypingBenchmark
from torchmimic.benchmarks import MultitaskBenchmark
```

At a minimum, the benchmark need a model to be trained. You can create the model from one of our existing ones or you can create your own.


## Weights and Biases Support

This library has built-in logging of model configurations and key metrics using the Weights and Biases library.


## TODOs

### Benchmarks

- [x] Phenotyping
- [x] Length of Stay
- [x] Decompensation
- [x] In Hospital Mortality
- [ ] Multitask

### Models
- [x] Standard LSTM
- [ ] Standard LSTM + Deep Supervision
- [ ] Logistic Regression
- [ ] Channelwise LSTM
- [ ] Channelwise LSTM + Deep Supervision
- [ ] Multitask LSTM
- [ ] Multitask Channelwise LSTM

### Testing

- [ ] Test data for GitHub Workflow
