.. mimic-benchmark-pytorch documentation master file, created by
   sphinx-quickstart on Thu Jun 23 12:54:17 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to torchmimic's documentation!
======================================

This library is an PyTorch-implemenation of the MIMIC-III Benchmark in a familiar PyTorch framework. It is designed to help experimenters rapidly test their methods against the MIMIC-III benchmark.

.. _installation:

Installation
============

You can install this package via the command line by entering::

    pip install torchmimic

Data
====

This library contains PyTorch datasets that will load your existing MIMIC Benchmark data. To use those datasets, just import the data you want from `torchmimic.data`::

    from torchmimic.data import DecompensationDataset
    from torchmimic.data import IHMDataset
    from torchmimic.data import LOSDataset
    from torchmimic.data import PhenotypingDataset
    from torchmimic.data import MultitaskDataset
    
    root_dir = "/path/to/your/data"
    training_dataset = DecompensationDataset(root_dir, train=True)
    val_dataset = DecompensationDataset(root_dir, train=False)

Models
======

We've included some of the models used in the benchmark paper for reproduction and comparison with your own methods.

You can access these models from `torchmimic.models`::

    from torchmimic.models import StandardLSTM
    
    model = StandardLSTM(
        n_classes=25,
        hidden_dim=256,
        num_layers=1,
        dropout_rate=0.3,
        bidirectional=True,
    )
        

Benchmarks
==========

Each of the benchmarks can be found in `torchmimic.benchmarks`::

    from torchmimic.benchmarks import DecompensationBenchmark
    from torchmimic.benchmarks import IHMBenchmark
    from torchmimic.benchmarks import LOSBenchmark
    from torchmimic.benchmarks import PhenotypingBenchmark
    from torchmimic.benchmarks import MultitaskBenchmark
    
    from torchmimic.models import StandardLSTM
    
    model = StandardLSTM(
        n_classes=25,
        hidden_dim=256,
        num_layers=1,
        dropout_rate=0.3,
        bidirectional=True,
    )
    
    data_dir = "/path/to/your/data"
    
    trainer = PhenotypingBenchmark(
        model=model,
        train_batch_size=8,
        test_batch_size=256,
        data=data_dir,
        learning_rate=0.001,
        weight_decay=0,
        report_freq=200,
        device=device,
        sample_size=None,
        wandb=True,
    )
    
    trainer.fit(100)

At a minimum, the benchmark need a model to be trained. You can create the model from one of our existing ones or you can create your own.

Loggers
==========================

We include benchmark specific loggers that print, save, and log important benchmark metrics along with training and model configurations. In order to log your run as  Loggers have the option to log "Weights and Biases" runs. You can create a logger for you experiments with the following::

    from torchmimic.loggers import PhenotypingLogger
    
    exp_name = "Phenotyping Benchmark"
    
    config = {
        "LR": 0.001,
        "Weight Decay": 0.0001
    }
    
    logger = PhenotypingLogger(exp_name, config, log_wandb=True)
    
You can use the logger during training like this::
    
    for epoch in range(epochs):
    
        logger.reset()                             # reset the logger before each epoch
        for data, labels in train_loader:
            outputs = model(batch)
            loss = loss_fn(output)
            
            ...
        
            logger.update(outputs, labels, loss)   # update the metrics in the logger
            
        logger.print_metrics(epoch, split='Train') # prints, logs, and save metrics
        
        logger.reset()
        for data, labels in val_loader:
            outputs = model(batch)
            loss = loss_fn(output)
            
            ...
        
            logger.update(outputs, labels, loss)
            
        logger.print_metrics(epoch, split='Eval')  # change "split" for t

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Contents
--------

.. toctree::

   models
   data
   metrics
   loggers