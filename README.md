<div align="center">

# Differentiable Modelling of Percussive Audio with Transient and Spectral Synthesis


[![Demo](https://img.shields.io/badge/Web-Audio_Examples-blue)](https://jordieshier.com/projects/differentiable_transient_synthesis/)
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](TBD) -->
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2207.08759-b31b1b.svg)](TBD) -->

<!-- <img width="700px" src="docs/new-generic-style-transfer-headline.svg"> -->

[Jordie Shier](https://jordieshier.com)<sup>1</sup>, [Franco Caspe](https://fcaspe.github.io/)<sup>1</sup>, Andrew Robertson<sup>2</sup>,<br> [Mark Sandler](http://eecs.qmul.ac.uk/people/profiles/sandlermark.html)<sup>1</sup>, [Charalampos Saitis](http://eecs.qmul.ac.uk/people/profiles/saitischaralampos.html)<sup>1</sup>, and [Andrew McPherson](https://www.imperial.ac.uk/people/andrew.mcpherson)<sup>3</sup>

<sup>1</sup> Centre for Digital Music, Queen Mary University of London<br>
<sup>2</sup> Ableton AG <br>
<sup>3</sup> Dyson School of Design Engineering, Imperial College London <br>

</div>

## Abstract
Differentiable digital signal processing (DDSP) techniques, including methods for audio synthesis, have gained attention in recent years and lend themselves to interpretability in the parameter space. However, current differentiable synthesis methods have not explicitly sought to model the transient portion of signals, which is important for percussive sounds. In this work, we present a unified synthesis framework aiming to address transient generation and percussive synthesis within a DDSP framework. To this end, we propose a model for percussive synthesis that builds on sinusoidal modeling synthesis and incorporates a modulated temporal convolutional network for transient generation. We use a modified sinusoidal peak picking algorithm to generate time-varying non-harmonic sinusoids and pair it with differentiable noise and transient encoders that are jointly trained to reconstruct drumset sounds. We compute a set of reconstruction metrics using a large dataset of acoustic and electronic percussion samples that show that our method leads to improved onset signal reconstruction for membranophone percussion instruments.

## Install
Clone the repo and then install the `drumblender` package. Requires Python version 3.9 or greater.

```bash
pip install --upgrade pip
pip install -e ".[modal]"
```

If you don't need to run modal extraction on audio, for example, if using a pre-processed dataset for training. You can exclude the optional modal requirements. The difference is that `modal` includes `nn-audio` which depends on an older version of `numpy`.
```bash
pip install -e .
```

### Pre-trained Models
A set of pre-trained models are available in a submodule, to download:
```bash
git submodule update --init --recursive
```

## Inference
To resynthesise an audio sample using a trained model:
```bash
drumblender-synth config checkpoint input output
```

For example:
```bash
drumblender-synth models/forum-acusticum-2023/noise_parallel_transient_params.yaml models/forum-acusticum-2023/noise_parallel_transient_params.ckpt audio/a_snare.wav snare_resynth.wav
```

## Dataset
We used a private dataset of commercially recorded and produced one-shot acoustic and electronic drum samples. We unfortunately can't share those, but we've included a config using [Freesound One-Shot Percussive Sounds](https://zenodo.org/record/3665275) to provide an example of dataset configuration and training.

To download a pre-processed version of that dataset:
```bash
wget https://pub-814e66019388451395cf43c0b6f10300.r2.dev/drumblender-freesound-v0.tar.gz
mkdir -p dataset/freesound
tar -zxf drumblender-freesound-v0.tar.gz -C dataset/freesound
```

## Training
Use the `drumblender` command to train a new model.

```bash
drumblender fit -c cfg/01_noise_params.yaml --data cfg/data/freesound.yaml
```

Training config files can be found in the directory `cfg`. The configuration files
in the root of that directory are the configurations used to train and test the different
model configurations presented in the Forum Acusticum paper.

We used the PyTorch Lightning
[LightningCLI](https://lightning.ai/docs/pytorch/LTS/api/pytorch_lightning.cli.LightningCLI.html?highlight=lightningcli#pytorch_lightning.cli.LightningCLI).

## Testing
Pass `test` as an argument to `drumblender` to test a trained model. For example, to test a model on the test set of the Freesound Percussive One-Shot dataset:

```bash
drumblender test -c models/forum-acusticum-2023/noise_parallel_transient_params.yaml --ckpt models/forum-acusticum-2023/noise_parallel_transient_params.ckpt --data cfg/data/freesound.yaml --trainer.logger CSVLogger --model.test_metrics cfg/metrics/drumblender_metrics.yaml
```

The `--trainer.logger` argument overrides the logging configuration in the saved yaml file. `model.test_metrics` adds extra metrics used in the evaluation for the paper.

To run this command on a CPU you can add the argument `--trainer.accelerator cpu`

## System Information
Experiments were run using

- Python version: 3.10.10.
- GPU: Tesla V100-PCIE-16GB

The exact Python packages in the environment during model training are in
`train-packages.txt`.

## For Developers
To install dev requirements and pre-commit hooks:

```bash
$ pip install -e ".[dev]"
```

Install pre-commit hooks if developing and contributing:

```bash
$ pre-commit install
```
