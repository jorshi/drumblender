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
Show an example of running model inference on a single file.
- Perhaps, also include as an example blending of the different elements using different samples.

## Dataset
We used a private dataset of commercially recorded and produced one-shot acoustic and electronic drum samples. We unfortunately can't share those, but we've included a config using [Freesound One-Shot Percussive Sounds](https://zenodo.org/record/3665275) to provide an example of dataset configuration and training.

To download a pre-processed version of that dataset:
```bash
wget https://pub-814e66019388451395cf43c0b6f10300.r2.dev/drumblender-freesound-v0.tar.gz
mkdir -p dataset/freesound
tar -zxf drumblender-freesound-v0.tar.gz -C dataset/freesound
```

## Training

## Run

Code in this repo is accessed through the PyTorch Lightning CLI, which is available through the `k2k` console script. To see help:

```bash
$ drumblender --help
```

To run an experiment, pass the appropriate config file to the `fit` subcommand:

```bash
$ drumblender fit -c cfg/kick_single_kicktcn.yaml
```

And so on.

## Dataset
To download a preprocessed dataset from a config file you can use the following
command. **Note requires R2 access, see below**:

```bash
$ drumblender-dataset cfg/data/singles/kick_single_ludwig.yaml
```

This will download the preprocessed archive and extract it to the appropiate directory.

To preprocess a dataset from raw audio files (make sure to delete pre-processed files
first).

```bash
$ drumblender-dataset cfg/data/singles/kick_single_ludwig.yaml --preprocess
```

For modal extraction nnAudio is required. Install using:

```bash
$ pip install ".[modal]"
```

### Cloudflare R2 Access

Requires correct authentication credentials for Cloudflare R2. See [cloudflare](https://developers.cloudflare.com/r2/data-access/s3-api/tokens/) for information on generating API tokens.

Save ID and key in a `.env` file in the project root:
```bash
CLOUDFLARE_ACCESS_KEY_ID=xxxx
CLOUDFLARE_ACCESS_SECRET_KEY=xxxx
```

## Reproducibility
Experiments were run using

- Python version: 3.10.10.
- GPU: Tesla V100-PCIE-16GB

The exact Python packages in the environment during model training are in
`train-packages.txt`.

## Developing
To install dev requirements and pre-commit hooks:

```bash
$ pip install -e ".[dev]"
```

Install pre-commit hooks if developing and contributing:

```bash
$ pre-commit install
```
