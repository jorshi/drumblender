# Drum Blender

## Install

To install with development dependencies:

```bash
$ pip install -e ".[dev]"
```

## Run

Code in this repo is accessed through the PyTorch Lightning CLI, which is available through the `k2k` console script. To see help:

```bash
$ drumblender --help
```

To run an experiment, pass the appropriate config file to the `fit` subcommand:

```bash
$ drumblender fit -c cfg/regression.yaml
```

And so on.

## Dataset
To download the unprocessed Kick Drum dataset:

```bash
$ drumblender-dataset --unprocessed
```

Requires correct authentication credentials for Cloudflare R2. See [cloudflare](https://developers.cloudflare.com/r2/data-access/s3-api/tokens/) for information on generating API tokens.

Save ID and key in a `.env` file in the project root:
```bash
CLOUDFLARE_ACCESS_KEY_ID=xxxx
CLOUDFLARE_ACCESS_SECRET_KEY=xxxx
