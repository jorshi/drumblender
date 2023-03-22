# 2 KICK 2 FURIOUS

## Install

To install with development dependencies:

```bash
$ pip install -e ".[dev]"
```

## Run

Code in this repo is accessed through the PyTorch Lightning CLI, which is available through the `k2k` console script. To see help:

```bash
$ k2k --help
```

To run an experiment, pass the appropriate config file to the `fit` subcommand:

```bash
$ k2k fit -c cfg/regression.yaml
```

And so on.

## Dataset
To download the unprocessed Kick Drum dataset:

```bash
$ k2k-dataset --unprocessed
```

Requires correct authentication credentials for Cloudflare R2. See [cloudflare](https://developers.cloudflare.com/r2/data-access/s3-api/tokens/) for information on generating API tokens.

Save ID and key in an environment variable:
```bash
CLOUDFLARE_ACCESS_KEY_ID=xxxx
CLOUDFLARE_ACCESS_SECRET_KEY=xxxx
