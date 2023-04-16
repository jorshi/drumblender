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
$ drumblender fit -c cfg/kick_single_kicktcn.yaml
```

And so on.

## Dataset
To download a preprocessed dataset from a config file you can use the following
command. **Note requires R2 access, see below**:

```bash
$ drumblender-dataset cfg/data/singles/kick_single_modal.yml
```

This will download the preprocessed archive and extract it to the appropiate directory.

To preprocess a dataset from raw audio files (make sure to delete pre-processed files
first). Depending on the preprocessing requirements it may require additional
packages.

```bash
$ drumblender-dataset cfg/data/singles/kick_single_modal.yml --preprocess
```

### Cloudflare R2 Access

Requires correct authentication credentials for Cloudflare R2. See [cloudflare](https://developers.cloudflare.com/r2/data-access/s3-api/tokens/) for information on generating API tokens.

Save ID and key in a `.env` file in the project root:
```bash
CLOUDFLARE_ACCESS_KEY_ID=xxxx
CLOUDFLARE_ACCESS_SECRET_KEY=xxxx
