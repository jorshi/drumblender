The percussion dataset used for training of models for Forum Acusticum 2023 is a private
datset and unfortunately cannot be shared.

This information is for provided for project collaborators.

To download a preprocessed dataset from a config file you can use the following
command. **Note requires R2 access, see below**:

```bash
$ drumblender-dataset cfg/data/percussion.yaml
```

This will download the preprocessed archive and extract it to the appropiate directory.

To preprocess a dataset from raw audio files.

```bash
$ drumblender-dataset cfg/data/percussion.yaml --preprocess
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
