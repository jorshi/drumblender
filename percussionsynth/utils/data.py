"""
Data utils
"""
import hashlib
import json
import logging
import os
import sys
import tarfile
import threading
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import boto3
from tqdm import tqdm

# Setup logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def download_full_dataset(
    url: str, bucket: str, metafile: str, output_dir: Union[str, Path]
) -> None:
    """
    Download the kick dataset from Cloudflare.

    Args:
        url: The URL of the Cloudflare endpoint.
        bucket: The name of the bucket to download from.
        metafile: The name of the metadata file.
        output_dir: The directory to download the dataset to.
    """
    s3 = boto3.client(
        "s3",
        endpoint_url=url,
        aws_access_key_id=os.environ.get("CLOUDFLARE_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("CLOUDFLARE_ACCESS_SECRET_KEY"),
        region_name="auto",
    )

    # Get the dataset metadata. Create dataset directory if it doesn't exist.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_output = output_dir.joinpath(metafile)
    s3.download_file(bucket, metafile, str(json_output))

    # Get the list of all the subfolders to download
    # These are the folders that contain kick drum samples
    with open(json_output, "r") as f:
        metadata = json.load(f)

    # Get the list of all the files to download
    log.info("Getting list of files to download...")
    file_list = get_file_list_r2(metadata, bucket, s3)

    log.info(f"Found {len(file_list)} files to download.")

    # Download the files
    download_filelist_r2(file_list, output_dir, bucket, s3)


def get_file_list_r2(metadata: Dict, bucket: str, s3) -> List:
    """
    List all objects in subfolders of the R2 bucket.

    Args:
        metadata (Dict): The dataset metadata. Contains a list of items and
            their subfolders, which contain the files to download.
        bucket (str): The name of the bucket.
        s3 (boto3.client): The boto3 client for the R2 bucket.

    Returns:
        List: A list of all the files to download.
    """
    file_list = []
    for item in tqdm(metadata):
        folders = metadata[item]["folders"]
        for folder in folders:
            file_list.extend(get_subfolder_filelist_r2(bucket, folder, s3))

    return file_list


def get_subfolder_filelist_r2(bucket: str, subfolder: str, s3) -> List:
    """
    List all objects in a subfolder of the R2 bucket. Makes sure to
    handle continuation tokens.

    Args:
        bucket (str): The name of the bucket.
        subfolder (str): The subfolder to list files from.
        s3 (boto3.client): The boto3 client for the R2 bucket.

    Returns:
        List: A list of all the files to download.
    """
    kwargs = {"Bucket": bucket, "Prefix": subfolder}
    file_list = []
    while True:
        objs = s3.list_objects_v2(**kwargs)
        for obj in objs["Contents"]:
            file_list.append(obj["Key"])

        # Check for continuation token
        if objs["IsTruncated"]:
            kwargs["ContinuationToken"] = objs["NextContinuationToken"]
        else:
            break

    return file_list


def download_filelist_r2(file_list: List, output_dir: Path, bucket: str, s3):
    """
    Download a list of files from the R2 bucket.

    Args:
        file_list (List): A list of files to download.
        bucket (str): The name of the bucket.
        s3 (boto3.client): The boto3 client for the R2 bucket.
    """
    for file in tqdm(file_list):
        # Create the directory if it doesn't exist and download the file
        Path(output_dir.joinpath(file).parent).mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, file, str(output_dir.joinpath(file)))


def download_file_r2(
    filename: str, url: str, bucket: str, output: Optional[str] = None
):
    """
    Download a file from an R2 bucket.

    Args:
        filename: The name of the file to download.
        url: The URL of the Cloudflare endpoint.
        bucket: The name of the bucket.
        output (optional): The name of the output file. Defaults to filename in bucket.
    """
    s3 = boto3.client(
        "s3",
        endpoint_url=url,
        aws_access_key_id=os.environ.get("CLOUDFLARE_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("CLOUDFLARE_ACCESS_SECRET_KEY"),
        region_name="auto",
    )

    if output is None:
        output = filename

    # Get object metadata
    obj = s3.head_object(Bucket=bucket, Key=filename)
    size = obj["ContentLength"]

    # Download the file
    s3.download_file(
        bucket,
        filename,
        str(output),
        Callback=R2ProgressPercentage(filename, upload=False, size=size),
    )
    sys.stdout.write("\n")


def upload_file_r2(filename: str, url: str, bucket: str):
    """
    Upload a file to the R2 bucket.

    Args:
        filename (str): The name of the file to upload.
        url (str): The URL of the Cloudflare endpoint.
        bucket (str): The name of the bucket.
    """
    s3 = boto3.client(
        "s3",
        endpoint_url=url,
        aws_access_key_id=os.environ.get("CLOUDFLARE_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("CLOUDFLARE_ACCESS_SECRET_KEY"),
        region_name="auto",
    )

    s3.upload_file(filename, bucket, filename, Callback=R2ProgressPercentage(filename))


class R2ProgressPercentage:
    """
    A class to track the progress of a file upload to the R2 bucket.
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html # noqa: E501

    Args:
        filename: The name of the file being transferred.
        upload: Whether the file is being uploaded or downloaded,
            defaults to True (Upload).
        size (optional): The size of the file being transferred, defaults to None.
            Required for downloads.
    """

    def __init__(
        self, filename: str, upload: bool = True, size: Optional[float] = None
    ):
        self._filename = filename
        if upload:
            self._size = float(os.path.getsize(filename))
        else:
            assert size is not None, "Must provide size for download."
            self._size = size
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)"
                % (self._filename, self._seen_so_far, self._size, percentage)
            )
            sys.stdout.flush()


def get_files_from_folders(basedir: str, folders: List[str], pattern: str) -> List:
    """
    List all files in a list of folders.

    Args:
        folders (List[str]): A list of folders to search for files.
        pattern (str): The pattern to search for. E.g. "*.wav"
    """
    file_list = []
    for folder in folders:
        file_list.extend(Path(basedir).joinpath(folder).rglob(pattern))

    return file_list


def create_tarfile(output_file: str, source_dir: str):
    """
    Create a tarfile from a directory.

    Args:
        output_file: The name of the tarfile to create.
        source_dir: The directory to create the tarfile from.
    """
    with tarfile.open(output_file, "w:gz") as tar:
        for item in tqdm(list(Path(source_dir).rglob("*"))):
            if item.is_file():
                tar.add(item, arcname=item.relative_to(source_dir))


def str2int(s: str) -> int:
    """
    Convert string to int using hex hashing.
    https://stackoverflow.com/a/16008760/82733
    """
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (2**32 - 1)
