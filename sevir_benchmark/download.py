import os
from typing import Iterable, List, Optional

import boto3
from botocore.handlers import disable_signing


SEVIR_BUCKET = "sevir"


def make_anonymous_s3_resource():
    resource = boto3.resource("s3")
    resource.meta.client.meta.events.register("choose-signer.s3.*", disable_signing)
    return resource


def list_vil_keys(years: Optional[Iterable[int]] = None) -> List[str]:
    resource = make_anonymous_s3_resource()
    bucket = resource.Bucket(SEVIR_BUCKET)
    prefixes = [f"data/vil/{int(year)}/" for year in years] if years else ["data/vil/"]
    keys: List[str] = []
    for prefix in prefixes:
        for obj in bucket.objects.filter(Prefix=prefix):
            if obj.key.endswith(".h5"):
                keys.append(obj.key)
    return sorted(keys)


def download_key(bucket, key: str, destination_root: str, overwrite: bool = False) -> str:
    destination = os.path.join(destination_root, key)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if os.path.exists(destination) and not overwrite:
        return destination
    bucket.download_file(key, destination)
    return destination


def download_sevir_vil(
    destination_root: str,
    years: Optional[Iterable[int]] = None,
    overwrite: bool = False,
    include_catalog: bool = True,
    include_vil: bool = True,
    limit_files: Optional[int] = None,
) -> List[str]:
    resource = make_anonymous_s3_resource()
    bucket = resource.Bucket(SEVIR_BUCKET)

    downloaded: List[str] = []
    if include_catalog:
        downloaded.append(download_key(bucket, "CATALOG.csv", destination_root, overwrite=overwrite))

    if include_vil:
        keys = list_vil_keys(years=years)
        if limit_files is not None:
            keys = keys[:limit_files]
        for key in keys:
            downloaded.append(download_key(bucket, key, destination_root, overwrite=overwrite))
    return downloaded
