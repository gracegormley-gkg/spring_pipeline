#!/usr/bin/env python3
"""
S3 downloader: fetch a single document folder from the Impulse S3 bucket.

Run separately before the pipeline. Requires AWS credentials configured
(via env vars, ~/.aws/credentials, or IAM role).

Usage:
    python fetch_data.py --bucket nu-impulse-production --doc-id 35556036063543 --output-dir ./data
    python fetch_data.py --bucket nu-impulse-production --doc-id 35556036063543 --project-id P0492 --output-dir ./data
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")


def main() -> None:
    p = argparse.ArgumentParser(description="Download EIS document folder from S3")
    p.add_argument("--bucket", default="nu-impulse-production")
    p.add_argument("--doc-id", required=True, help="Document barcode ID (e.g. 35556036063543)")
    p.add_argument("--project-id", default="P0491", help="Project/collection prefix (default: P0491)")
    p.add_argument("--output-dir", default="./data", help="Local output directory")
    p.add_argument(
        "--dry-run", action="store_true", help="List objects that would be downloaded, don't fetch"
    )
    args = p.parse_args()

    try:
        import boto3  # type: ignore
        from botocore.exceptions import ClientError, NoCredentialsError  # type: ignore
    except ImportError:
        logger.error("boto3 not installed. Run: pip install boto3")
        sys.exit(1)

    s3 = boto3.client("s3")
    prefix = f"{args.project_id}_{args.doc_id}/"
    output_root = Path(args.output_dir) / f"{args.project_id}_{args.doc_id}"

    logger.info("Listing s3://%s/%s ...", args.bucket, prefix)

    try:
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=args.bucket, Prefix=prefix)

        objects: list[str] = []
        for page in pages:
            for obj in page.get("Contents", []):
                objects.append(obj["Key"])

    except NoCredentialsError:
        logger.error(
            "AWS credentials not found. Configure via:\n"
            "  export AWS_ACCESS_KEY_ID=...\n"
            "  export AWS_SECRET_ACCESS_KEY=...\n"
            "  export AWS_DEFAULT_REGION=..."
        )
        sys.exit(1)
    except ClientError as exc:
        logger.error("S3 error: %s", exc)
        sys.exit(1)

    if not objects:
        logger.error("No objects found at s3://%s/%s", args.bucket, prefix)
        logger.error("Check --doc-id and --project-id values, and that you have bucket access.")
        sys.exit(1)

    logger.info("Found %d objects", len(objects))

    if args.dry_run:
        for key in objects:
            print(f"  s3://{args.bucket}/{key}")
        logger.info("Dry run — nothing downloaded")
        return

    output_root.mkdir(parents=True, exist_ok=True)

    for key in objects:
        # Strip the prefix to get the relative path
        rel_path = key[len(prefix):]
        if not rel_path:
            continue
        local_path = output_root / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists():
            logger.info("  SKIP (exists): %s", rel_path)
            continue

        logger.info("  Downloading: %s", rel_path)
        try:
            s3.download_file(args.bucket, key, str(local_path))
        except ClientError as exc:
            logger.error("  FAILED: %s — %s", rel_path, exc)

    logger.info("Download complete: %s", output_root)
    logger.info("Next: python inspect_layout.py --doc-dir %s", output_root)


if __name__ == "__main__":
    main()
