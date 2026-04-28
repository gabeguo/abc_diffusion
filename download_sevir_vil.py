import argparse

from sevir_benchmark.download import download_sevir_vil


def main():
    parser = argparse.ArgumentParser(description="Download the VIL modality of the SEVIR dataset from AWS Open Data.")
    parser.add_argument("--output-root", type=str, default="datasets/sevir", help="Destination folder.")
    parser.add_argument(
        "--years",
        type=int,
        nargs="*",
        default=[2017, 2018, 2019],
        help="Years to download from s3://sevir/data/vil/. "
        "The default covers the standard nowcasting train/test split around June 1, 2019.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--catalog-only", action="store_true")
    parser.add_argument("--limit-files", type=int, default=None, help="Debug-only cap on number of H5 files to download.")
    args = parser.parse_args()

    downloaded = download_sevir_vil(
        destination_root=args.output_root,
        years=args.years,
        overwrite=args.overwrite,
        include_catalog=True,
        include_vil=not args.catalog_only,
        limit_files=args.limit_files,
    )
    print(f"Downloaded {len(downloaded)} files into {args.output_root}")


if __name__ == "__main__":
    main()
