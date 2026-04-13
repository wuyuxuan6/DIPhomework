from pathlib import Path
import tarfile
import urllib.request


def main():
    root = Path(__file__).resolve().parent
    datasets_dir = root / "datasets"
    datasets_dir.mkdir(exist_ok=True)

    file_name = "facades"
    url = f"http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{file_name}.tar.gz"
    tar_path = datasets_dir / f"{file_name}.tar.gz"
    target_dir = datasets_dir / file_name

    if not target_dir.exists():
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, tar_path)
        print(f"Extracting to {datasets_dir} ...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(datasets_dir)
        tar_path.unlink(missing_ok=True)
    else:
        print(f"Dataset already exists at {target_dir}")

    for split in ("train", "val"):
        image_paths = sorted((target_dir / split).glob("*.jpg"))
        list_path = root / f"{split}_list.txt"
        list_path.write_text(
            "\n".join(str(path.resolve()) for path in image_paths) + ("\n" if image_paths else ""),
            encoding="utf-8",
        )
        print(f"Wrote {len(image_paths)} entries to {list_path.name}")


if __name__ == "__main__":
    main()
