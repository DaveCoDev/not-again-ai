from pathlib import Path


def create_file_dir(filepath: str | Path) -> None:
    """Creates the parent directories for the specified filepath.
    Does not throw any errors if the directories already exist.

    Args:
        filepath (str | Path): path to a file
    """
    root_path = Path(filepath).parent
    root_path.mkdir(parents=True, exist_ok=True)


def readable_size(size: float) -> str:
    """Convert a file size given in bytes to a human-readable format.

    Args:
        size (int): file size in bytes

    Returns:
        str: human-readable file size
    """
    # Define the suffixes for each size unit
    suffixes = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]

    # Start with bytes
    count = 0
    while size >= 1024 and count < len(suffixes) - 1:
        count += 1
        size /= 1024

    # Format the size to two decimal places and append the appropriate suffix
    return f"{size:.2f} {suffixes[count]}"
