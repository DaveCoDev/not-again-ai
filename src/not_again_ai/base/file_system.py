import pathlib


def create_file_dir(filepath: str) -> None:
    """Creates the parent directories for the specified filepath.
    Does not throw any errors if the directories already exist.

    Args:
        filepath (str): path to a file
    """
    root_path = pathlib.Path(filepath).parent
    root_path.mkdir(parents=True, exist_ok=True)
