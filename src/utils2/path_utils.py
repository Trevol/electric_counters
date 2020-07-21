import os


def ensureRelativeToDir(directory: str, path: str):
    assert path is not None and directory is not None
    directory = str(directory)
    path = str(path)
    # absolute path (/....) or explicitly relative (../ or ./)
    if path.startswith(os.sep) or path.startswith('.'):
        return path
    return os.path.join(directory, path)
