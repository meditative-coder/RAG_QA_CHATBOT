import tempfile
from git import Repo
from typing import List, Tuple

def clone_repos(repo_urls: List[str]) -> List[Tuple[str, str]]:
    """
    Clones multiple Git repositories into temporary directories.

    :param repo_urls: List of Git repository URLs to clone.
    :returns: List of tuples containing (repo_name, local_path).
    """
    repo_paths: List[Tuple[str, str]] = []
    for url in repo_urls:
        temp_dir = tempfile.mkdtemp()
        Repo.clone_from(url, temp_dir)
        repo_name = url.split('/')[-1].replace('.git', '')
        repo_paths.append((repo_name, temp_dir))
    return repo_paths
