import pytest
from unittest.mock import patch
from src.clone_repo import clone_repos

@patch("src.clone_repo.Repo.clone_from")
@patch("tempfile.mkdtemp", return_value="/tmp/fake-temp-dir")
def test_clone_repos_valid_url(mock_mkdtemp, mock_clone_from):
    urls = ["https://github.com/psf/requests"]

    # Act
    repo_paths = clone_repos(urls)

    # Assert
    assert isinstance(repo_paths, list)
    assert len(repo_paths) == 1

    repo_name, path = repo_paths[0]
    assert repo_name == "requests"
    assert path == "/tmp/fake-temp-dir"

    mock_clone_from.assert_called_once_with(urls[0], "/tmp/fake-temp-dir")
