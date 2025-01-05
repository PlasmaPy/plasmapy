"""
Functionality for downloading files from a URL.

Intended for downloading files from |PlasmaPy's data repository|.
"""

import contextlib
import json
import os
import time
import warnings
from pathlib import Path
from urllib.parse import urljoin

import requests

__all__ = ["_API_CONNECTION_ESTABLISHED", "Downloader"]

_API_CONNECTION_ESTABLISHED = False
_IS_CI = "GH_TOKEN" in os.environ


# TODO: use a config file variable to allow users to set a location
# for the data download folder?

try:
    response = requests.get("https://api.github.com/", timeout=10)

    _API_CONNECTION_ESTABLISHED = response.status_code == 200
except (
    requests.exceptions.ConnectionError,
    requests.exceptions.ReadTimeout,
) as e:  # coverage: ignore
    # TODO: logging library when??
    print(f"Failed to connect to GitHub API:\n{e}")  # noqa: T201
    _API_CONNECTION_ESTABLISHED = False


class Downloader:
    """
    Accesses the PlasmaPy resource files.

    Retrieves local paths to resource files, and downloads those files from
    |PlasmaPy's data repository| if they cannot be found locally.

    Parameters
    ----------
    directory : `~pathlib.Path`, optional
        The directory into which files will be downloaded. The default
        is :file:`/~//.plasmapy//downloads//`

    validate : `bool`, default: `True`
        If `True`, verify that local files are up-to-date with the data
        repository, and use the GitHub API to verify download URLs before
        downloading files. If `False`, return any matching local file without
        verification and, if a local file cannot be found, attempt to download
        from the repository without validation.

    api_token :  `str`, optional
        A GitHub authorization token that, if provided,
        will be used for queries to the GitHub API. If none is provided, public
        API calls will be used.

    """

    # URL for the PlasmaPy-data repository through the GitHub API
    _API_BASE_URL = "https://api.github.com/repos/PlasmaPy/PlasmaPy-data/contents/"

    # Name for the local file that stores the SHA hashes of downloaded files
    # and information about the SHA on the server
    _blob_file_name = "RESOURCE_BLOB_SHA.json"

    # Base URL for RAW files
    _RAW_BASE_URL = "https://raw.githubusercontent.com/PlasmaPy/PlasmaPy-data/main/"

    def __init__(
        self,
        directory: Path | None = None,
        validate: bool = True,
        api_token: str | None = None,
    ) -> None:
        if directory is None:
            # No test coverage for default directory, since pytest always
            # saves into a temporary directory
            self._download_directory = (
                Path.home() / ".plasmapy" / "downloads"
            )  # coverage: ignore
        else:
            self._download_directory = Path(directory)

        # If currently operating in a CI environment and no token was specified,
        # take the token from the CI environment variables
        if _IS_CI and api_token is None:
            api_token = api_token if api_token is not None else os.getenv("GH_TOKEN")

        self._validate = validate
        self._api_token = api_token

        # Flag to record whether the blob file has been updated from the repo
        # by this instantiation of the class. Once the file has been updated
        # once, we won't update it again to limit API calls.
        self._updated_blob_file_from_repo = False

        self._download_directory.mkdir(parents=True, exist_ok=True)

        # Path to the local SHA blob file
        self._blob_file = Path(self._download_directory, self._blob_file_name)

        # Create the SHA blob file if it doesn't already exist
        if not self._blob_file.is_file():
            self._blob_dict = {}
            self._write_blobfile()
        # Otherwise, read the SHA blob file
        else:
            self._read_blobfile()

    def _write_blobfile(self) -> None:
        """
        Write the _local_blob_dict to disk.
        """
        with self._blob_file.open("w") as f:
            json.dump(self._blob_dict, fp=f)

    def _read_blobfile(self) -> None:
        """
        Read the _local_blob_dict from disk.
        """
        with self._blob_file.open("r") as f:
            self._blob_dict = json.load(f)

    @property
    def _api_connected(self) -> bool:
        """
        Return `True` if a connection exists to the API, otherwise `False`.
        """
        try:
            # Requesting this URL does not count as an API query
            self._http_request("https://api.github.com/rate_limit")
        # No testing because CI always has a connection to the API
        except requests.ConnectionError:  # coverage: ignore
            return False
        return True

    @property
    def _api_usage(self) -> tuple[int, int]:
        """
        Return the API call limit and the number currently used from this IP.
        """
        # Ensure that the GitHub API is not rate limited
        reply = self._http_request("https://api.github.com/rate_limit")
        info = reply.json()
        rate_info = info["resources"]["core"]
        limit = int(rate_info["limit"])
        used = int(rate_info["used"])

        return limit, used

    @property
    def _api_is_rate_limited(self) -> bool:
        """
        Whether or not the API is currently rate limited.
        """
        limit, used = self._api_usage
        return used >= limit

    @property
    def _do_validation(self) -> bool:
        """
        Determine whether or not to enforce validation using the GitHub API.
        """
        return self._validate and self._api_connected and not self._api_is_rate_limited

    def _update_repo_blob_dict(self) -> None:
        """
        Update the blob file with a call to the repository.

        Raises
        ------
        ValueError
            If the URL does not return the expected JSON file with the
            expected keys.

        Returns
        -------
        repo_blob_dict : dict
            Dictionary with filenames as keys. Each item is another entry
            with keys ``"sha"`` and ``"download_url"``.
        """
        # If the current blob file has been updated in the past 5 minutes,
        # don't bother doing it again
        # Ignore in tests, as this won't happen in CI
        with contextlib.suppress(KeyError):
            # If the _timestamp key hasn't been set yet, the blob file has
            # never been updated before
            if time.time() - self._blob_dict["_timestamp"] < 300:
                return None  # coverage : ignore

        # If this instance of Downloader has already updated from the API once,
        # don't do it again. Almost certainly nothing has changed!
        # Not tested, since CI never waits >5 min with the same Downloader
        # instantiated.
        if self._updated_blob_file_from_repo:  # coverage: ignore
            return None

        reply = self._http_request(self._API_BASE_URL)

        # Extract the SHA hash and the download URL from the response

        # Extract contents to JSON
        # Not tested, since any URL on the GitHub API that doesn't raise a 404 error
        # should return a JSON
        try:  # coverage: ignore
            info = reply.json()
        except requests.exceptions.JSONDecodeError as err:  # coverage: ignore
            warnings.warn(
                "URL did not return the expected JSON file: "
                f"{self._API_BASE_URL}. "
                f"Response content: {reply.content.decode('utf-8')}. Exception: {err}"
            )
            self._validate = False
            return None

        for item in info:
            try:
                filename = item["name"]
                repo_sha = item["sha"]
                download_url = item["download_url"]

            # Not tested, since any URL on the GitHub API that doesn't return a 404
            # should be a JSON with these keys
            except (KeyError, TypeError) as err:  # coverage: ignore
                warnings.warn(
                    f"URL {self._API_BASE_URL} returned JSON file, "
                    "missing expected keys 'sha' and 'download_url`."
                    f" JSON contents: {info}. Exception: {err}"
                )
                filename = None
                repo_sha = None
                download_url = None

            if filename is not None:
                self._update_blob_entry(
                    filename, repo_sha=repo_sha, download_url=download_url
                )

        # Save the current epoch time in the blob file as a record of when
        # it was updated
        self._blob_dict["_timestamp"] = time.time()

        # At the end, write back to the blobfile
        self._write_blobfile()

        # The blob file has been updated: set this flag so we won't do it
        # again on this instance of Downloader
        self._updated_blob_file_from_repo = True

    def _update_blob_entry(
        self,
        filename: str,
        local_sha: str | None = None,
        repo_sha: str | None = None,
        download_url: str | None = None,
    ) -> None:
        """
        Update an entry in the blobfile, or create a new one if one doesn't
        exist.
        """

        if filename in self._blob_dict:
            if local_sha is not None:
                self._blob_dict[filename]["local_sha"] = local_sha
            if repo_sha is not None:
                self._blob_dict[filename]["repo_sha"] = repo_sha
            if download_url is not None:
                self._blob_dict[filename]["download_url"] = download_url
        else:
            self._blob_dict[filename] = {
                "local_sha": local_sha,
                "repo_sha": repo_sha,
                "download_url": download_url,
            }

    def _http_request(self, url: str) -> requests.Response:
        """
        Issue an HTTP request to the specified URL, handling exceptions.
        """

        # Only send GitHub api authorization if querying GitHub
        # auth = self._api_auth if "github.com" in url else None

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "PlasmaPy.utils.Downloader",
        }

        if self._api_token is not None:
            headers["authorization"] = f"Bearer {self._api_token}"

        try:
            reply = requests.get(url, headers=headers, timeout=10)

        # No test coverage for this exception since we can't test it without
        # severing the network connectivity in pytest
        except requests.ConnectionError as err:  # coverage: ignore
            raise requests.ConnectionError(
                f"Unable to connect to data repository {self._API_BASE_URL}"
            ) from err

        # Extract the 'message' value if it is there
        # If the file does not exist on the repository, the GitHub API
        # will return `Not Found` in response to this but not raise a 404 error
        if reply.status_code == 404:
            raise ValueError(f"URL returned 404: {url}")

        return reply

    def _filepath(self, filename: str) -> Path:
        """Formats a filepath from a filename."""
        return Path(self._download_directory, filename)

    def _download_file(self, filename: str, dl_url: str) -> Path:
        """
        Download a file from a given URL to a specified path.

        Parameters
        ----------
        filename : str
            Name of the file to download: determines the filepath

        dl_url : str
            URL from which to download

        Returns
        -------
        filepath : str
            Path to the downloaded file

        """

        # Request the contents of the file from the download URL
        reply = self._http_request(dl_url)

        filepath = self._filepath(filename)

        # Write the contents to file
        with filepath.open(mode="wb") as f:
            f.write(reply.content)

        return filepath

    def _get_file_without_validation(self, filename: str) -> Path:
        """
        Return file logic without validation.

        Returns
        -------
        filepath : Path
            Path to file.

        Raises
        ------
        ValueError
            If the resource cannot be found locally or on the repository.

        """
        filepath = self._filepath(filename)

        # If the file exists locally, return that
        if filepath.is_file():
            return filepath

        # Try blindly downloading from the base URL
        # Note that downloading directly from the RAW url does not
        # require an API call.
        with contextlib.suppress(ValueError):
            dl_url = urljoin(self._RAW_BASE_URL, filename)
            return self._download_file(filename, dl_url)

        raise ValueError(
            "Resource could not be found locally or "
            "retrieved from the PlasmaPy-data repository: "
            f"{filename}."
        )

    def _get_file_with_validation(self, filename: str) -> Path:
        """
        Return file logic with validation.
        """
        filepath = self._filepath(filename)

        self._update_repo_blob_dict()

        # Retrieve the values: try/except catches KeyError both for
        # key `filename` and `local_sha`/`repo_sha`
        try:
            local_sha = self._blob_dict[filename]["local_sha"]
        except KeyError:
            local_sha = None

        try:
            repo_sha = self._blob_dict[filename]["repo_sha"]
        except KeyError:
            repo_sha = None

        # If local sha and online sha are equal, return the local filepath
        if local_sha == repo_sha and local_sha is not None:
            return filepath

        # If the file is found online, try downloading from the repository
        elif repo_sha is not None:
            dl_url = self._blob_dict[filename]["download_url"]
            # Download the file
            filepath = self._download_file(filename, dl_url)

            # This is a verified download, so we now know the local_sha is
            # the same as the repo_sha
            self._update_blob_entry(filename, local_sha=repo_sha)
            self._write_blobfile()

            return filepath

        # Otherwise fall back to retrieving the file without validation
        else:
            self._validate = False
            warnings.warn(
                f"Could not retrieve file {filename} with validation: "
                "trying again without validation."
            )
            return self._get_file_without_validation(filename)

    def get_file(self, filename: str) -> Path:
        """
        Returns a local path to a resource file, downloading it if necessary.

        Parameters
        ----------
        filename : str
            The name of the file in the |PlasmaPy's data repository|.

        Returns
        -------
        Path : `~pathlib.Path`
            The local path to the resource file.

        """
        if self._do_validation:
            return self._get_file_with_validation(filename)
        return self._get_file_without_validation(filename)
