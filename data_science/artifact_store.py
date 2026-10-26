"""Object storage for lakehouse snapshots and model artifacts.

``ArtifactStore`` is the narrow interface the jobs depend on. ``S3ArtifactStore``
talks to MinIO / S3; ``LocalArtifactStore`` mirrors the same key layout in a
local directory (tests and offline experiments). Keys are POSIX-style paths
relative to the bucket (or root directory). Every write replaces the whole
object atomically: readers see either the previous or the new content.
"""

from __future__ import annotations

import abc
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, HTTPClientError
from botocore.exceptions import ConnectionError as BotoConnectionError
from ds_runtime import retry_call
from ds_settings import S3Settings

_T = TypeVar("_T")

OCTET_STREAM = "application/octet-stream"
JSON_CONTENT_TYPE = "application/json"
_DELETE_BATCH_SIZE = 1000  # S3 DeleteObjects limit
_COPY_BUFFER_BYTES = 1 << 20
_TRANSIENT_S3_CODES = frozenset(
    {"RequestTimeout", "SlowDown", "Throttling", "ServiceUnavailable", "InternalError"}
)
_MISSING_KEY_CODES = frozenset({"NoSuchKey", "404", "NotFound"})
# A suffix range on an empty object is unsatisfiable (HTTP 416).
_EMPTY_RANGE_CODES = frozenset({"InvalidRange", "416"})


class ArtifactNotFoundError(KeyError):
    """The requested key does not exist in the store."""


class ArtifactStoreError(RuntimeError):
    """The store rejected an operation."""


def validate_key(key: str) -> str:
    """Reject keys that are absolute, empty or escape the store root."""
    path = PurePosixPath(key)
    if not key or key.startswith("/") or ".." in path.parts or key.endswith("/"):
        raise ValueError(f"invalid artifact key: {key!r}")
    return key


class ArtifactStore(abc.ABC):
    """Key/value object store with atomic whole-object writes."""

    @abc.abstractmethod
    def put_bytes(self, key: str, data: bytes, content_type: str = OCTET_STREAM) -> None:
        """Write ``data`` to ``key``, replacing any previous object."""

    @abc.abstractmethod
    def put_file(self, key: str, path: Path, content_type: str = OCTET_STREAM) -> None:
        """Upload the local file ``path`` to ``key``, replacing any previous object."""

    @abc.abstractmethod
    def get_bytes(self, key: str) -> bytes:
        """Read the object at ``key``; raise ArtifactNotFoundError when it is missing."""

    @abc.abstractmethod
    def get_tail(self, key: str, length: int) -> bytes:
        """Read the last ``length`` bytes of the object at ``key`` (all of it when shorter)."""

    @abc.abstractmethod
    def download_file(self, key: str, path: Path) -> None:
        """Stream the object at ``key`` into the local file ``path`` without buffering it in memory."""

    @abc.abstractmethod
    def list_keys(self, prefix: str) -> list[str]:
        """Return all keys starting with ``prefix``, sorted lexicographically."""

    @abc.abstractmethod
    def delete_keys(self, keys: Sequence[str]) -> None:
        """Delete ``keys``; keys that do not exist are ignored."""

    @abc.abstractmethod
    def uri(self, key: str) -> str:
        """Return a human-readable URI of ``key`` for logs and metadata."""

    def put_json(self, key: str, payload: Any) -> None:
        """Write ``payload`` as pretty-printed JSON."""
        body = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        self.put_bytes(key, body, JSON_CONTENT_TYPE)

    def get_json(self, key: str) -> Any:
        """Read and parse the JSON object at ``key``."""
        return json.loads(self.get_bytes(key))


def _is_transient_s3_error(exc: Exception) -> bool:
    if isinstance(exc, ClientError):
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        code = exc.response.get("Error", {}).get("Code", "")
        return status >= 500 or code in _TRANSIENT_S3_CODES
    return isinstance(exc, BotoConnectionError | HTTPClientError)


def _is_missing_key(exc: ClientError) -> bool:
    return exc.response.get("Error", {}).get("Code", "") in _MISSING_KEY_CODES


class S3ArtifactStore(ArtifactStore):
    """ArtifactStore backed by one S3 / MinIO bucket."""

    def __init__(self, settings: S3Settings, *, attempts: int = 5) -> None:
        self._bucket = settings.bucket
        self._attempts = attempts
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.endpoint_url,
            aws_access_key_id=settings.access_key_id,
            aws_secret_access_key=settings.secret_access_key,
            region_name=settings.region,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                connect_timeout=10,
                read_timeout=120,
                # Retries are handled explicitly by retry_call; botocore must not stack its own.
                retries={"total_max_attempts": 1},
                # S3-compatible stores do not all accept the optional CRC trailers newer
                # botocore releases send by default.
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
            ),
        )

    def _retry(self, description: str, operation: Callable[[], _T]) -> _T:
        return retry_call(
            operation,
            description=description,
            is_retryable=_is_transient_s3_error,
            attempts=self._attempts,
        )

    def put_bytes(self, key: str, data: bytes, content_type: str = OCTET_STREAM) -> None:
        """Write ``data`` to ``key`` with a single (atomic) PUT."""
        validate_key(key)
        self._retry(
            f"put {self.uri(key)}",
            lambda: self._client.put_object(
                Bucket=self._bucket, Key=key, Body=data, ContentType=content_type
            ),
        )

    def put_file(self, key: str, path: Path, content_type: str = OCTET_STREAM) -> None:
        """Upload ``path`` with a single (atomic) PUT; every attempt re-reads the file."""
        validate_key(key)

        def _upload() -> None:
            with path.open("rb") as body:
                self._client.put_object(
                    Bucket=self._bucket,
                    Key=key,
                    Body=body,
                    ContentType=content_type,
                    ContentLength=path.stat().st_size,
                )

        self._retry(f"upload {self.uri(key)}", _upload)

    def _get_object(self, key: str, **request: str) -> Any:
        """GetObject, mapping a missing key to ArtifactNotFoundError."""
        try:
            return self._client.get_object(Bucket=self._bucket, Key=validate_key(key), **request)
        except ClientError as exc:
            if _is_missing_key(exc):
                raise ArtifactNotFoundError(self.uri(key)) from exc
            raise

    def get_bytes(self, key: str) -> bytes:
        """Read the object at ``key``."""
        validate_key(key)

        def _get() -> bytes:
            body = self._get_object(key)["Body"]
            try:
                return body.read()
            finally:
                body.close()

        return self._retry(f"get {self.uri(key)}", _get)

    def get_tail(self, key: str, length: int) -> bytes:
        """Read the last ``length`` bytes with one suffix-range GET."""
        validate_key(key)
        if length < 1:
            raise ValueError("length must be >= 1")

        def _get() -> bytes:
            try:
                body = self._get_object(key, Range=f"bytes=-{length}")["Body"]
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code", "") in _EMPTY_RANGE_CODES:
                    return b""
                raise
            try:
                return body.read()
            finally:
                body.close()

        return self._retry(f"get the last {length} bytes of {self.uri(key)}", _get)

    def download_file(self, key: str, path: Path) -> None:
        """Stream the object into ``path`` in 1 MiB chunks; every attempt rewrites the file."""
        validate_key(key)

        def _download() -> None:
            body = self._get_object(key)["Body"]
            try:
                with path.open("wb") as sink:
                    shutil.copyfileobj(body, sink, _COPY_BUFFER_BYTES)
            finally:
                body.close()

        self._retry(f"download {self.uri(key)}", _download)

    def list_keys(self, prefix: str) -> list[str]:
        """List every key under ``prefix`` (all pages)."""

        def _list() -> list[str]:
            paginator = self._client.get_paginator("list_objects_v2")
            keys: list[str] = []
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                keys.extend(item["Key"] for item in page.get("Contents", []))
            return sorted(keys)

        return self._retry(f"list {self.uri(prefix)}", _list)

    def delete_keys(self, keys: Sequence[str]) -> None:
        """Delete ``keys`` in batches of up to 1000 objects."""
        for start in range(0, len(keys), _DELETE_BATCH_SIZE):
            batch = [{"Key": validate_key(key)} for key in keys[start : start + _DELETE_BATCH_SIZE]]
            response = self._retry(
                f"delete {len(batch)} objects from s3://{self._bucket}",
                lambda batch=batch: self._client.delete_objects(
                    Bucket=self._bucket, Delete={"Objects": batch, "Quiet": True}
                ),
            )
            errors = response.get("Errors", [])
            if errors:
                raise ArtifactStoreError(f"could not delete {len(errors)} objects, first: {errors[0]}")

    def uri(self, key: str) -> str:
        """Return ``s3://<bucket>/<key>``."""
        return f"s3://{self._bucket}/{key}"


class LocalArtifactStore(ArtifactStore):
    """ArtifactStore backed by a local directory, with the same key layout as S3."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._root.joinpath(*PurePosixPath(validate_key(key)).parts)

    def _replace_atomically(self, key: str, write: Callable[[Path], None]) -> None:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            write(temp_path)
            os.replace(temp_path, target)
        finally:
            temp_path.unlink(missing_ok=True)

    def put_bytes(self, key: str, data: bytes, content_type: str = OCTET_STREAM) -> None:
        """Write ``data`` via a temporary file and an atomic rename."""
        self._replace_atomically(key, lambda temp: temp.write_bytes(data))

    def put_file(self, key: str, path: Path, content_type: str = OCTET_STREAM) -> None:
        """Copy ``path`` via a temporary file and an atomic rename."""
        self._replace_atomically(key, lambda temp: shutil.copyfile(path, temp))

    def _existing_path(self, key: str) -> Path:
        path = self._path(key)
        if not path.is_file():
            raise ArtifactNotFoundError(self.uri(key))
        return path

    def get_bytes(self, key: str) -> bytes:
        """Read the file stored at ``key``."""
        return self._existing_path(key).read_bytes()

    def get_tail(self, key: str, length: int) -> bytes:
        """Read the last ``length`` bytes of the file stored at ``key``."""
        if length < 1:
            raise ValueError("length must be >= 1")
        path = self._existing_path(key)
        with path.open("rb") as source:
            source.seek(max(0, path.stat().st_size - length))
            return source.read()

    def download_file(self, key: str, path: Path) -> None:
        """Copy the file stored at ``key`` to ``path``."""
        shutil.copyfile(self._existing_path(key), path)

    def list_keys(self, prefix: str) -> list[str]:
        """List stored keys under ``prefix``, ignoring in-flight temporary files."""
        keys = (
            path.relative_to(self._root).as_posix()
            for path in self._root.rglob("*")
            if path.is_file() and not path.name.startswith(".")
        )
        return sorted(key for key in keys if key.startswith(prefix))

    def delete_keys(self, keys: Sequence[str]) -> None:
        """Remove the files stored at ``keys``."""
        for key in keys:
            self._path(key).unlink(missing_ok=True)

    def uri(self, key: str) -> str:
        """Return ``file:///<root>/<key>``, unescaped so it matches the S3 key layout in logs."""
        root = self._root.as_posix()
        return f"file://{root if root.startswith('/') else '/' + root}/{key}"
