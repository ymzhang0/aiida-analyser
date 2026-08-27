"""Read-only access to AiiDA ``.aiida`` archives.

Archives use AiiDA's ``core.sqlite_zip`` backend. They must be loaded as a
profile before the ORM can query them, but registering that profile globally
is unnecessary and can disturb the caller's active profile. This module
provides a small, scoped interface around AiiDA's native archive backend.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from aiida import profile_context
from aiida.manage.configuration import Profile
from aiida.storage.sqlite_zip.backend import SqliteZipBackend


def create_archive_profile(filepath: str | Path) -> Profile:
    """Create the transient, read-only profile for an AiiDA archive.

    The returned profile is not written to ``config.json``. Prefer
    :func:`archive_context` for normal use so that it is loaded only while
    archive nodes are being inspected.

    :raises FileNotFoundError: if *filepath* does not exist.
    :raises IsADirectoryError: if *filepath* is not a file.
    """
    path = Path(filepath).expanduser()

    if not path.exists():
        raise FileNotFoundError(f'AiiDA archive does not exist: {path}')
    if not path.is_file():
        raise IsADirectoryError(f'AiiDA archive path is not a file: {path}')

    return SqliteZipBackend.create_profile(path.resolve())


@contextmanager
def archive_context(filepath: str | Path) -> Iterator[Profile]:
    """Temporarily make an AiiDA archive the active, read-only profile.

    AiiDA ORM calls made inside this context query the archive through the
    ``core.sqlite_zip`` backend. The caller's previously active profile is
    restored automatically when the context exits, including after errors.

    Example:
        >>> from aiida import orm
        >>> from aiida_analyser.core import archive_context
        >>> with archive_context('calculation.aiida'):
        ...     groups = orm.QueryBuilder().append(orm.Group).all(flat=True)
    """
    profile = create_archive_profile(filepath)
    with profile_context(profile, allow_switch=True):
        yield profile
