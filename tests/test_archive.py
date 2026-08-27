from pathlib import Path

import pytest
from aiida import orm
from aiida.manage import get_manager
from aiida.storage.sqlite_zip.backend import SqliteZipBackend

from aiida_analyser import (
    archive_context,
    count_groups,
    count_nodes,
    create_archive_profile,
    get_and_count_types,
)


def _create_archive(path):
    """Create an empty, valid archive without registering a profile."""
    profile = SqliteZipBackend.create_profile(path)
    assert SqliteZipBackend.initialise(profile)


def test_archive_context_loads_read_only_sqlite_profile_and_unloads(tmp_path):
    archive = tmp_path / 'example.aiida'
    _create_archive(archive)

    previous_profile = get_manager().get_profile()

    with archive_context(archive) as profile:
        assert profile.storage_backend == 'core.sqlite_zip'
        assert Path(profile.storage_config['filepath']) == archive.resolve()
        assert get_manager().get_profile() is profile
        assert get_manager().get_profile_storage().read_only
        assert orm.QueryBuilder().append(orm.Node).count() == 0
        log = []
        count_groups(log=log.append)
        get_and_count_types(log=log.append)
        assert count_nodes('data.core.dict.Dict.', None) == 0

        assert log == [
            f"{'pk':<10} {'label':<35} {'count':<10}",
            f"{'count':<10} {'node_type':<50} {'process_type':<10}",
        ]

    assert get_manager().get_profile() is previous_profile


def test_archive_context_restores_outer_profile(tmp_path):
    outer_archive = tmp_path / 'outer.aiida'
    inner_archive = tmp_path / 'inner.aiida'
    _create_archive(outer_archive)
    _create_archive(inner_archive)

    with archive_context(outer_archive) as outer_profile:
        with archive_context(inner_archive):
            assert get_manager().get_profile().name == inner_archive.name
        assert get_manager().get_profile() is outer_profile


def test_create_archive_profile_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match='does not exist'):
        create_archive_profile(tmp_path / 'missing.aiida')
