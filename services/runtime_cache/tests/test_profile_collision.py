"""A cache directory must hold one event's snapshots, not a mix.

FIRMS snapshot files are named by date alone, but their contents are scoped to
the event's bounding box - palisades_mid and eaton_mid each have a
2025-01-08.csv holding different detections over different longitudes. Syncing
one into a directory that already holds the other overwrites it silently, and
that fire is then forecast from another fire's detections.
"""
import pytest

from services.runtime_cache.pipeline import ProfileCollision, _guard_profile_marker


def test_first_profile_claims_the_directory(tmp_path):
    _guard_profile_marker(tmp_path, "palisades_mid")

    assert (tmp_path / ".runtime_cache_profile").read_text() == "palisades_mid"


def test_resyncing_the_same_profile_is_allowed(tmp_path):
    _guard_profile_marker(tmp_path, "palisades_mid")
    _guard_profile_marker(tmp_path, "palisades_mid")   # must not raise


def test_a_different_profile_is_refused(tmp_path):
    _guard_profile_marker(tmp_path, "palisades_mid")

    with pytest.raises(ProfileCollision) as excinfo:
        _guard_profile_marker(tmp_path, "eaton_mid")

    message = str(excinfo.value)
    assert "palisades_mid" in message and "eaton_mid" in message
    assert "per-profile path" in message, "the error must say how to fix it"


def test_an_empty_marker_does_not_block_a_sync(tmp_path):
    (tmp_path / ".runtime_cache_profile").write_text("")

    _guard_profile_marker(tmp_path, "camp_mid")

    assert (tmp_path / ".runtime_cache_profile").read_text() == "camp_mid"
