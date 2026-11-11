from dataclasses import asdict
import json

import pytest

from interaction.persistence import JsonStateStore, RuntimePaths
from interaction.vision import CalibrationProfile, CalibrationSample, NormalizedPoint


def validated_profile(camera_index=0):
    fitting = [NormalizedPoint(x, y) for x, y in [(0.2, 0.2), (0.8, 0.2), (0.5, 0.5), (0.2, 0.8), (0.8, 0.8)]]
    validation = [NormalizedPoint(x, y) for x, y in [(0.35, 0.35), (0.65, 0.35), (0.35, 0.65), (0.65, 0.65)]]
    return CalibrationProfile.fit([CalibrationSample(point, point) for point in fitting]).validate(
        [CalibrationSample(point, point) for point in validation],
        raw_stddevs=[NormalizedPoint(0.001, 0.001)] * 4, sample_counts=[6] * 4,
        camera_index=camera_index,
    )


def test_validated_profile_round_trips_typed_evidence_and_camera(tmp_path):
    store = JsonStateStore(RuntimePaths(tmp_path))
    profile = validated_profile(camera_index=2)
    path = store.save_calibration_profile(profile, name="webcam-live")

    loaded = store.load_calibration_profile("webcam-live", require_validated=True, camera_index=2)
    assert loaded == profile
    assert loaded.is_usable(camera_index=2)
    assert json.loads(path.read_text())["quality"]["points"][0]["sample_count"] == 6
    assert store.load_calibration_profile("webcam-live", require_validated=True, camera_index=0) is None
    assert store.load_calibration_profile("webcam-live", require_validated=True, tracker_id="new-tracker") is None


def test_legacy_four_number_profile_is_script_readable_but_cannot_enable_live_control(tmp_path):
    store = JsonStateStore(RuntimePaths(tmp_path))
    legacy = {"x_scale": 1.1, "y_scale": 0.9, "x_offset": 0.01, "y_offset": -0.01}
    path = store.paths.profiles_dir / "legacy.json"
    path.write_text(json.dumps(legacy))

    profile = store.load_calibration_profile("legacy")
    assert profile is not None
    assert profile.schema_version == 1
    assert profile.x_scale == 1.1
    assert not profile.is_usable()
    assert store.load_calibration_profile("legacy", require_validated=True) is None
    assert store.load_calibration_profile("legacy", camera_index=0) is None


def test_unvalidated_versioned_profile_is_only_available_to_scripted_use(tmp_path):
    store = JsonStateStore(RuntimePaths(tmp_path))
    profile = CalibrationProfile(x_scale=1.2)
    store.save_calibration_profile(profile)

    assert store.load_calibration_profile() == profile
    assert store.load_calibration_profile(require_validated=True) is None


@pytest.mark.parametrize("contents", ["{", "null", "[]", "{}", '{"x_scale": 1}', '{"x_scale": NaN, "y_scale": 1, "x_offset": 0, "y_offset": 0}'])
def test_corrupt_or_incomplete_profiles_are_unavailable_without_crashing(tmp_path, contents):
    store = JsonStateStore(RuntimePaths(tmp_path))
    (store.paths.profiles_dir / "bad.json").write_text(contents)

    assert store.load_calibration_profile("bad") is None
    assert store.load_calibration_profile("bad", require_validated=True) is None


@pytest.mark.parametrize("mutation", [
    lambda payload: payload.update(schema_version=999),
    lambda payload: payload.update(schema_version=True),
    lambda payload: payload.update(x_scale=float("inf")),
    lambda payload: payload.update(camera_index=True),
    lambda payload: payload.update(quality="validated"),
    lambda payload: payload["quality"].update(mean_error=0.1),
    lambda payload: payload["quality"].update(sample_count=999),
    lambda payload: payload["quality"]["points"][0].update(error=False),
    lambda payload: payload["quality"]["points"][0].update(sample_count=2),
    lambda payload: payload["quality"]["points"][0]["raw_stddev"].update(x=-0.01),
    lambda payload: payload["quality"]["points"][0]["projected"].update(x=0.99),
    lambda payload: payload["quality"]["points"][0].update(expected=payload["training_expected"][0]),
])
def test_malformed_or_inconsistent_evidence_never_becomes_live_usable(tmp_path, mutation):
    store = JsonStateStore(RuntimePaths(tmp_path))
    payload = asdict(validated_profile())
    mutation(payload)
    (store.paths.profiles_dir / "bad.json").write_text(json.dumps(payload))

    assert store.load_calibration_profile("bad") is None
    assert store.load_calibration_profile("bad", require_validated=True) is None


def test_missing_or_unreadable_profile_returns_none(tmp_path, monkeypatch):
    store = JsonStateStore(RuntimePaths(tmp_path))
    assert store.load_calibration_profile("missing") is None
    path = store.paths.profiles_dir / "unreadable.json"
    path.write_bytes(b"\xff\xfe")
    assert store.load_calibration_profile("unreadable") is None


def test_failed_atomic_replace_preserves_last_valid_profile_and_cleans_tempfile(tmp_path, monkeypatch):
    store = JsonStateStore(RuntimePaths(tmp_path))
    original = validated_profile(camera_index=0)
    path = store.save_calibration_profile(original, name="webcam-live")
    original_bytes = path.read_bytes()

    def fail_replace(source, destination):
        assert destination == path
        assert source.parent == destination.parent
        assert json.loads(source.read_text())["camera_index"] == 1
        assert destination.read_bytes() == original_bytes
        raise OSError("Simulated interrupted replace")

    monkeypatch.setattr("interaction.persistence.store.os.replace", fail_replace)
    with pytest.raises(OSError, match="interrupted replace"):
        store.save_calibration_profile(validated_profile(camera_index=1), name="webcam-live")

    assert path.read_bytes() == original_bytes
    assert store.load_calibration_profile("webcam-live", require_validated=True, camera_index=0) == original
    assert list(store.paths.profiles_dir.iterdir()) == [path]
