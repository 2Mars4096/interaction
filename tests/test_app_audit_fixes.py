import io
import json
from contextlib import redirect_stdout
from types import SimpleNamespace

import pytest

from interaction.app import main
from interaction.contracts import ActionName, ActionProposal, EnvironmentSnapshot, ExecutionResult, ExecutionStatus, RiskLevel, SessionMode
from interaction.control import CommandBroker
from interaction.persistence import JsonStateStore, RuntimePaths
from interaction.platform import MacOSPlatformAdapter
from interaction.platform.macos_runtime import CGPoint, _post_mouse_event
from interaction.runtime import VoiceCommandLoop
from interaction.runtime.webcam import run_live_cursor_follow
from interaction.vision import CalibrationProfile, GazeSample, NormalizedPoint


class Provider:
    def __init__(self, interrupt=False):
        self.reads = 0
        self.interrupt = interrupt
        self.closed = False

    def open(self):
        pass

    def close(self):
        self.closed = True

    def read(self, **kwargs):
        self.reads += 1
        if self.interrupt and self.reads == 3:
            raise KeyboardInterrupt
        return SimpleNamespace(sample=GazeSample(point=NormalizedPoint(.8, .8), confidence=.95, delta_ms=100))


class DeniedAdapter(MacOSPlatformAdapter):
    def execute(self, request):
        return ExecutionResult(status=ExecutionStatus.FAILURE, message="Accessibility denied", proposal=request.proposal)


def run_cursor(provider, adapter, **kwargs):
    return run_live_cursor_follow(provider, CalibrationProfile(), adapter=adapter, frames=1000,
        delta_ms=100, min_move_distance=0, smoothing=1, max_step=1,
        response_gain_x=1, response_gain_y=1, initial_cursor_point=(.5,.5), **kwargs)


def test_failed_cursor_execution_is_emitted_immediately_and_stops():
    provider = Provider()
    delivered = []
    events, summary = run_cursor(provider, DeniedAdapter(), event_sink=delivered.append)
    assert provider.reads == 1
    assert summary['status'] == 'error'
    assert summary['moved_events'] == 0
    assert summary['failed_moves'] == 1
    assert 'Accessibility denied' in delivered[0].message
    assert all('Moved cursor' not in event.message for event in events)


def test_preview_is_not_counted_as_executed_movement():
    _, summary = run_cursor(Provider(), MacOSPlatformAdapter(dry_run=True))
    assert summary['preview_moves'] > 0
    assert summary['moved_events'] == summary['executed_moves'] == 0


def test_cursor_history_bounded_and_stop_reported(monkeypatch):
    ticks = iter(range(10000))
    monkeypatch.setattr('interaction.runtime.webcam.time.monotonic', lambda: next(ticks))
    events, _ = run_cursor(Provider(), MacOSPlatformAdapter(dry_run=True))
    assert len(events) <= 256
    delivered = []
    _, summary = run_cursor(Provider(interrupt=True), MacOSPlatformAdapter(dry_run=True), event_sink=delivered.append)
    assert summary['status'] == 'stopped'
    assert any('stopped' in event.message for event in delivered)


def test_live_gaze_stop_persists_events_once(monkeypatch, tmp_path, capsys):
    store = JsonStateStore(RuntimePaths(tmp_path))
    store.save_calibration_profile(CalibrationProfile(), name='webcam-live')
    provider = Provider(interrupt=True)
    original_read = provider.read
    def read_with_durable_log_check(**kwargs):
        if provider.reads == 2:
            # Verify durability before interruption, not only after final payload generation.
            logs = list((tmp_path / 'logs').glob('*.jsonl'))
            assert logs and logs[0].read_text().strip()
        return original_read(**kwargs)
    provider.read = read_with_durable_log_check
    monkeypatch.setattr('interaction.app._build_webcam_provider', lambda *args, **kwargs: provider)
    monkeypatch.setattr('interaction.app._current_cursor_point', lambda: (.5,.5))
    output = io.StringIO()
    with redirect_stdout(output):
        main(['gaze-live', '--action', 'cursor', '--frames', '1000', '--dry-run', '--runtime-dir', str(tmp_path)])
    payload = json.loads(output.getvalue())
    records = [json.loads(line) for line in open(payload['session_log'])]
    assert payload['live_gaze']['status'] == 'stopped'
    assert provider.closed
    assert sum('stopped' in (record.get('message') or '') for record in records) == 1
    assert any(record.get('result') for record in records)
    assert 'stopped' in capsys.readouterr().err
    assert [record['index'] for record in records] == list(range(1, len(records)+1))


def test_live_dictation_correction_does_not_dispatch_backspaces():
    commands = []
    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=False, runner=commands.append))
    env = EnvironmentSnapshot(active_app='Notes', active_window_title='Draft')
    loop.run_text_turn('hello team', env, mode=SessionMode.DICTATION)
    assert commands
    commands.clear()
    events = loop.run_text_turn('scratch that', env, mode=SessionMode.DICTATION)
    assert commands == []
    assert any('caret cannot be verified' in (event.message or '') for event in events)


def test_adapter_blocks_even_confirmed_unverified_deletion():
    commands = []
    adapter = MacOSPlatformAdapter(dry_run=False, runner=commands.append)
    broker = CommandBroker()
    proposal = ActionProposal(action=ActionName.TYPE_TEXT, arguments={'delete_characters':5,'text':'replacement'},
        risk=RiskLevel.L2, requires_confirmation=True, confidence=.99, rationale='test')
    request = broker.build_execution_request(broker.confirm(broker.decide(proposal)), EnvironmentSnapshot(active_app='Notes'))
    result = adapter.execute(request)
    assert result.status == ExecutionStatus.BLOCKED
    assert not commands


def test_pointer_permission_denial_does_not_post_an_event():
    quartz = SimpleNamespace(AXIsProcessTrusted=lambda: False)
    with pytest.raises(PermissionError, match='Accessibility'):
        _post_mouse_event(quartz, 5, CGPoint(10,10), 0)
