"""Resource monitor tests.

Covers the bug where a bound method was passed where an iterable was expected,
which silently stopped every sample from being recorded.
"""
from __future__ import annotations

import pytest

from backend.app.services.event_bus import EventBus
from backend.app.services.log_store import LogStore
from backend.app.services.resource_monitor import (
    ResourceMonitor, _aggregate_engine_counters, _luid_of, _pid_of_engine,
    get_commit_status,
)


@pytest.fixture
def monitor(tmp_path) -> ResourceMonitor:
    from backend.app.config import Settings
    settings = Settings(converted_dir=str(tmp_path / "converted"))
    return ResourceMonitor(
        get_settings=lambda: settings,
        events=EventBus(),
        log=LogStore(),
        history_max=10,
    )


def test_sample_returns_expected_shape(monitor: ResourceMonitor):
    sample = monitor.sample()
    assert sample["ts"] > 0
    assert "cpu_percent" in sample
    assert sample["ram_total_bytes"] > 0
    assert "gpu" in sample
    # GPU counters may be unavailable on some systems - report, never invent.
    assert isinstance(sample["gpu"]["available"], bool)


def test_samples_accumulate_in_bounded_history(monitor: ResourceMonitor):
    for _ in range(3):
        monitor.sample()
    assert len(monitor.history()) == 3
    assert len(monitor.history(seconds=0)) <= 3


def test_history_respects_maxlen(monitor: ResourceMonitor):
    for _ in range(25):
        monitor.sample()
    assert len(monitor.history()) <= 10


def test_instances_ref_may_be_a_callable(monitor: ResourceMonitor):
    """Regression: passing a bound method must not break sampling."""
    monitor.start(lambda: [{
        "instance_id": "i1", "state": "ready", "pid": None,
    }])
    sample = monitor.sample()
    assert "i1" in sample["instances"]


def test_instances_ref_may_be_a_list(monitor: ResourceMonitor):
    monitor.start([{"instance_id": "i2", "state": "busy", "pid": None}])
    sample = monitor.sample()
    assert "i2" in sample["instances"]


def test_broken_instances_ref_does_not_crash(monitor: ResourceMonitor):
    monitor.start(12345)  # nonsense on purpose
    sample = monitor.sample()  # must not raise
    assert sample["instances"] == {}


def test_last_sample_returns_most_recent(monitor: ResourceMonitor):
    assert monitor.last_sample() == {}
    first = monitor.sample()
    assert monitor.last_sample()["ts"] == first["ts"]


def test_commit_status_reports_real_windows_numbers():
    commit = get_commit_status()
    if not commit:
        pytest.skip("GlobalMemoryStatusEx unavailable on this platform")
    assert commit["commit_limit_bytes"] > 0
    assert commit["commit_used_bytes"] <= commit["commit_limit_bytes"]


def test_pdh_instance_name_parsing():
    name = "pid_1234_luid_0x00000000_0x000103db_phys_0_eng_2_engtype_compute 0"
    assert _pid_of_engine(name) == 1234
    assert _luid_of(name) == "luid_0x00000000_0x000103db"
    assert _pid_of_engine("luid_0x00000000_0x1_phys_0") is None


def test_engine_counters_map_pid_to_adapter_luid():
    """Regression: the instance's GPU adapter was always reported as null."""
    engines = {
        "pid_22092_luid_0x00000000_0x000103db_phys_0_eng_2_engtype_compute 0": 3.5,
        "pid_22092_luid_0x00000000_0x000103db_phys_0_eng_7_engtype_copy 0": 1.2,
    }
    agg = _aggregate_engine_counters(engines)
    assert agg["per_pid_utilization"] == {"22092": 4.7}   # engine types summed
    assert agg["per_pid_luid"]["22092"] == "luid_0x00000000_0x000103db"
    assert agg["per_luid_utilization"]["luid_0x00000000_0x000103db"] == 4.7


def test_engine_counter_aggregation_tolerates_partial_instances():
    # Adapter-only instance: no pid to attribute to.
    agg = _aggregate_engine_counters(
        {"luid_0x00000000_0x000103db_phys_0": 5.0})
    assert agg["per_pid_utilization"] == {}
    assert agg["per_pid_luid"] == {}
    assert agg["per_luid_utilization"] == {
        "luid_0x00000000_0x000103db": 5.0}
    # Pid without a LUID token: utilization counted, luid stays unknown
    # rather than being guessed from the raw counter name.
    agg = _aggregate_engine_counters({"pid_77_phys_0_eng_1": 2.0})
    assert agg["per_pid_utilization"] == {"77": 2.0}
    assert agg["per_pid_luid"] == {}
    assert agg["per_luid_utilization"] == {}
