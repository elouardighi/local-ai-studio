"""Lifecycle state-machine tests (no engine, no GPU required)."""
from __future__ import annotations

import pytest

from backend.app.services.instance_manager import (
    ACTIVE_STATES, Instance, IllegalTransition, TRANSITIONS, VALID_STATES,
)


def make() -> Instance:
    return Instance(instance_id="abc123", model_id="m1", model_name="Test",
                    model_path="C:/models/test.gguf")


def test_happy_path_reaches_ready():
    inst = make()
    inst.transition("validating")
    inst.transition("estimating_memory")
    inst.transition("queued")
    inst.transition("starting_engine")
    inst.transition("loading_model")
    inst.transition("ready", ready_at=1.0)
    assert inst.is_loaded
    assert inst.state == "ready"


def test_ready_busy_ready_cycle():
    inst = make()
    for state in ("validating", "estimating_memory", "queued",
                  "starting_engine", "loading_model", "ready"):
        inst.transition(state)
    inst.transition("busy")
    assert inst.is_loaded and inst.state == "busy"
    inst.transition("ready")
    assert inst.state == "ready"


def test_illegal_jump_raises():
    inst = make()
    with pytest.raises(IllegalTransition):
        inst.transition("ready")      # discovered -> ready is invalid
    assert inst.state == "discovered"  # state must not silently change


def test_unknown_state_raises():
    inst = make()
    with pytest.raises(IllegalTransition):
        inst.transition("exploded")


def test_failure_states_are_reachable_and_terminal():
    inst = make()
    for state in ("validating", "estimating_memory", "queued",
                  "starting_engine"):
        inst.transition(state)
    inst.transition("engine_failed")
    assert not inst.is_loaded
    with pytest.raises(IllegalTransition):
        inst.transition("ready")       # failed -> ready is never allowed


def test_crash_from_ready():
    inst = make()
    for state in ("validating", "estimating_memory", "queued",
                  "starting_engine", "loading_model", "ready"):
        inst.transition(state)
    inst.transition("crashed")
    assert inst.state == "crashed"
    assert not inst.is_active


def test_cancel_is_possible_during_load():
    inst = make()
    inst.transition("validating")
    inst.transition("estimating_memory")
    inst.transition("cancelled")
    assert inst.state == "cancelled"


def test_every_valid_state_is_reachable_from_somewhere():
    """Guards against adding a state that can never be entered."""
    reachable = {"discovered"}
    changed = True
    while changed:
        changed = False
        for src, targets in TRANSITIONS.items():
            for dst in targets:
                if src in reachable and dst not in reachable:
                    reachable.add(dst)
                    changed = True
    missing = VALID_STATES - reachable
    assert not missing, f"unreachable states: {missing}"


def test_active_states_are_a_subset_of_valid():
    assert ACTIVE_STATES <= VALID_STATES


def test_unloaded_is_terminal():
    inst = make()
    for state in ("validating", "estimating_memory", "queued",
                  "starting_engine", "loading_model", "ready", "unloading",
                  "unloaded"):
        inst.transition(state)
    assert TRANSITIONS["unloaded"] == set()
    with pytest.raises(IllegalTransition):
        inst.transition("ready")
