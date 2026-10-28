from __future__ import annotations

from store_support.governance.kill_switch import KillSwitch


def test_default_off() -> None:
    assert KillSwitch(env={}).engaged is False


def test_env_true_engages() -> None:
    assert KillSwitch(env={"KILL_SWITCH": "true"}).engaged is True


def test_file_flag_engages_without_restart(tmp_path) -> None:
    flag = tmp_path / "KILL"
    ks = KillSwitch(env={}, flag_file=flag)
    assert ks.engaged is False
    flag.write_text("1")
    assert ks.engaged is True  # re-read each call: flip at runtime, no redeploy
    flag.unlink()
    assert ks.engaged is False


def test_env_wins_over_missing_file(tmp_path) -> None:
    assert KillSwitch(env={"KILL_SWITCH": "1"}, flag_file=tmp_path / "nope").engaged is True
