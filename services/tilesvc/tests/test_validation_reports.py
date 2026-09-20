import json

from services.tilesvc import validation_reports as vr


def test_a_successful_run_is_readable_back(tmp_path):
    vr.write_report({"status": "ok", "event": "palisades",
                     "checkpoint": "control60.pt", "weighted_cos": -0.28},
                    directory=tmp_path)

    runs = vr.list_reports(tmp_path)

    assert len(runs) == 1
    assert runs[0]["status"] == "ok"
    assert runs[0]["weighted_cos"] == -0.28
    assert runs[0]["recorded_at"]


def test_failures_are_recorded_not_lost(tmp_path):
    # The whole point: a job that dies leaves no stdout behind.
    vr.write_report({"status": "error", "event": "palisades",
                     "checkpoint": "control60.pt",
                     "error": {"type": "FileNotFoundError", "message": "/tmp/c60.pt"}},
                    directory=tmp_path)

    run = vr.list_reports(tmp_path)[0]

    assert run["status"] == "error"
    assert run["error"]["type"] == "FileNotFoundError"


def test_runs_come_back_newest_first(tmp_path):
    for stamp in ("2026-08-01T00:00:00+00:00", "2026-08-29T00:00:00+00:00", "2026-08-15T00:00:00+00:00"):
        vr.write_report({"recorded_at": stamp, "event": "palisades", "checkpoint": "c.pt"},
                        directory=tmp_path)

    stamps = [r["recorded_at"] for r in vr.list_reports(tmp_path)]

    assert stamps == sorted(stamps, reverse=True)


def test_report_count_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(vr, "MAX_REPORTS", 3)
    for i in range(8):
        vr.write_report({"recorded_at": f"2026-08-{i + 1:02d}T00:00:00+00:00",
                         "event": "palisades", "checkpoint": "c.pt"}, directory=tmp_path)

    assert len(list(tmp_path.glob("*.json"))) <= 3


def test_an_unreadable_report_is_surfaced_rather_than_skipped(tmp_path):
    vr.write_report({"event": "palisades", "checkpoint": "c.pt"}, directory=tmp_path)
    (tmp_path / "20260829-corrupt-c.pt.json").write_text("{not json", encoding="utf-8")

    statuses = [r.get("status") for r in vr.list_reports(tmp_path)]

    assert "unreadable" in statuses


def test_missing_directory_is_empty_not_an_error(tmp_path):
    assert vr.list_reports(tmp_path / "absent") == []


def test_checkpoint_names_cannot_escape_the_directory(tmp_path):
    vr.write_report({"event": "../../etc", "checkpoint": "../../passwd"}, directory=tmp_path)

    written = list(tmp_path.glob("*.json"))

    assert len(written) == 1
    assert written[0].parent == tmp_path
