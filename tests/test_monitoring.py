from src.utils import monitoring


def test_monitoring_writes_metrics(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    monitoring_file = reports_dir / "api_monitoring.csv"

    monkeypatch.setattr(monitoring, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(monitoring, "MONITORING_FILE", monitoring_file)

    monitoring.record_api_metrics(
        status_code=200,
        response_time_ms=125.567,
        cache_hit=False,
        recommendation_count=5,
    )

    monitoring.record_api_metrics(
        status_code=200,
        response_time_ms=1.25,
        cache_hit=True,
        recommendation_count=5,
    )

    rows = monitoring_file.read_text(encoding="utf-8").splitlines()

    assert len(rows) == 3
    assert rows[0] == (
        "timestamp,status_code,response_time_ms,cache_hit,"
        "recommendation_count"
    )
    assert ",200,125.57,False,5" in rows[1]
    assert ",200,1.25,True,5" in rows[2]


def test_monitoring_records_failed_requests(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    monitoring_file = reports_dir / "api_monitoring.csv"

    monkeypatch.setattr(monitoring, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(monitoring, "MONITORING_FILE", monitoring_file)

    monitoring.record_api_metrics(
        status_code=500,
        response_time_ms=250,
        cache_hit=False,
        recommendation_count=0,
    )

    rows = monitoring_file.read_text(encoding="utf-8").splitlines()

    assert len(rows) == 2
    assert ",500,250,False,0" in rows[1]