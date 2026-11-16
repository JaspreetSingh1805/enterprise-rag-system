from app.core.logging import (
    get_current_trace_id,
    set_current_trace_id,
)


def test_trace_id_propagation():
    test_id = "test-trace-uuid-12345"
    set_current_trace_id(test_id)
    assert get_current_trace_id() == test_id


def test_trace_id_in_response_headers(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert "X-Trace-ID" in res.headers
    assert "X-Response-Time-Ms" in res.headers


def test_custom_trace_id_honored(client):
    custom_trace = "custom-client-trace-999"
    res = client.get("/api/v1/health", headers={"X-Trace-ID": custom_trace})
    assert res.status_code == 200
    assert res.headers["X-Trace-ID"] == custom_trace


def test_global_exception_handler(client):
    # Requesting non-existent KB triggers KnowledgeBaseNotFoundError (404)
    res = client.get("/api/v1/knowledge-bases/non_existent_kb_id")
    assert res.status_code == 404
    data = res.json()
    assert "error" in data
    assert "trace_id" in data
    assert data["status_code"] == 404
