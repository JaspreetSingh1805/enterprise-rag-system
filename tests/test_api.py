import io
import time


def test_api_health_check(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "active_config_version" in data


def test_api_knowledge_bases_crud(client):
    # 1. Create KB
    kb_payload = {
        "id": "hr_docs",
        "name": "HR Department Documents",
        "description": "Internal leave policies and guidelines",
    }
    create_res = client.post("/api/v1/knowledge-bases/", json=kb_payload)
    assert create_res.status_code == 201
    created_kb = create_res.json()["knowledge_base"]
    assert created_kb["id"] == "hr_docs"

    # 2. List KBs
    list_res = client.get("/api/v1/knowledge-bases/")
    assert list_res.status_code == 200
    assert any(k["id"] == "hr_docs" for k in list_res.json()["knowledge_bases"])

    # 3. Get KB
    get_res = client.get("/api/v1/knowledge-bases/hr_docs")
    assert get_res.status_code == 200
    assert get_res.json()["knowledge_base"]["name"] == "HR Department Documents"

    # 4. Update KB
    update_res = client.put("/api/v1/knowledge-bases/hr_docs", json={"name": "HR Global Docs"})
    assert update_res.status_code == 200
    assert update_res.json()["knowledge_base"]["name"] == "HR Global Docs"

    # 5. Delete KB
    del_res = client.delete("/api/v1/knowledge-bases/hr_docs")
    assert del_res.status_code == 204

    # 6. Verify deleted
    assert client.get("/api/v1/knowledge-bases/hr_docs").status_code == 404


def test_api_configs_management(client):
    # 1. List configs (default v1 should be present)
    list_res = client.get("/api/v1/configs/")
    assert list_res.status_code == 200
    assert list_res.json()["active_version"] == "v1"

    # 2. Create new config v2
    new_cfg = {
        "version": "v2",
        "description": "Config with custom chunking and hybrid search",
        "chunking": {"strategy": "structure", "chunk_size": 400, "chunk_overlap": 40},
        "retrieval": {"search_mode": "hybrid", "top_k": 7},
    }
    create_res = client.post("/api/v1/configs/", json=new_cfg)
    assert create_res.status_code == 201

    # 3. Activate v2 dynamically
    act_res = client.put("/api/v1/configs/v2/activate")
    assert act_res.status_code == 200
    assert act_res.json()["is_active"] is True

    # 4. Verify v2 is now active
    active_check = client.get("/api/v1/configs/")
    assert active_check.json()["active_version"] == "v2"


def test_api_document_upload_and_query_flow(client):
    # 1. Create KB
    client.post(
        "/api/v1/knowledge-bases/",
        json={"id": "company_faq", "name": "Company FAQ", "description": "FAQ KB"},
    )

    # 2. Upload Document
    file_bytes = b"# Working Hours Policy\nStandard office working hours are 9:00 AM to 5:00 PM."
    upload_res = client.post(
        "/api/v1/documents/upload",
        data={"knowledge_base_id": "company_faq", "custom_fields": '{"dept": "Operations"}'},
        files={"file": ("working_hours.md", io.BytesIO(file_bytes), "text/markdown")},
    )
    assert upload_res.status_code == 202
    job_id = upload_res.json()["job"]["id"]

    # 3. Check Job Status
    job_res = client.get(f"/api/v1/jobs/{job_id}")
    assert job_res.status_code == 200
    assert job_res.json()["job"]["id"] == job_id

    # Give async task a brief moment to complete in local test environment
    time.sleep(0.1)

    # 4. List Documents
    docs_res = client.get("/api/v1/documents/?knowledge_base_id=company_faq")
    assert docs_res.status_code == 200
    docs = docs_res.json()["documents"]
    assert len(docs) >= 1

    # 5. Execute RAG Query
    query_payload = {
        "knowledge_base_id": "company_faq",
        "question": "What are the standard office working hours?",
    }
    query_res = client.post("/api/v1/query/", json=query_payload)
    assert query_res.status_code == 200
    q_data = query_res.json()
    assert "answer" in q_data
    assert "trace" in q_data
    assert q_data["trace"]["total_latency_ms"] >= 0
    assert "config_version" in q_data
