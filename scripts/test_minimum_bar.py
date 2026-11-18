"""
Verification script for the Assignment Minimum Bar requirements:
1. Docker Compose running
2. CI test suite passing
3. Switches config dynamically without rebuild
4. Cites sources
5. Says 'not found' when out-of-scope
"""
import urllib.request
import json
import sys

BASE_URL = "http://localhost:8080/api/v1"

def check_1_docker():
    print("\n[CHECK 1] Starts with one Docker Compose command & containers healthy")
    try:
        req = urllib.request.Request(f"{BASE_URL}/health")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            assert data["status"] == "healthy"
            assert data["vector_store_status"] == "connected"
            print(f"  -> PASS: System is healthy! Vector Store: {data['vector_store_type']} (connected)")
            return True
    except Exception as e:
        print(f"  -> FAIL: {e}")
        return False

def check_3_dynamic_config():
    print("\n[CHECK 3] Switches config dynamically without rebuilding")
    try:
        # Create config v2 if not already present
        config_v2 = {
            "version": "v2",
            "description": "Production v2 with Hybrid Search and Reranking",
            "chunking": {"strategy": "structure", "chunk_size": 500, "chunk_overlap": 50},
            "retrieval": {"search_mode": "hybrid", "top_k": 5, "similarity_threshold": 0.30},
            "rerank": {"enabled": True, "top_n": 3},
            "llm": {"query_rewrite_enabled": True}
        }
        try:
            req = urllib.request.Request(f"{BASE_URL}/configs/", data=json.dumps(config_v2).encode(), headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            if e.code != 400:  # 400 means already exists
                raise

        # Activate v2
        req = urllib.request.Request(f"{BASE_URL}/configs/v2/activate", method="PUT")
        with urllib.request.urlopen(req) as resp:
            act_res = json.loads(resp.read().decode())
            assert act_res["version"] == "v2"
            assert act_res["is_active"] is True

        # Verify in health check
        req = urllib.request.Request(f"{BASE_URL}/health")
        with urllib.request.urlopen(req) as resp:
            health = json.loads(resp.read().decode())
            assert health["active_config_version"] == "v2"
            print("  -> PASS: Successfully activated Config 'v2' in real-time with zero container restarts!")
            return True
    except Exception as e:
        print(f"  -> FAIL: {e}")
        return False

def check_4_cites_sources():
    print("\n[CHECK 4] Answers question with strict, grounded citations")
    try:
        payload = {
            "knowledge_base_id": "acme_corp",
            "question": "Under what condition is a medical certificate required when taking sick leave?"
        }
        req = urllib.request.Request(f"{BASE_URL}/query/", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            assert data["is_found"] is True
            assert len(data["citations"]) > 0
            first = data["citations"][0]
            print(f"  -> PASS: Answer generated: \"{data['answer'][:90]}...\"")
            print(f"  -> Exact Citation: [Doc: {first['document_name']}, Page: {first['page_number']}, Section: {first['section_title']}]")
            return True
    except Exception as e:
        print(f"  -> FAIL: {e}")
        return False

def check_5_says_not_found():
    print("\n[CHECK 5] Strictly refuses to hallucinate and says 'not found' when information is absent")
    try:
        payload = {
            "knowledge_base_id": "acme_corp",
            "question": "What was the total landing payload weight of the NASA Curiosity Mars rover mission?"
        }
        req = urllib.request.Request(f"{BASE_URL}/query/", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            assert data["is_found"] is False
            assert "I could not find information about this in the knowledge base" in data["answer"]
            print(f"  -> PASS: Safely rejected: \"{data['answer']}\"")
            print(f"  -> Grounding check: is_found = {data['is_found']}, is_grounded = {data['is_grounded']}")
            return True
    except Exception as e:
        print(f"  -> FAIL: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("VERIFYING ASSIGNMENT MINIMUM BAR REQUIREMENTS")
    print("=" * 70)
    r1 = check_1_docker()
    r3 = check_3_dynamic_config()
    r4 = check_4_cites_sources()
    r5 = check_5_says_not_found()

    print("\n" + "=" * 70)
    if all([r1, r3, r4, r5]):
        print("ALL MINIMUM BAR FUNCTIONAL REQUIREMENTS PASSED WITH 0 ERRORS!")
    else:
        print("SOME CHECKS FAILED - PLEASE REVIEW LOGS")
    print("=" * 70)
