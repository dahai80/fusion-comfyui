import pytest
from fastapi.testclient import TestClient

from fusion_comfyui.server.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# New Vue frontend hits /api/* prefixed routes. Each must return JSON (not 404),
# else the frontend aborts init and the toolbar stays display:none.
class TestApiRouteAliases:
    def test_api_system_stats(self, client):
        r = client.get("/api/system_stats")
        assert r.status_code == 200
        body = r.json()
        assert "system" in body
        assert "devices" in body

    def test_api_object_info(self, client):
        r = client.get("/api/object_info")
        assert r.status_code == 200
        assert isinstance(r.json(), dict)

    def test_api_settings_get(self, client):
        r = client.get("/api/settings")
        assert r.status_code == 200
        assert isinstance(r.json(), dict)

    def test_api_extensions(self, client):
        r = client.get("/api/extensions")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_api_users(self, client):
        r = client.get("/api/users")
        assert r.status_code == 200
        assert isinstance(r.json(), dict)

    def test_api_userdata_workflows(self, client):
        r = client.get("/api/userdata", params={"dir": "workflows"})
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_api_i18n(self, client):
        r = client.get("/api/i18n")
        assert r.status_code == 200
        assert isinstance(r.json(), dict)

    def test_api_global_subgraphs(self, client):
        r = client.get("/api/global_subgraphs")
        assert r.status_code == 200

    def test_api_settings_post_key(self, client):
        r = client.post("/api/settings/Comfy.InstalledVersion", json="0.2.8")
        assert r.status_code == 200

    def test_api_queue(self, client):
        r = client.get("/api/queue")
        assert r.status_code == 200

    def test_api_interrupt(self, client):
        r = client.post("/api/interrupt")
        assert r.status_code == 200
        # /api/interrupt 置 _interrupt_flag=True；消费残留 flag 防污染后续
        # DAG executor（check_interrupt 读到残留 → status=interrupted）。
        from fusion_comfyui.server.protocol import check_interrupt

        assert check_interrupt() is True

    def test_api_embeddings(self, client):
        assert client.get("/api/embeddings").status_code == 200

    def test_api_experiment_models(self, client):
        assert client.get("/api/experiment/models").status_code == 200

    def test_api_workflow_templates(self, client):
        assert client.get("/api/workflow_templates").status_code == 200

    def test_api_view_metadata(self, client):
        assert client.get("/api/view_metadata/foo.png").status_code == 200

    def test_api_userdata_file_binary_404(self, client):
        assert client.get("/api/userdata/missing.png").status_code == 404

    def test_api_free(self, client):
        assert client.post("/api/free").status_code == 200

    def test_api_global_subgraphs_single(self, client):
        assert client.get("/api/global_subgraphs/abc").status_code == 404

    def test_api_userdata_template_json(self, client):
        r = client.get("/api/userdata/comfy.templates.json")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_core_templates_index_json(self, client):
        r = client.get("/templates/index.json")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_api_jobs_pagination_snake_case(self, client):
        r = client.get("/api/jobs")
        body = r.json()
        assert "has_more" in body["pagination"]


# Legacy flat routes must remain for ComfyUI-protocol clients.
class TestLegacyRoutesPreserved:
    def test_legacy_system_stats(self, client):
        assert client.get("/system_stats").status_code == 200

    def test_legacy_object_info(self, client):
        assert client.get("/object_info").status_code == 200

    def test_legacy_extensions(self, client):
        assert client.get("/extensions").status_code == 200

    def test_legacy_settings(self, client):
        assert client.get("/settings").status_code == 200
