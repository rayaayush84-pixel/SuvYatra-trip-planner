import os
import tempfile

# Must be set before the app is imported
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ["AUTO_INGEST"] = "false"
os.environ["CHROMA_DIR"] = tempfile.mkdtemp()
os.environ["ENVIRONMENT"] = "development"

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c
