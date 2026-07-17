import os
import shutil
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.state import state, STATE_FILE


@pytest.fixture(autouse=True)
def clean_state():
    """Fixture to ensure state is clean before/after tests and state file is deleted."""
    # Reset in-memory AppState properties
    state.pending_requests.clear()
    state.update_count = 0
    
    # Remove any existing state snapshot file
    if os.path.exists(STATE_FILE):
        try:
            os.remove(STATE_FILE)
        except Exception:
            pass
            
    # Yield to run tests
    yield
    
    # Cleanup after tests
    if os.path.exists(STATE_FILE):
        try:
            os.remove(STATE_FILE)
        except Exception:
            pass


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "agent" in data
        assert data["agent"]["total_rounds"] == 0


def test_get_price_endpoint():
    with TestClient(app) as client:
        # Request a price with valid context parameters
        response = client.get(
            "/price?segment=student&day_type=weekday&competitor_price=15.0"
        )
        assert response.status_code == 200
        data = response.json()
        assert "price" in data
        assert "arm_id" in data
        assert "request_id" in data
        assert data["price"] in [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 65.0, 80.0]
        
        # Verify request is recorded in pending list
        request_id = data["request_id"]
        assert request_id in state.pending_requests
        arm_idx, x_agent, ctx = state.pending_requests[request_id]
        assert ctx.segment == "student"
        assert ctx.day_type == "weekday"
        assert ctx.competitor_price == 15.0


def test_get_price_invalid_params():
    with TestClient(app) as client:
        # Invalid segment
        response = client.get(
            "/price?segment=alien&day_type=weekday&competitor_price=15.0"
        )
        assert response.status_code == 400
        
        # Invalid day type
        response = client.get(
            "/price?segment=student&day_type=holiday&competitor_price=15.0"
        )
        assert response.status_code == 400
        
        # Negative competitor price
        response = client.get(
            "/price?segment=student&day_type=weekday&competitor_price=-5.0"
        )
        assert response.status_code == 400


def test_post_outcome_success():
    with TestClient(app) as client:
        # Get a price first
        response = client.get(
            "/price?segment=professional&day_type=weekend&competitor_price=25.0"
        )
        assert response.status_code == 200
        data = response.json()
        request_id = data["request_id"]
        arm_id = data["arm_id"]

        # Report outcome
        outcome_response = client.post(
            "/outcome",
            json={"request_id": request_id, "purchased": True}
        )
        assert outcome_response.status_code == 200
        outcome_data = outcome_response.json()
        assert outcome_data["status"] == "success"
        assert "updated" in outcome_data["message"]

        # Check in-memory state changes
        assert state.agent.total_rounds == 1
        assert state.agent.counts[arm_id] == 1
        assert request_id not in state.pending_requests


def test_post_outcome_404():
    with TestClient(app) as client:
        # Post outcome for unregistered request_id
        response = client.post(
            "/outcome",
            json={"request_id": "nonexistent-uuid", "purchased": True}
        )
        assert response.status_code == 404


def test_persistence_saving_and_loading():
    with TestClient(app) as client:
        # We need state.save_interval = 1 for testing persistence triggers
        state.save_interval = 1
        
        # Get a price
        response = client.get(
            "/price?segment=default&day_type=weekday&competitor_price=20.0"
        )
        request_id = response.json()["request_id"]
        arm_id = response.json()["arm_id"]

        # Post outcome - this should trigger a save_snapshot() because save_interval = 1
        outcome_response = client.post(
            "/outcome",
            json={"request_id": request_id, "purchased": True}
        )
        assert outcome_response.status_code == 200
        
        # Verify snapshot is written to disk
        assert os.path.exists(STATE_FILE)
        
        # Restart state container and assert it loads the saved counts
        state2 = state
        state2.agent = None
        state2.initialize()
        assert state2.agent.total_rounds == 1
        assert state2.agent.counts[arm_id] == 1
