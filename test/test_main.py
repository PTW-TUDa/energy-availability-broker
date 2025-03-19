import pytest
from fastapi.testclient import TestClient

from energy_information_service.main import app, get_data_provider
from energy_information_service.services import DataProvider


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {}


@pytest.fixture()
def mock_data_provider(mocker):
    mock = mocker.AsyncMock(spec=DataProvider)

    # Ensure get_data is async and returns another async object
    class MockDataFrame:
        def to_dict(self, orient):
            return [{"price": 100}]  # Make it return a normal dictionary

        def to_csv(self, index):
            return "price\n100"  # Make it return a normal string

    async def mock_get_data():
        return MockDataFrame()

    mock.get_data.side_effect = mock_get_data
    return mock


@pytest.fixture()
def client(mock_data_provider):
    app.dependency_overrides[get_data_provider] = lambda: mock_data_provider
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_get_price_matrix(client):
    response = client.get("/data")
    assert response.status_code == 200
    assert response.json() == [{"price": 100}]


def test_get_data_csv(client):
    response = client.get("/csv")
    assert response.status_code == 200
    assert response.text == "price\n100"
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
