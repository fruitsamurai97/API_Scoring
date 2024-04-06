# test_routes.py
import pytest
from app import app  # Assurez-vous que ceci importe correctement votre instance d'application Flask

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_get_client(client):
    """Test pour vérifier que l'endpoint /client renvoie une liste d'ID."""
    response = client.get('/client')
    assert response.status_code == 200
    assert isinstance(response.json, list)  # S'assurer que la réponse est une liste

def test_predict(client):
    """Test pour vérifier que l'endpoint /predict fonctionne correctement."""
    # Ce test nécessite un ID valide depuis votre ensemble de données pour fonctionner
    id_valide = 100001  # Assurez-vous que c'est un ID valide présent dans votre df
    response = client.get(f'/predict?id={id_valide}')
    assert response.status_code == 200
    # Test pour vérifier la structure de la réponse
    data = response.json
    assert isinstance(data, dict)  # Vérifie que la réponse est un dictionnaire
    assert "Classe 1" in data  # Vérifie la présence d'une clé attendue dans la réponse
    # Vous pouvez ajouter plus de validations ici en fonction de la structure attendue de vos données

def test_predict_id_invalide(client):
    """Test pour vérifier le comportement avec un ID invalide."""
    id_invalide = -1  # Un ID qui ne devrait pas exister dans votre df
    response = client.get(f'/predict?id={id_invalide}')
    assert response.status_code == 404
    # Vous pouvez tester le message d'erreur spécifique ici si vous voulez
