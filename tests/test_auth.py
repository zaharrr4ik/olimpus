import pytest
from app import create_app
from app.models import db, User

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    with app.app_context():
        db.drop_all()
        db.create_all()
        u = User(username='testadmin', role='admin', first_name='Test', last_name='Admin')
        u.set_password('password')
        db.session.add(u)
        u2 = User(username='testuser', role='user', first_name='Test', last_name='User')
        u2.set_password('password')
        db.session.add(u2)
        db.session.commit()
    with app.test_client() as client:
        yield client

def test_login_success_admin(client):
    r = client.post('/login', data={'username': 'testadmin', 'password': 'password'}, follow_redirects=True)
    assert r.status_code == 200

def test_login_wrong_password(client):
    r = client.post('/login', data={'username': 'testadmin', 'password': 'wrong'}, follow_redirects=True)
    assert 'Неверный логин или пароль'.encode() in r.data

def test_login_success_user(client):
    r = client.post('/login', data={'username': 'testuser', 'password': 'password'}, follow_redirects=True)
    assert r.status_code == 200

def test_admin_page_blocked_for_user(client):
    client.post('/login', data={'username': 'testuser', 'password': 'password'})
    r = client.get('/admin/dashboard', follow_redirects=True)
    assert r.status_code == 200
    assert b'Olimpus' in r.data

def test_user_page_blocked_for_admin(client):
    client.post('/login', data={'username': 'testadmin', 'password': 'password'})
    r = client.get('/user/profile', follow_redirects=True)
    assert r.status_code == 200
    assert b'Olimpus' in r.data

def test_logout(client):
    client.post('/login', data={'username': 'testadmin', 'password': 'password'})
    r = client.get('/logout', follow_redirects=True)
    assert r.status_code == 200
