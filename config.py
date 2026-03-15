import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

SECRET_KEY = 'olimpus-secret-key'
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR, 'db.sqlite3')
SQLALCHEMY_TRACK_MODIFICATIONS = False
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.h5')
HISTORY_PATH = os.path.join(BASE_DIR, 'model', 'training_history.json')
DATA_DIR = os.path.join(BASE_DIR, 'Data')
UPLOAD_FOLDER = os.path.join(BASE_DIR, '__temp')
