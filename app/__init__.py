from flask import Flask
from flask_login import LoginManager
from app.models import db, User
import config

def create_app():
    app = Flask(__name__)
    app.config.from_object(config)

    db.init_app(app)

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    from app.routes import auth, admin, user
    app.register_blueprint(auth)
    app.register_blueprint(admin)
    app.register_blueprint(user)

    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            u = User(username='admin', role='admin', first_name='Admin', last_name='Admin')
            u.set_password('admin')
            db.session.add(u)
            db.session.commit()

    return app
