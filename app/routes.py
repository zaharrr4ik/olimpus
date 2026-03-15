from flask import Blueprint, render_template, redirect, url_for, request, flash, session, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from app.models import db, User
from functools import wraps
import numpy as np
import json
import os

auth = Blueprint('auth', __name__)
admin = Blueprint('admin', __name__, url_prefix='/admin')
user = Blueprint('user', __name__, url_prefix='/user')

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated

def user_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'user':
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated

@auth.route('/', methods=['GET', 'POST'])
@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        u = User.query.filter_by(username=username).first()
        if u and u.check_password(password):
            login_user(u)
            if u.role == 'admin':
                return redirect(url_for('admin.dashboard'))
            return redirect(url_for('user.profile'))
        flash('Неверный логин или пароль')
    return render_template('login.html')

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))

@admin.route('/dashboard')
@admin_required
def dashboard():
    users = User.query.filter_by(role='user').all()
    return render_template('admin.html', users=users)

@admin.route('/create_user', methods=['GET', 'POST'])
@admin_required
def create_user():
    if request.method == 'POST':
        u = User(
            username=request.form.get('username'),
            role='user',
            first_name=request.form.get('first_name'),
            last_name=request.form.get('last_name')
        )
        u.set_password(request.form.get('password'))
        db.session.add(u)
        db.session.commit()
        return redirect(url_for('admin.dashboard'))
    return render_template('admin/create_user.html')

@user.route('/profile')
@user_required
def profile():
    return render_template('profile.html', user=current_user)

@user.route('/upload', methods=['GET', 'POST'])
@user_required
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        if not f:
            flash('Файл не выбран')
            return redirect(url_for('user.upload'))

        import tempfile
        from app.predict import predict
        from app.preprocess import load_test_npz

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name

        try:
            test_x, test_y = load_test_npz(tmp_path)
            result = predict(test_x, test_y)
            session['result'] = result
        finally:
            os.unlink(tmp_path)

        return redirect(url_for('user.analytics'))

    return render_template('upload.html')

@user.route('/analytics')
@user_required
def analytics():
    from config import HISTORY_PATH, DATA_DIR

    history = {}
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            history = json.load(f)

    train_y = np.load(os.path.join(DATA_DIR, 'train_y.npy'), allow_pickle=True)
    mapping_path = os.path.join(DATA_DIR, 'class_mapping.json')
    with open(mapping_path) as f:
        mapping = json.load(f)

    train_names = [s[32:] for s in train_y]
    train_counts = {name: train_names.count(name) for name in mapping}

    valid_y = np.load(os.path.join(DATA_DIR, 'valid_y.npy'), allow_pickle=True)
    valid_names = [s[32:] for s in valid_y]
    from collections import Counter
    top5 = Counter(valid_names).most_common(5)

    result = session.get('result', {})

    return render_template('analytics.html',
        history=history,
        train_counts=train_counts,
        top5=top5,
        result=result
    )
