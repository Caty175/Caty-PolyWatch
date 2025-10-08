from flask import render_template
from app import app

@app.route('/')
def landing():
    return render_template('Landing.html')

@app.route('/login')
def login():
    return render_template('Login.html')

# Signup GET/POST handled by server.signUp blueprint

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/reports')
def reports():
    return render_template('reports.html')

@app.route('/sandbox')
def sandbox():
    return render_template('sandbox_view.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')
