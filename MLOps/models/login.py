from main import db
from functools import wraps
from flask import session, redirect, url_for, request, flash

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not 'username' in session or session['username'] is None:
            flash('Necessário fazer o login para acessar essa pagina')
            return redirect(url_for('login', next=request.url)) 
        return f(*args, **kwargs) 
    return decorated_function

class Usuarios(db.Model):
    nome = db.Column(db.String(20), primary_key=True, nullable=False)    
    senha = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return '<Name %r>' % self.nome
