from flask import render_template, url_for,redirect,flash,request, session
from main import app
from models.login import Usuarios
from forms.forms import *
from forms.forms import FormularioLogin
from werkzeug.security import check_password_hash

@app.route('/login')
def login():
    return render_template('login.html', titulo='Login', form=FormularioLogin(), next=request.args.get('next'))

@app.route('/autenticar', methods=['POST'])
def autenticar():
    next = request.form['next']
    if next is None or next == 'None':next=url_for('index')

    form = FormularioLogin(request.form)
    if not form.validate_on_submit():
        flash('Erro no formulario')
        return redirect(url_for('login', next=next))
    nome = request.form['nome']
    senha = request.form['senha']
    usuario = Usuarios.query.filter_by(nome=nome).first()
    if usuario:
        if check_password_hash(usuario.senha, senha):
            session['username']=nome
            return redirect(next)

    flash('Erro ao fazer login')    
    return redirect(url_for('login', next=next))
        
@app.route('/logout')
def logout():
    session['username']=None
    return redirect(url_for('login'))        