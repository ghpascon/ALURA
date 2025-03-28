from flask import render_template, request, redirect, session, flash
from main import app
from helper.models import *
from helper.forms import *
from flask_bcrypt import check_password_hash

@app.route('/login')
def login():
    page = request.args.get('page')
    return render_template('login.html', titulo='Faça seu Login', page=page, form=FormularioUsuario())

@app.route('/autenticar', methods=['POST'])
def autenticar():
    page = request.form['page']
    form = FormularioUsuario(request.form)
    if not form.validate_on_submit():
        flash('Erro ao fazer login')
        return redirect(f"/login?page={page}")  
    usuario = Usuarios.query.filter_by(nome=request.form['usuario']).first()
    print(usuario)
    if usuario:
        if check_password_hash(usuario.senha, request.form['senha']):
            session['usuario'] = usuario.nome
            flash(f"Usuário: {session['usuario']} logado com sucesso")
            if page is None or page =='None':
                page = ""
            return redirect('/'+page)
        flash('Senha Incorreta')
        return redirect(f"/login?page={page}")  
    flash('Usuario não cadastrado')
    return redirect(f"/login?page={page}")  

@app.route('/logout')
def logout():
    if not 'usuario' in session or session['usuario'] is None:
        return redirect('/login')
    session['usuario'] = None
    flash('Logout com Sucesso')
    return redirect('/login')