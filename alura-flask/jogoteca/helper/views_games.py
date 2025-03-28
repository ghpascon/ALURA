from flask import render_template, request, redirect, session, flash
from main import app,db
from helper.models import *
from helper.forms import *
import time
import os

@app.route('/')
def index():
    jogos = Jogos.query.all()
    return render_template(
        'lista.html',
        titulo="Jogos",
        jogos=jogos
    )

@app.route('/novo')
def novo():
    if not 'usuario' in session or session['usuario'] is None:
        flash('Necessário fazer o login para editar')
        return redirect('/login?page=novo')
    
    return render_template(
        'novo.html',
        titulo='Novo Jogo',
        time=time.time(),
        form=FormularioJogo()
    )

@app.route('/editar')
def editar():
    id = request.args.get('id')
    if not 'usuario' in session or session['usuario'] is None:
        flash('Necessário fazer o login para editar')
        return redirect(f'/login?page=editar?id={id}')

    jogo = Jogos.query.filter_by(id=id).first()
    if not jogo:
        return redirect('/')
    
    form=FormularioJogo()
    form.nome.data = jogo.nome
    form.categoria.data = jogo.categoria
    form.console.data = jogo.console

    return render_template(
        'editar.html',
        titulo='Editar Jogo',
        jogo=jogo,
        time=time.time(),
        form=form
    )

@app.route('/excluir')
def excluir():
    id = request.args.get('id')
    if not 'usuario' in session or session['usuario'] is None:
        flash('Necessário fazer o login para editar')
        return redirect(f'/login?page=excluir?id={id}')
    Jogos.query.filter_by(id=id).delete()
    db.session.commit()
    try:
        os.remove(f'static/images/{id}.jpg')
    except:pass

    flash(f'Jogo {id} excluido com sucesso')
    return redirect('/')


@app.route('/criar', methods=['POST'])
def criar():
    form = FormularioJogo(request.form)
    if not form.validate_on_submit():
        flash('Erro no formulario')
        return redirect('/novo')

    nome = request.form['nome']
    categoria = request.form['categoria']
    console = request.form['console']

    jogo = Jogos.query.filter_by(nome=nome).first()
    if jogo:
        flash('Jogo já existente')
        return redirect('/novo')
    
    novo_jogo = Jogos(nome=nome, categoria=categoria, console=console)
    try:
        db.session.add(novo_jogo)
        db.session.commit()
    except Exception as e:
        flash(f'Erro: {e}')
        return redirect("/novo")

    flash(f'{nome} adicionado com sucesso')

    arquivo = request.files['arquivo']
    if arquivo and arquivo.filename != '':
        arquivo.save(f'static/images/{novo_jogo.id}.jpg')

    return redirect('/')

@app.route('/atualizar', methods=['POST'])
def atualizar():
    form = FormularioJogo(request.form)
    if not form.validate_on_submit():
        flash('Erro no formulario')
        return redirect(f"/editar?id={request.form['id']}")    

    id = request.form['id']
    nome = request.form['nome']
    categoria = request.form['categoria']
    console = request.form['console']
    
    jogo = Jogos.query.filter_by(id=id).first()
    
    if jogo:
        jogo.nome = nome
        jogo.categoria = categoria
        jogo.console = console
        
        try:
            db.session.add(jogo)
            db.session.commit()
        except Exception as e:
            flash(f'Erro: {e}')
            return redirect(f'/editar?id={id}')
        
        flash(f'Jogo {id} alterado com sucesso')
        arquivo = request.files['arquivo']
        if arquivo and arquivo.filename != '':
            arquivo.save(f'static/images/{jogo.id}.jpg')
    return redirect('/')




