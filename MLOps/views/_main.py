from flask import render_template, request, session
from main import app
from models.login import login_required
from forms.forms import *
@app.route('/')
@login_required
def index():
    return render_template(
        'index.html',
        titulo="MLOps",
        form = FormularioAnalise(),
        username=session['username']
    )

@app.route('/resultado')
@login_required
def resultado():
    result = request.args.get('result')
    page = request.args.get('page')
    return render_template(
        'resultado.html',
        titulo="Resultado",
        resultado=result,
        page=page
    )
