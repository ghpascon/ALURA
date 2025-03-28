from flask import render_template, url_for, request, redirect, flash, session
from main import app
from models.models import analizador
from models.login import login_required
from forms.forms import FormularioAnalise

@app.route('/analise_sentimento')
@login_required
def analise_sentimento():
    return render_template(
        'analise_sentimento.html',
        titulo="Analisador de sentimento",
        form = FormularioAnalise()
    )

@app.route('/sentimento',  methods=['POST'])
@login_required
def sentimento():
    form = FormularioAnalise(request.form)
    if not form.validate_on_submit():
        flash('Erro no formulario')
        return redirect(url_for('analise_sentimento'))
    frase = request.form['frase']
    sentimento = analizador.analise(frase)
    return redirect(url_for('resultado', result=f'Polaridade da frase é: {sentimento}', page='analise_sentimento'))
