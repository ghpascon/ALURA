from flask import render_template, url_for, request, redirect, flash, session
from main import app
from models.models import previsao_casa
from models.login import login_required
from forms.forms import FormularioCasa

@app.route('/preco_casa')
@login_required
def preco_casa():
    return render_template(
        'preco_casa.html',
        titulo="Previsão de preço de casa",
        form = FormularioCasa()
    )

@app.route('/prever_casa',  methods=['POST'])
@login_required
def prever_casa():
    form = FormularioCasa(request.form)
    if not form.validate_on_submit():
        flash('Erro no formulario')
        return redirect(url_for('preco_casa'))
    tamanho = request.form['tamanho']
    ano = request.form['ano']
    garagem = request.form['garagem']
    preco = previsao_casa.prever(tamanho,ano,garagem)
    return redirect(url_for('resultado',result=f"Preço previsto: {preco}", page="preco_casa"))
print('casa')