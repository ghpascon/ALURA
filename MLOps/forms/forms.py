from flask_wtf import FlaskForm
from wtforms import validators, StringField, PasswordField, IntegerField, SubmitField

class FormularioLogin(FlaskForm):
    nome = StringField('Nome de ususário', [validators.data_required(), validators.Length(min=1, max=200)])
    senha = PasswordField('Senha', [validators.data_required(), validators.Length(min=1, max=200)])
    submit = SubmitField('Login')

class FormularioAnalise(FlaskForm):
    frase = StringField('Frase', [validators.data_required(), validators.Length(min=1, max=200)])
    submit = SubmitField('Enviar frase para analise')

class FormularioCasa(FlaskForm):
    tamanho = IntegerField('Tamanho da casa (m²)', [validators.data_required(), validators.number_range(1,5000)])
    ano = IntegerField('Ano da casa', [validators.data_required(), validators.number_range(1800,2025)])
    garagem = IntegerField('Carros na garagem', [validators.data_required(), validators.number_range(0,100)])
    submit = SubmitField('Calcular preço de casa')
