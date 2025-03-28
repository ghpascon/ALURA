from flask_wtf import FlaskForm
from wtforms import validators, StringField, PasswordField, SubmitField

class FormularioJogo(FlaskForm):
    nome = StringField('Nome', [validators.data_required(), validators.Length(min=1, max=50)])
    categoria = StringField('Categoria', [validators.data_required(), validators.Length(min=1, max=40)])
    console = StringField('Console', [validators.data_required(), validators.Length(min=1, max=20)])
    submit = SubmitField('Salvar')

class FormularioUsuario(FlaskForm):
    usuario = StringField('Nome de usuario', [validators.data_required(), validators.Length(min=1, max=50)])
    senha = PasswordField('Senha', [validators.data_required(), validators.Length(min=1, max=100)])
    submit = SubmitField('Salvar')