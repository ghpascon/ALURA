from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import json
from flask_wtf.csrf import CSRFProtect
import os
import importlib

def import_views():
    '''
    importa todas as rotas da pasta views
    '''
    views_path = 'views'
    for filename in os.listdir(views_path):
        if filename.endswith('.py'):
            module_name = filename[:-3]
            module = importlib.import_module(f'.{module_name}', package='views')
            globals().update({name: getattr(module, name) for name in dir(module) if not name.startswith('__')})


with open("config/config.json", 'r') as f:
    cfg_data = json.load(f) 

app = Flask(__name__)
app.secret_key = cfg_data['secret']
app.config['SQLALCHEMY_DATABASE_URI'] = f"{cfg_data['SGBD']}://{cfg_data['usuario']}:{cfg_data['senha']}@{cfg_data['host']}/{cfg_data['db_name']}"

db = SQLAlchemy(app)
csrf = CSRFProtect(app)

import_views()
if __name__=="__main__":
    app.run(debug=True)


