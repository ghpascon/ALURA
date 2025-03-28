from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import json
from flask_wtf.csrf import CSRFProtect
from flask_bcrypt import Bcrypt

with open("config/config.json", 'r') as f:
    cfg_data = json.load(f) 

app = Flask(__name__)
app.secret_key = cfg_data['secret']
app.config['SQLALCHEMY_DATABASE_URI'] = f"{cfg_data['SGBD']}://{cfg_data['usuario']}:{cfg_data['senha']}@{cfg_data['host']}/{cfg_data['db_name']}"

db = SQLAlchemy(app)
csrf = CSRFProtect(app)
bcrypt = Bcrypt(app)

from helper.views_games import *
from helper.views_users import *
if __name__=="__main__":
    app.run(debug=True)