from flask import Flask,render_template, request, Response
import google.generativeai as genai
from dotenv import load_dotenv
import os
from time import sleep
from pathlib import Path
import uuid
from werkzeug.utils import secure_filename
import shutil

caminho_imagem = None
# Define o caminho do arquivo .env
dotenv_path = Path(r"C:\Users\Usuario\Desktop\GEMINI\.env")  # Substitua pelo caminho real

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv(dotenv_path=dotenv_path)

# Configura a chave de API do Gemini usando a variável de ambiente
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Create the model
generation_config = {
  "temperature": 1.0,
  "top_p": 0.9,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

def get_data():
    with open(r"gemini\flask_chatbot\dados\dados.txt", encoding="utf-8") as file:
        return file.read()
    
personas = {
    'positivo': """
    Assuma que você é o Entusiasta Musical, um atendente virtual da MusiMart, cujo amor pela música é contagiante. 
    Sua energia é sempre alta, seu tom é extremamente positivo, e você adora usar emojis para transmitir emoção 🎶🎸. 
    Você vibra com cada decisão que os clientes tomam para aprimorar sua jornada musical, seja comprando um novo instrumento ou escolhendo acessórios 🎧. 
    Seu objetivo é fazer os clientes se sentirem empolgados e inspirados a continuar explorando o mundo da música.
    Além de fornecer informações, você elogia os clientes por suas escolhas musicais e os encoraja a seguir crescendo como músicos. 
    """,
    'neutro': """
    Assuma que você é o Informante Técnico, um atendente virtual da MusiMart que valoriza a precisão, a clareza e a eficiência em todas as interações. 
    Sua abordagem é formal e objetiva, sem o uso de emojis ou linguagem casual. 
    Você é o especialista que os músicos e clientes procuram quando precisam de informações detalhadas sobre instrumentos, equipamentos de som ou técnicas musicais. 
    Seu principal objetivo é fornecer dados precisos para que os clientes possam tomar decisões informadas sobre suas compras. 
    Embora seu tom seja sério, você ainda demonstra um profundo respeito pela arte da música e pelo compromisso dos clientes em aprimorar suas habilidades.
    """,
    'negativo': """
    Assuma que você é o Suporte Acolhedor, um atendente virtual da MusiMart, conhecido por sua empatia, paciência e capacidade de entender as preocupações dos músicos. 
    Você usa uma linguagem calorosa e encorajadora e expressa apoio emocional, especialmente para músicos que estão enfrentando desafios, como a escolha de um novo instrumento ou problemas técnicos com seus equipamentos. Sem uso de emojis. 
    Você está aqui não apenas para resolver problemas, mas também para escutar, oferecer conselhos e validar os esforços dos clientes em sua jornada musical. 
    Seu objetivo é construir relacionamentos duradouros, garantir que os clientes se sintam compreendidos e apoiados, e ajudá-los a superar os desafios com confiança.
    """
}

system_prompt = f"""
#pesona
Você responde apenas perguntas relacionadas ao ecomerce e assume um papel diferente de acordo com o sentimento do cliente

#personas
{personas}

{get_data()}

#imagens
caso existir imagens, utilize as caracteristicas dela na resposta, fale das caracteristicas, como cores
"""

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-exp",
  generation_config=generation_config,
  system_instruction=system_prompt,
)

chat_session = model.start_chat(
  history=[
  ]
)

app = Flask(__name__)
app.secret_key = 'alura'

def bot(prompt):
    global caminho_imagem
    maximo_tentativas = 1
    repeticao = 0

    while True:
        try:
            if caminho_imagem is None:
                resposta = chat_session.send_message(prompt)
                return resposta.text
            resposta = chat_session.send_message([genai.upload_file(caminho_imagem), prompt])
            caminho_imagem = None
            return resposta.text
        except Exception as erro:
            repeticao += 1
            if repeticao >= maximo_tentativas:
                return "Erro no Gemini: %s" % erro
            
            sleep(50)


@app.route("/chat", methods=["POST"])
def chat():
    prompt = request.json["msg"]
    resposta = bot(prompt)
    clear_temp_img()
    return resposta

@app.route("/")
def home():
    return render_template("index.html")


UPLOAD_FOLDER = "gemini/flask_chatbot/imagens_temporarias" # Set directory for temporary files
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"} # Set allowed file extensions

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/upload_imagem", methods=["POST"])
def upload_imagem():
    global caminho_imagem
    if "imagem" not in request.files:
        return "Nenhum arquivo enviado", 400

    imagem_enviada = request.files["imagem"]
    if imagem_enviada.filename == "":
        return "Nenhum arquivo selecionado", 400

    if not allowed_file(imagem_enviada.filename):
         return "Extensão do arquivo não permitida", 400

    # Generate a secure filename and create directory if it doesn't exist
    filename = secure_filename(imagem_enviada.filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True) # Creates directory
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        imagem_enviada.save(file_path) # Correct way to save the file
        caminho_imagem = file_path
        return "Sucesso", 200
    except Exception as e:
        return f"Erro ao processar o arquivo: {e}", 500


def clear_temp_img():
    try:
        if os.path.exists(UPLOAD_FOLDER):
           shutil.rmtree(UPLOAD_FOLDER)  # Delete the entire directory
           os.makedirs(UPLOAD_FOLDER) # Recreate the empty directory
           print(f"Successfully cleared all files in {UPLOAD_FOLDER}")
        else:
            print(f"Directory {UPLOAD_FOLDER} doesn't exist.")
    except Exception as e:
        print(f"Error clearing directory {UPLOAD_FOLDER}: {e}")


if __name__ == "__main__":
    app.run(debug = True)