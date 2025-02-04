import os
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

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

system_prompt = "Você responde apenas perguntas relacionadas a um e-comerce sustentavél"

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-exp",
  generation_config=generation_config,
  system_instruction=system_prompt,
)

chat_session = model.start_chat(
  history=[
  ]
)

response = chat_session.send_message("Liste 3 produtos de limpeza")

print('\n\n------------------------------------------------------------------------------------------------')
print(response.text)
print('------------------------------------------------------------------------------------------------\n\n')
