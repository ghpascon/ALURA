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

lista_categorias_possiveis = "Eletrônicos Verdes,Moda Sustentável,Produtos de Limpeza Ecológicos,Alimentos Orgânicos"
system_prompt = f"""
Você deve classificar os produtos que o usuário te enviar com base em {lista_categorias_possiveis}

#formato de saida
Produto: nome do produto
categoria: nome do categoria

#exemplo de saida
Produto: Escova de dentes de bambu
Categoria: Produtos de Limpeza Ecológicos
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


while True:
  pergunta = input('Digite o produto para classificar (99 para sair): ')
  if pergunta == '99':
    break
  
  response = chat_session.send_message(pergunta)

  print('\n\n------------------------------------------------------------------------------------------------')
  print(response.text)
  print('------------------------------------------------------------------------------------------------\n\n')
