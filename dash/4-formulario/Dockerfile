FROM python:3.10

ENV APP_HOME /app
WORKDIR $APP_HOME

# Copia o arquivo requirements.txt para o diretório de trabalho no container
COPY requirements.txt $APP_HOME/

# Instala as dependências do Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copia todos os arquivos para o diretório de trabalho no container
COPY . $APP_HOME/

# Expõe a porta 8080
EXPOSE 8080

# Define o comando padrão a ser executado no container
CMD ["python", "create_dashboard.py"]
