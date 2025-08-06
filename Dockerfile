# Use a imagem base do Python 3.10
FROM python:3.10-slim

# Defina o diretório de trabalho dentro do container
WORKDIR /app

# Copie o arquivo requirements.txt para o container
COPY requirements.txt .

# Instale as dependências do projeto
RUN pip install --no-cache-dir -r requirements.txt

# Copie todos os arquivos do projeto para o container
COPY . .

# Exponha a porta (se necessário, ajuste conforme sua aplicação)
EXPOSE 8000

# Comando para executar a aplicação
CMD ["python", "main.py"]