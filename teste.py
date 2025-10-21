import requests 
EVOLUTION_API_KEY = "E9D4279FF6D9-4EB4-8CCE-93374B6D5FB5"

url = "https://saraevo-evolution-api.jntduz.easypanel.host/message/sendText/Papagaio_dev"
payload = {"number": "554196137682", "text": "Mensagem de teste"}
headers = {"apikey": EVOLUTION_API_KEY, "Content-Type": "application/json"}
response = requests.post(url, json=payload, headers=headers)
print(response.status_code, response.text)