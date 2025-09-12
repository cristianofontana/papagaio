


import requests

url = "https://saraevo-evolution-api.jntduz.easypanel.host/message/sendMedia/Papagaio_dev"

payload = {
        "number": '554196137682',
        "mediatype": "document",
        "fileName": 'Imoveis_Eder_Maia.pdf',
        "caption": 'Imovevis Eder Maia, confira todos os detalhes',
        "media": 'https://xxwqlenrsuslzsrlcqhi.supabase.co/storage/v1/object/public/eder_maia/Imoveis_Eder_Maia.pdf'
    }
headers = {
    "apikey": "E9D4279FF6D9-4EB4-8CCE-93374B6D5FB5",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.json())