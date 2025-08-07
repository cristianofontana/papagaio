numero_completo = "5511915629331@s.whatsapp.net"
sufixo = "@s.whatsapp.net"

if numero_completo.endswith(sufixo):
    numero_limpo = numero_completo[:-len("@s.whatsapp.net")]
else:
    numero_limpo = numero_completo  # Fallback se não tiver o sufixo

print(numero_limpo)  # Saída: 5511915629331