import os

def renomear_imagens_vidro(pasta, prefixo, numero_inicial):
    # Listar todos os arquivos na pasta
    arquivos = os.listdir(pasta)
    arquivos.sort()  # Ordenar para garantir a ordem desejada

    # Filtrar apenas arquivos com extensão .jpg
    arquivos_imagens = [arquivo for arquivo in arquivos if arquivo.endswith('.jpg')]

    contador = numero_inicial
    
    # Iterar sobre cada imagem e renomeá-la
    for arquivo in arquivos_imagens:
        nome_antigo = os.path.join(pasta, arquivo)
        nome_novo = os.path.join(pasta, f"{prefixo}_{contador}.jpg")
        
        os.rename(nome_antigo, nome_novo)
        contador += 1
    
    print(f"Renomeação concluída! Total de imagens renomeadas: {len(arquivos_imagens)}")

# Configurações da pasta e prefixo
pasta_vidro = 'C:\\Users\\Breno\\OneDrive\\Pictures\\Teste IA'  # Nome da pasta onde estão as imagens
prefixo = 'vidro'  # Prefixo a ser utilizado
numero_inicial = 1468  # Valor inicial do contador

# Executar o script de renomeação
renomear_imagens_vidro(pasta_vidro, prefixo, numero_inicial)
