document.getElementById('enviar').addEventListener('click', async function() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (!file) {
        document.getElementById('result').textContent = "Por favor, selecione uma imagem.";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://127.0.0.1:8000/classify/", {
            method: "POST",
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            document.getElementById('result').textContent = `O objeto foi classificado como: ${data.class}`;
        } else {
            document.getElementById('result').textContent = "Erro ao identificar o objeto. Tente novamente.";
        }
    } catch (error) {
        console.error("Erro:", error);
        document.getElementById('result').textContent = "Erro na comunicação com o servidor.";
    }
});
