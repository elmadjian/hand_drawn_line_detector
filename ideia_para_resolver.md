# primeira ideia
Poderíamos usar a IFT para construir todos os caminhos válidos do grafo e depois dessa etapa passear com uma janela sobre os caminhos de modo a escolher aquele que apresenta a menor curvatura e maior continuidade.

# segunda ideia
Varrer toda a imagem em busca de junções (são os pontos críticos, afinal). A partir das junções, construir um mapa de caminhos que caiba dentro de uma janela, em que o critério de continuidade é dada pela curvatura formada pelo caminho que entra e sai do ponto de junção.

Depois desse estágio, basta armazenar esses caminhos já pré-processados, rodar a IFT sobre o restante das sementes e ligar os caminhos encontrados com aqueles pré-processados, sempre que for possível.
