Estamos investigando novas possibilidades arquiteturais, esse código está sendo construído para rodar em pods com uma GPU A100, o código deve usar a compilação do pyTorch.

Ler todos arquivos dentro do diretório 'docs', em especial o arquivo LOWRANK.md

O ideal é construir modelos com diversas flags para ideias diferentes para fazer diferentes testes comparando as várias possibilidades. Sempre montar ao final uma tabela com diversas estatísticas interessantes como: Perplexidade, Loss, tempo, dimensão do espaço latente, dimensão interna dos blocos de atenção, número de parâmetros e outras que possam ser interessantes.

Nesse caso o objetivo NÃO é minimizar o tempo e/ou número de parametros! É realmente investigar diferentes arquiteturas, desde que elas não excedam no máximo 3x ou 4x o tempo e/ou parâmetros originais mas que possam ser usados para investigar novos caminhos. Por exemplo reduzir a dimensão interna dos blocos de atenção deveria tornar estes blocos mais interpretáveis, o que pode nos dar ideias no futuro de como utilizar essas especializações maiores de forma mais eficiente, principalmente para fazer a arquitetura aprender mais rápidos e com menos exemplos. Talvez com essas dimensões menores fique mais fácil ou intuitivo imaginar como introduzir formas que empurrem a arquitetura a terem partes que avaliam expressões lógicas e também de probabilidade bayesiana através de modificações de méotodos de gradientes como AdamW para tentar regularizar alguns desses tensores em formas que talvez sejam mais simétricas ou talvez que sigam algum tipo de equação (se inspirar na Física?)...


Outra avenida de experimentação importante é quanto permitir a recursividade, não necessariamente em todos os layers nem necessariamente em todos os blocos, mas pelo menos em partes deles.Uma ideia é rodar metade dos steps com a estrutura "padrão" e depois criar um tensor de dependências entre os diversos blocos de atenção que podem ter valores 0 ou 1, inicializasse com todos os blocos dependendo dos blocos da camada anterior, daí se reinicia o resto dos steps permitindo que esse tensor de dependências seja atualizado para trocar de 1 para 0 e assim permitir que diversos blocos sejam executados em paralelo na parte recursiva.


Outra avenida de experimentação é tentar limitar os blocos de atenção não apenas com a máscara de tokens mas para prestar atenção também em só parte dos tensores do espaço latente, não necessariamente à dimensão do espaço latente como um todo.

Os papers que estão na pasta docs apresentam também diversas ideias que devem ser experimentadas em conjunto. O ideal é ter um script que rode primeiro cada uma dessas ideias independentemente e depois tente (talvez de forma aleatória se forem muitas combinações possíveis) misturar diversas das ideias e ver o resultado final tabelado de todas tentativas.
