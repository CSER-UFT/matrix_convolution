# Entradas fixas

Os arquivos `.bin` são gerados uma única vez por `python3 scripts/generate_inputs.py all`. O formato contém um cabeçalho de 40 bytes que identifica a precisão dos elementos. `inputs/float/` armazena valores nativos `float` e `inputs/double/` armazena valores nativos `double`.

As duas versões partem da mesma sequência determinística; a representação em `float` é feita pelo gerador, antes dos experimentos. Assim, nenhum executável converte entradas durante a medição. `SHA256SUMS` registra a identidade exata de todos os arquivos usados.

O conjunto completo ocupa aproximadamente 330 MiB. Não edite os binários manualmente nem execute novamente o gerador no meio de uma campanha.
