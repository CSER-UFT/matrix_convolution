# Auditoria da campanha completa no Saruê

## Identificação

- Campanha: `full_20260922_095004`
- Arquivo recebido: `condor_full_results.tar.gz`
- SHA-256 esperado e encontrado: `5d84ad66c37f36cd588ebc914ee40b5bfa4aa08c0dffd88ba5a2405017a6d66f`
- Execução: HTCondor, encerramento normal com código `0`
- Início registrado: 22/09/2026 às 09:50:04
- Final registrado pelo job: 22/09/2026 às 20:33:34
- Uso remoto registrado pelo Condor: 10h15min de CPU de usuário e 16min de sistema

O arquivo compactado está íntegro e não contém caminhos inseguros. Os 38 arquivos de entrada existentes na cópia local correspondem aos hashes preservados pela campanha.

## Completude dos dados

| Item | Encontrado | Esperado | Situação |
| --- | ---: | ---: | --- |
| Configurações de programas | 276 | 276 | Completo |
| Repetições por configuração | 50 | 50 | Completo |
| Linhas em `results_raw.csv` | 13.800 | 13.800 | Completo |
| Linhas em `results_summary.csv` | 276 | 276 | Completo |
| Comparações pareadas brutas | 10.350 | 10.350 | Completo |
| Resumos de comparações | 207 | 207 | Completo |
| Comparações de erro numérico | 207 | 207 | Completo |
| Arquivos P1 | 13.800 | 13.800 | Completo |
| Arquivos P2 | 13.800 | 13.800 | Completo |
| Saídas dos programas | 27.600 | 27.600 | Completo |
| Arquivos de erro | 27.600 | 27.600 | Todos vazios |

Foram encontrados 82.808 arquivos no total. Não há identificadores duplicados, repetições ausentes, valores numéricos não finitos ou checksums divergentes entre P1 e P2. As células vazias dos CSVs correspondem apenas a `aux_input` nas execuções GEMM, que não usam kernel auxiliar.

## Qualidade das medições do perf

- Nenhum `<not counted>`.
- Nenhum `<not supported>`.
- Nenhum erro de permissão nos arquivos da campanha.
- Nenhuma passagem abaixo de 99% de tempo ativo.
- Todos os eventos esperados estão presentes.
- ASLR desativado.
- Afinidade fixada na CPU 0.
- NMI watchdog permaneceu ativo, mas não ocorreu multiplexação.
- O Condor alocou quatro CPUs ao job, embora o programa tenha usado uma CPU fixada.

Eventos P1:

- `instructions:u`
- `cycles:u`
- `cache-references:u`
- `cache-misses:u`
- `duration_time`

Eventos P2:

- `branches:u`
- `branch-misses:u`
- `L1-dcache-load-misses:u`
- `LLC-load-misses:u`

O tempo foi coletado em P1 por `duration_time`. P2 não possui outro tempo independente; por isso `time_p2` repete `time_p1` no CSV. O campo `time_sec` representa o tempo de P1 e não a média de duas medições independentes.

## Ambiente registrado

- Servidor: Saruê
- CPU: Intel Xeon E5-2407 v2, 4 núcleos e 4 threads
- Frequência nominal máxima: 2,4 GHz
- Memória: 32 GB
- Kernel: Ubuntu `5.4.0-216-generic`
- GCC: 9.4.0
- perf: 5.4.291
- Governador: `powersave`
- NMI watchdog: ativo

O campo `cpus_allowed=0-63` informa CPUs permitidas pelo ambiente do processo, enquanto `lscpu` registra somente as CPUs 0-3 como online. A execução foi fixada corretamente na CPU 0.

## Limitação principal: variabilidade do tempo

O governador `powersave` permitiu grande variação de frequência. A frequência efetiva estimada por `cycles/time` variou aproximadamente entre 0,63 e 2,38 GHz:

| Percentil | Frequência efetiva aproximada |
| --- | ---: |
| 5% | 1,02 GHz |
| 25% | 1,39 GHz |
| 50% | 1,69 GHz |
| 75% | 2,00 GHz |
| 95% | 2,36 GHz |

Consequentemente:

- CV mediano do tempo: 17,38%;
- CV no percentil 95 do tempo: 23,71%;
- CV máximo do tempo: 26,30%;
- meia largura mediana do IC 95% da média do tempo: aproximadamente 4,94%;
- CV mediano dos ciclos: 0,257%;
- CV no percentil 95 dos ciclos: 2,02%;
- CV mediano das instruções: aproximadamente `1,38 × 10⁻⁸`.

As instruções são praticamente determinísticas e os ciclos são muito mais estáveis que o tempo. Portanto:

1. instruções, ciclos, IPC, contadores de cache, checksums e erros numéricos podem ser analisados;
2. resultados de tempo devem ser apresentados com intervalo de confiança e aviso sobre o governador;
3. para usar segundos como evidência principal do TCC, recomenda-se repetir a campanha com governador `performance`;
4. não se deve corrigir artificialmente o tempo dividindo ciclos por 2,4 GHz, pois isso produziria tempo estimado, não tempo observado.

## Resultados de precisão: float contra double

O speedup é definido como `double / float`. Valor maior que 1 indica vantagem de `float`.

### Mediana por tamanho, usando ciclos

| Operação | N=512 | N=1024 | N=2048 | Interpretação |
| --- | ---: | ---: | ---: | --- |
| Convolução contígua | 1,106× | 1,049× | 1,052× | `float` usa menos ciclos |
| Convolução por linhas | 0,843× | 0,920× | 0,926× | `float` usa mais ciclos |
| GEMM | 1,239× | 1,098× | 1,085× | `float` usa menos ciclos |

Na convolução contígua, `float` executou cerca de 37% mais instruções, mas obteve IPC maior e menos ciclos. Na convolução por linhas, `float` executou cerca de 24% mais instruções e também consumiu mais ciclos. No GEMM, a quantidade de instruções de `float` e `double` foi praticamente igual, mas `float` apresentou menos ciclos e menos faltas de cache.

O tempo sugere vantagem de `float` também na convolução por linhas, em conflito com os ciclos. Essa divergência reforça que a frequência variável afetou as comparações em segundos.

## Resultados de organização da memória

Comparando convolução por linhas com convolução contígua, o speedup foi calculado como `linear / malloc`. Valor maior que 1 indica que a versão por linhas foi mais rápida.

| Precisão | Técnica | Speedup mediano em ciclos da versão por linhas |
| --- | --- | ---: |
| double | completo | 1,383× |
| double | skip_kernel | 1,158× |
| float | completo | 1,209× |
| float | skip_kernel | 1,089× |

A implementação por linhas apresentou menos instruções e menos ciclos em todos esses grupos. Apesar das múltiplas alocações, os acessos por ponteiros de linha evitaram parte da aritmética de indexação usada no vetor linear. Esse resultado deve ser discutido como propriedade das implementações avaliadas, e não como regra geral de que alocação por linhas sempre supera memória contígua.

## Resultados das técnicas aproximadas

### Convolução

Speedup mediano em ciclos da aproximação sobre a versão completa:

| Layout e precisão | K=3 | K=5 | K=7 |
| --- | ---: | ---: | ---: |
| Contígua double | 1,624× | 2,022× | 2,207× |
| Contígua float | 1,718× | 2,065× | 2,239× |
| Por linhas double | 1,301× | 1,623× | 2,284× |
| Por linhas float | 1,512× | 1,852× | 2,157× |

O ganho cresce de forma geral com o tamanho do kernel, pois `skip_kernel` elimina uma parcela maior do trabalho da versão completa.

### GEMM

Speedup mediano em ciclos de `skip_k` sobre a versão completa:

| Precisão | B=8 | B=16 | B=32 | B=64 | B=128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| double | 1,498× | 2,291× | 2,206× | 1,869× | 1,979× |
| float | 1,391× | 2,158× | 2,250× | 2,094× | 1,994× |

O melhor bloco em ciclos para as versões completas foi 8, com exceção de `float` em N=512, cujo melhor bloco foi 16. Para `skip_k`, o melhor bloco foi 16 em todos os tamanhos e precisões.

## Distribuições das entradas

A distribuição dos valores quase não alterou os ciclos das convoluções. Mantendo programa, tamanho e kernel, a variação relativa entre as três distribuições teve:

- mediana: 0,095%;
- percentil 95: 0,444%;
- máximo: 0,998%.

Isso é coerente com núcleos cujos laços e acessos à memória não dependem dos valores numéricos.

## Erro numérico da precisão

As diferenças entre versões completas `float` e `double` foram pequenas:

- convoluções: erro relativo médio típico na ordem de `10⁻⁸` a `10⁻⁷`;
- GEMM: erro relativo médio entre aproximadamente `2,44 × 10⁻⁷` e `4,79 × 10⁻⁷`.

Isso confirma que a troca para `float` introduziu erro pequeno para as entradas avaliadas.

## Erro numérico das aproximações

### skip_kernel

Para distribuição uniforme, a mediana do erro relativo médio ficou entre aproximadamente 11,0% e 17,6%. Para distribuição exponencial, ficou entre aproximadamente 18,6% e 26,6%.

Na distribuição normal, o erro relativo ponto a ponto chegou a valores muito altos, entre aproximadamente 8,2 e 14,1 vezes o valor de referência. Isso ocorre porque a saída pode ficar próxima de zero, tornando a divisão usada no erro relativo instável. Para essa distribuição, RMSE e erro absoluto médio devem receber mais destaque que o erro relativo ponto a ponto.

Os resultados de erro de `conv_linear` e `conv_malloc` são iguais, como esperado: o layout muda a organização da memória, mas não a operação matemática.

### skip_k

Erro relativo médio mediano:

| N | double | float |
| --- | ---: | ---: |
| 512 | 3,049% | 3,049% |
| 1024 | 2,188% | 2,188% |
| 2048 | 1,555% | 1,555% |

O erro relativo diminuiu com o tamanho da matriz, enquanto o erro absoluto e o RMSE cresceram com a escala dos valores acumulados.

## Características microarquiteturais

Medianas aproximadas de IPC:

- convoluções: de 2,06 a 3,04;
- GEMM: de 1,47 a 1,91;
- as versões `float` completas apresentaram IPC maior que as equivalentes `double`;
- as versões GEMM aproximadas também apresentaram IPC maior que as completas.

Taxas de falta de cache medianas:

- convoluções contíguas: aproximadamente 19% a 30%;
- convoluções por linhas: aproximadamente 33% a 38%;
- GEMM: aproximadamente 0,13% a 1,11%.

Taxas devem ser apresentadas junto das contagens absolutas. Uma taxa elevada pode ocorrer com poucas referências contabilizadas e não deve ser interpretada isoladamente.

## O que aproveitar dos gráficos do projeto sort-evaluation

O projeto do professor usa dois níveis de visualização:

1. `plot_metrics.py`: gráficos sistemáticos por algoritmo e por número de threads;
2. `analyze_perf.py`: gráficos interpretativos de complexidade, microarquitetura, memória, distribuição, correlação e reprodutibilidade.

Práticas que devem ser mantidas:

- Matplotlib com saída vetorial em PDF e 300 DPI;
- estilo único de fontes, cores, marcadores, grade e dimensões;
- médias acompanhadas de intervalo de confiança de 95%;
- cálculo do intervalo com distribuição t de Student e 50 repetições;
- boxplots para mostrar a distribuição das repetições;
- escala logarítmica quando os valores cobrem ordens de grandeza;
- heatmaps para várias métricas normalizadas;
- gráficos separados por tema em subpastas;
- fechamento explícito das figuras depois de salvar;
- CSV bruto como fonte das distribuições e CSV resumido como fonte das tendências.

Cuidados ao adaptar:

- não usar gráficos de threads, Amdahl ou energia, pois não fazem parte desta campanha;
- evitar gerar centenas de PDFs sem uma pergunta analítica clara;
- usar rótulos em português e nomes legíveis para operações e técnicas;
- manter cores fixas: precisão pela cor, técnica por estilo e layout por marcador ou painel;
- não colocar todas as 12 versões na mesma figura quando isso prejudicar a leitura;
- mostrar valores absolutos além de heatmaps normalizados;
- para tempo, exibir boxplot, IC 95% e aviso do governador `powersave`;
- priorizar ciclos para conclusões de desempenho desta campanha.

## Conjunto recomendado de gráficos para o TCC

### Desempenho principal

1. Ciclos por N, separados por operação, precisão e técnica.
2. Speedup de `float` sobre `double`, usando ciclos e IC 95%.
3. Speedup de `skip_kernel` e `skip_k` sobre o completo de mesma precisão.
4. Speedup do layout por linhas sobre o layout contíguo.
5. Tempo total por N com boxplots e indicação explícita da variabilidade.
6. Bloco GEMM: ciclos por tamanho de bloco, com uma figura para cada N.

### Qualidade numérica

7. Erro de `float` contra `double`.
8. Erro das aproximações por kernel, bloco, distribuição e N.
9. Dispersão speedup versus RMSE ou erro relativo, formando o compromisso desempenho-qualidade.

### Microarquitetura e reprodutibilidade

10. IPC por versão.
11. Taxas e contagens de cache misses.
12. Heatmap normalizado de instruções, ciclos, IPC e cache.
13. CV de tempo, ciclos e instruções em escala logarítmica.
14. Matriz de correlação entre contadores e desempenho.

## Parecer final

A campanha terminou corretamente e produziu todos os dados planejados. Não há falha estrutural, perda de arquivos, erro de contador, multiplexação ou divergência de resultados.

Os dados de ciclos, instruções, IPC, cache e erro numérico são adequados para análise. O tempo total foi medido corretamente, mas sob frequência altamente variável causada pelo governador `powersave`. Ele pode ser explorado com transparência e intervalos de confiança, porém não deve ser a única base para conclusões de desempenho. Uma repetição com governador `performance` é recomendada antes de considerar a campanha definitiva para afirmações em segundos.

Os próximos scripts de gráficos devem ler os CSVs preservados na campanha, gerar PDFs e PNGs, manter os dados brutos intactos e produzir também tabelas derivadas que documentem cada comparação.
