# Índice dos gráficos

Cada figura é gerada em PDF vetorial e PNG a 300 DPI. Os gráficos de tempo devem ser interpretados com a limitação do governador `powersave`; para esta campanha, ciclos são a métrica principal de desempenho.

Total: **37 figuras**, em **74 arquivos gráficos**.

| Categoria | Figura | Pergunta | Filtros | Fonte |
| --- | --- | --- | --- | --- |
| `01_desempenho` | `conv_linear_cycles` | Como ciclos escala com N, precisão e técnica? | dist=0; 50 repetições; média e IC 95% | `results_raw.csv` |
| `01_desempenho` | `conv_linear_instructions` | Como instruções escala com N, precisão e técnica? | dist=0; 50 repetições; média e IC 95% | `results_raw.csv` |
| `01_desempenho` | `conv_linear_ipc` | Como instruções por ciclo (ipc) escala com N, precisão e técnica? | dist=0; 50 repetições; média e IC 95% | `results_raw.csv` |
| `01_desempenho` | `conv_linear_time_sec` | Como tempo observado (s) escala com N, precisão e técnica? | dist=0; 50 repetições; média e IC 95% | `results_raw.csv` |
| `01_desempenho` | `conv_malloc_cycles` | Como ciclos escala com N, precisão e técnica? | dist=0; 50 repetições; média e IC 95% | `results_raw.csv` |
| `01_desempenho` | `conv_malloc_instructions` | Como instruções escala com N, precisão e técnica? | dist=0; 50 repetições; média e IC 95% | `results_raw.csv` |
| `01_desempenho` | `conv_malloc_ipc` | Como instruções por ciclo (ipc) escala com N, precisão e técnica? | dist=0; 50 repetições; média e IC 95% | `results_raw.csv` |
| `01_desempenho` | `conv_malloc_time_sec` | Como tempo observado (s) escala com N, precisão e técnica? | dist=0; 50 repetições; média e IC 95% | `results_raw.csv` |
| `01_desempenho` | `gemm_cycles` | Como ciclos escala com N, precisão e técnica? | dist=-1; 50 repetições; média e IC 95% | `results_raw.csv` |
| `01_desempenho` | `gemm_instructions` | Como instruções escala com N, precisão e técnica? | dist=-1; 50 repetições; média e IC 95% | `results_raw.csv` |
| `01_desempenho` | `gemm_ipc` | Como instruções por ciclo (ipc) escala com N, precisão e técnica? | dist=-1; 50 repetições; média e IC 95% | `results_raw.csv` |
| `01_desempenho` | `gemm_time_sec` | Como tempo observado (s) escala com N, precisão e técnica? | dist=-1; 50 repetições; média e IC 95% | `results_raw.csv` |
| `02_comparacoes` | `precisao_conv_linear` | A precisão float reduz os ciclos da versão completa? | comparison=precision; dist=0 | `comparisons_raw.csv` |
| `02_comparacoes` | `precisao_conv_malloc` | A precisão float reduz os ciclos da versão completa? | comparison=precision; dist=0 | `comparisons_raw.csv` |
| `02_comparacoes` | `precisao_gemm` | A precisão float reduz os ciclos da versão completa? | comparison=precision; dist=-1 | `comparisons_raw.csv` |
| `02_comparacoes` | `aproximacao_conv_linear` | Quanto a técnica aproximada reduz os ciclos em cada configuração? | approximations; dist=0; média e IC 95% | `comparisons_raw.csv` |
| `02_comparacoes` | `aproximacao_conv_malloc` | Quanto a técnica aproximada reduz os ciclos em cada configuração? | approximations; dist=0; média e IC 95% | `comparisons_raw.csv` |
| `02_comparacoes` | `aproximacao_gemm` | Quanto a técnica aproximada reduz os ciclos em cada configuração? | approximations; dist=-1; média e IC 95% | `comparisons_raw.csv` |
| `02_comparacoes` | `layout_cycles` | Qual implementação de convolução usa menos recursos? | conv_linear pareada com conv_malloc; dist=0 | `results_raw.csv` |
| `02_comparacoes` | `layout_instructions` | Qual implementação de convolução usa menos recursos? | conv_linear pareada com conv_malloc; dist=0 | `results_raw.csv` |
| `03_gemm_blocos` | `gemm_blocos_ciclos` | Qual tamanho de bloco minimiza os ciclos da GEMM? | dist=-1 (GEMM não usa distribuição) | `results_raw.csv` |
| `04_qualidade_numerica` | `precisao_erro_conv_linear` | Qual erro é introduzido ao trocar double por float? | comparison=precision | `accuracy.csv` |
| `04_qualidade_numerica` | `precisao_erro_conv_malloc` | Qual erro é introduzido ao trocar double por float? | comparison=precision | `accuracy.csv` |
| `04_qualidade_numerica` | `precisao_erro_gemm` | Qual erro é introduzido ao trocar double por float? | comparison=precision | `accuracy.csv` |
| `04_qualidade_numerica` | `skip_kernel_error_rel_mean` | Como o erro do skip_kernel varia com K, N, precisão e distribuição? | conv_linear; aproximações | `accuracy.csv` |
| `04_qualidade_numerica` | `skip_kernel_rmse` | Como o erro do skip_kernel varia com K, N, precisão e distribuição? | conv_linear; aproximações | `accuracy.csv` |
| `04_qualidade_numerica` | `skip_k_error_rel` | Como o erro do skip_k varia com N, bloco, precisão e distribuição? | Todos os casos aplicáveis | `accuracy.csv` |
| `05_compromisso` | `tradeoff_conv_linear` | Quanto de erro acompanha o ganho de ciclos da aproximação? | Todos os casos aplicáveis | `comparisons_summary.csv + accuracy.csv` |
| `05_compromisso` | `tradeoff_conv_malloc` | Quanto de erro acompanha o ganho de ciclos da aproximação? | Todos os casos aplicáveis | `comparisons_summary.csv + accuracy.csv` |
| `05_compromisso` | `tradeoff_gemm` | Quanto de erro acompanha o ganho de ciclos da aproximação? | Todos os casos aplicáveis | `comparisons_summary.csv + accuracy.csv` |
| `06_microarquitetura` | `ipc_cache` | Como IPC e faltas de cache distinguem as implementações? | Todos os casos aplicáveis | `results_summary.csv` |
| `06_microarquitetura` | `perfil_contadores` | Quais programas têm os maiores valores relativos em cada contador? | N=2048; dist=0; K=5/B=16 | `results_summary.csv` |
| `07_reprodutibilidade` | `cv_metricas` | Quais métricas variaram mais entre as 50 repetições? | N=2048; dist=0; K=5/B=16 | `results_summary.csv` |
| `07_reprodutibilidade` | `boxplot_cycles` | Como as 50 medições de ciclos se distribuem? | N=2048; dist=0; K=5/B=16 | `results_raw.csv` |
| `07_reprodutibilidade` | `boxplot_time_sec` | Como as 50 medições de tempo observado (s) se distribuem? | N=2048; dist=0; K=5/B=16 | `results_raw.csv` |
| `08_diagnosticos` | `efeito_distribuicao` | A distribuição dos valores altera o custo das convoluções? | Todos os casos aplicáveis | `results_summary.csv` |
| `08_diagnosticos` | `correlacao_metricas` | Quais métricas variam juntas no conjunto completo? | Todos os casos aplicáveis | `results_raw.csv` |
