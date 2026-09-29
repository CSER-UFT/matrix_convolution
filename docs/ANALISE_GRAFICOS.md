# Geração dos gráficos da campanha completa

Os CSVs consolidados da campanha publicada estão em `results/full_20260922_095004`. Os milhares de arquivos brutos do `perf` e o pacote original não são versionados. Quando um novo pacote estiver disponível, `prepare_analysis_data.py` pode extrair somente os CSVs, o ambiente e os hashes necessários à análise.

## Preparação do ambiente

No Ubuntu, dentro da pasta do projeto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Gerar todos os gráficos

No Ubuntu:

```bash
python3 scripts/generate_plots.py \
  --data results/full_20260922_095004 \
  --output results/figures_full_20260922 --clean
```

No Windows:

```powershell
python scripts\generate_plots.py `
  --data results\full_20260922_095004 `
  --output results\figures_full_20260922 --clean
```

O mesmo processo pode ser executado no Ubuntu com:

```bash
make plots
```

As figuras serão gravadas em `results/figures_full_20260922`, organizadas por tema. Cada figura terá uma versão PDF vetorial, indicada para o texto do TCC, e uma versão PNG em 300 DPI, útil para inspeção e apresentações.

O arquivo `INDICE_GRAFICOS.md` explica a pergunta respondida, os filtros e a fonte de cada figura. O arquivo `INDICE_GRAFICOS.csv` contém o mesmo catálogo em formato tabular.

## Organização das figuras

- `01_desempenho`: ciclos, instruções, IPC e tempo observado;
- `02_comparacoes`: precisão, aproximações e organização da memória;
- `03_gemm_blocos`: efeito do tamanho de bloco na GEMM;
- `04_qualidade_numerica`: erros de `float`, `skip_kernel` e `skip_k`;
- `05_compromisso`: ganho em ciclos comparado com erro numérico;
- `06_microarquitetura`: IPC, cache e perfil normalizado de contadores;
- `07_reprodutibilidade`: coeficientes de variação e boxplots das 50 repetições;
- `08_diagnosticos`: efeito da distribuição e correlações entre métricas.

## Interpretação do tempo

A campanha foi executada com o governador da CPU em `powersave`, o que causou variação de frequência. Os gráficos de tempo são mantidos como registro do valor observado e mostram intervalo de confiança, mas os ciclos devem ser usados como métrica principal de desempenho desta campanha. Os scripts não corrigem nem estimam tempos artificialmente.

## Opções úteis

Gerar somente PDF:

```bash
python3 scripts/generate_plots.py --formats pdf --clean
```

Escolher outra pasta de saída:

```bash
python3 scripts/generate_plots.py --output results/minhas_figuras --clean
```

O parâmetro `--clean` apaga apenas a pasta informada em `--output`. Ele não altera os CSVs, os inputs ou os executáveis.

Para preparar os CSVs de um novo pacote do Condor:

```bash
python3 scripts/prepare_analysis_data.py caminho/do/pacote.tar.gz
python3 scripts/generate_plots.py --clean
```
