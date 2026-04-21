# Thetis — Spatio-Temporal Action Recognition Research

Pesquisa e implementação de algoritmos espaço-temporais para reconhecimento de ações de tênis usando o dataset [THETIS](https://github.com/THETIS-dataset/dataset).

---

## Sobre o dataset

O **THETIS (Three dimEnsional TennIs Shots)** foi capturado com um dispositivo Kinect e contém:

- **8.374 sequências de vídeo**
- **55 sujeitos**: p1–p31 iniciantes, p32–p55 especialistas
- **12 classes de ações**: Backhand, Backhand 2 mãos, Backhand slice, Backhand volley, Forehand flat, Forehand open stance, Forehand slice, Forehand volley, Serviço flat, Serviço kick, Serviço slice, Smash
- **5 modalidades**: RGB, Depth, Mask (silhueta), Skeleton 2D, Skeleton 3D

---

## Estrutura do projeto

```text
Thetis/
├── dataset/                  # Clone do repositório THETIS (não versionado)
│   ├── VIDEO_RGB/
│   ├── VIDEO_Depth/
│   ├── VIDEO_Mask/
│   ├── VIDEO_Skelet2D/
│   └── VIDEO_Skelet3D/
│
├── data/                     # Dados processados (não versionados)
│   ├── processed/            # Features extraídas (esqueletos normalizados, optical flow, etc.)
│   └── splits/               # Índices de treino / validação / teste
│
├── src/
│   ├── data/
│   │   ├── loader.py         # Carregamento e parsing das sequências
│   │   └── augment.py        # Aumentação de dados espaço-temporal
│   ├── models/
│   │   ├── baseline.py       # Baseline (ex: SVM sobre features de esqueleto)
│   │   └── spatiotemporal.py # Modelos principais (ST-GCN, TCN, etc.)
│   └── utils/
│       ├── metrics.py        # Acurácia, F1, matriz de confusão
│       └── viz.py            # Visualização de sequências e resultados
│
├── notebooks/
│   ├── 01_eda.ipynb          # Análise exploratória do dataset
│   └── 02_experiments.ipynb  # Análise de resultados dos experimentos
│
├── experiments/
│   ├── configs/              # Um .yaml por configuração de experimento
│   └── logs/                 # Logs de treinamento (gerados automaticamente)
│
├── outputs/
│   ├── checkpoints/          # Pesos dos modelos salvos (não versionados)
│   └── results/              # Métricas e plots finais
│
├── tests/
│   ├── test_data.py
│   └── test_models.py
│
├── docs/
│   ├── references/           # PDFs de artigos relacionados
│   └── notes.md              # Anotações de pesquisa
│
├── README.md
├── pyproject.toml
├── setup.py
├── Makefile
└── .gitignore
```

---

## Instalação

### 1. Clonar este repositório

```bash
git clone <url-deste-repo> Thetis
cd Thetis
```

### 2. Instalar dependências com uv

```bash
uv venv
uv sync
```

> Dica: para executar comandos Python sem ativar manualmente o ambiente virtual, use `uv run <comando>`.

### 3. Clonar o dataset THETIS

```bash
git clone https://github.com/THETIS-dataset/dataset dataset
```

> **Atenção:** o dataset contém vídeos pesados (dezenas de GB). A pasta `dataset/` está no `.gitignore` e **não deve ser versionada**.

### 4. Pré-processar os dados

```bash
make preprocess
# ou diretamente (enquanto o Makefile está em implementação):
uv run python src/data/loader.py --input dataset/ --output data/ --seed 42
```

Esse comando cria automaticamente:

- `data/processed/manifest.csv`: tabela por amostra (sujeito, ação, modalidade, sequência, caminho).
- `data/processed/integrity_report.json`: relatório de integridade e cobertura por modalidade/classe.
- `data/processed/counts_by_modality_action.csv`: contagens por modalidade e ação.
- `data/splits/cross_subject.csv`: split por sujeitos (train/val/test).
- `data/splits/cross_action.csv`: split por ações (train/val/test).
- `data/splits/split_metadata.json`: metadados de seed e partições.

---

## Uso

### Treinar um modelo

```bash
uv run python src/models/spatiotemporal.py --config experiments/configs/stgcn_skeleton3d.yaml
```

### Avaliar

```bash
uv run python src/utils/metrics.py --checkpoint outputs/checkpoints/<run>/best.pt
```

### Rodar os testes

```bash
uv run pytest tests/
```

### Comandos via Makefile

```bash
make preprocess    # extrai features do dataset bruto
make train         # treina com a config padrão
make eval          # avalia o último checkpoint
make test          # roda a suíte de testes
make clean         # limpa logs e arquivos temporários
```

---

## Modalidades e convenções de nomenclatura

Cada arquivo de vídeo segue o padrão `{actor}_{action}_{sequence}.avi`.

| Código no arquivo | Ação |
| --- | --- |
| `backhand` | Backhand |
| `backhand2h` | Backhand com duas mãos |
| `bslice` | Backhand slice |
| `foreflat` | Forehand flat |
| `foreopen` | Forehand open stance |
| `fslice` | Forehand slice |
| `serflat` | Serviço flat |
| `serkick` | Serviço kick |
| `serslice` | Serviço slice |
| `smash` | Smash |
| `fvolley` | Forehand volley |
| `bvolley` | Backhand volley |

---

## Experimentos

Cada experimento é definido por um arquivo `.yaml` em `experiments/configs/`. Exemplo de configuração:

```yaml
# experiments/configs/stgcn_skeleton3d.yaml
model: stgcn
modality: skeleton_3d
split: cross_subject        # ou cross_action
epochs: 100
batch_size: 32
learning_rate: 0.001
seed: 42
```

Os splits seguem a divisão padrão da literatura:

- **Cross-subject**: treino em parte dos sujeitos, teste no restante.
- **Cross-action**: treino em parte das ações, teste nas demais.

---

## Citação

Se este trabalho usar o dataset THETIS, cite:

```bibtex
@inproceedings{gourgari2013thetis,
  title     = {THETIS: Three dimensional tennis shots a human action dataset},
  author    = {Gourgari, S. and Goudelis, G. and Karpouzis, K. and Kollias, S.},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition workshops},
  pages     = {676--681},
  year      = {2013}
}
```
