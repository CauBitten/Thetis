# Thetis — Few-Shot Action Recognition para Golpes de Tênis

Pesquisa e implementação de métodos de **Few-Shot Learning (FSL)** aplicados ao reconhecimento de ações específicas em vídeos de tênis. Os experimentos são conduzidos sobre o dataset [THETIS](https://github.com/THETIS-dataset/dataset), adaptado ao protocolo **N-way K-shot** padrão da literatura de Few-Shot Action Recognition (FSAR).

---

## Contexto e motivação

O reconhecimento de ações em vídeo evoluiu fortemente com arquiteturas profundas treinadas em datasets massivos (Kinetics-400, ~400 classes, ~600k vídeos). Em domínios esportivos específicos, no entanto, a anotação é cara e exige especialistas, e classes vizinhas (por exemplo, *forehand topspin* vs *forehand flat*, ou *saque plano* vs *saque slice*) têm baixa variância visual. Datasets de tênis como o Tennis7 possuem poucas dezenas de exemplos por classe, o que torna inviável o paradigma supervisionado tradicional.

Este projeto investiga como métodos de **Few-Shot Action Recognition** se comportam nesse cenário, com atenção a duas dificuldades específicas do tênis:

- variação de velocidade do mesmo golpe entre jogadores de níveis distintos (iniciante, amador, profissional);
- granularidade fina entre classes vizinhas, que compartilham boa parte da cinemática.

A avaliação segue o protocolo **N-way K-shot** consolidado por Snell et al. (Prototypical Networks): o modelo recebe N classes novas com K exemplos rotulados (suporte) e classifica consultas dessas classes.

---

## Perguntas de pesquisa

1. Entre os métodos de FSAR recentes, quais entregam melhor desempenho no reconhecimento de golpes específicos de tênis?
2. Como esses métodos lidam com a variação de velocidade do mesmo golpe entre jogadores de níveis diferentes?
3. A combinação de **pose 2D**, **fluxo óptico** e **descrições textuais** melhora a discriminação de classes de granularidade fina, em comparação com métodos puramente RGB?
4. É possível obter desempenho competitivo com apenas 5 exemplos por classe (5-way 5-shot)?

---

## Métodos avaliados

A comparação cobre famílias representativas de FSAR:

| Família | Método | Referência |
| --- | --- | --- |
| Metric learning (baseline) | Prototypical Networks | Snell et al. |
| Cross-attention temporal | TRX | Perrett et al. |
| Alinhamento multi-velocidade | MVP-Shot | — |
| Multimodal (vídeo + texto) | SAFSAR | Tang et al. |
| Baseado em pose | VPD | Hong et al. |
| Movimento denso | SOAP | — |
| Atenção bidirecional fina | BAM + CML | — |

A intenção não é propor uma arquitetura nova, mas comparar sistematicamente as abordagens no domínio do tênis.

---

## Dataset experimental: THETIS

Os experimentos usam o **THETIS (Three dimEnsional TennIs Shots)**, capturado com Kinect:

- **8.374 sequências de vídeo**
- **55 sujeitos**: p1–p31 iniciantes, p32–p55 especialistas
- **12 classes**: Backhand, Backhand 2 mãos, Backhand slice, Backhand volley, Forehand flat, Forehand open stance, Forehand slice, Forehand volley, Serviço flat, Serviço kick, Serviço slice, Smash
- **5 modalidades**: RGB, Depth, Mask (silhueta), Skeleton 2D, Skeleton 3D

> **Observação metodológica.** O THETIS, em sua forma original, possui centenas de exemplos por classe e portanto não é, por si só, um benchmark few-shot. Neste projeto ele é usado como base experimental: subamostragens controladas das 12 classes geram episódios **N-way K-shot** (tipicamente 5-way 1-shot e 5-way 5-shot) que simulam o cenário de escassez de dados. A divisão de classes entre meta-train, meta-val e meta-test é detalhada em `experiments/configs/`. A inclusão do Tennis7 como dataset complementar para validação cruzada está prevista.

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
│   ├── processed/            # Features extraídas: esqueletos normalizados,
│   │                         # fluxo óptico, embeddings de texto, etc.
│   └── episodes/             # Episódios N-way K-shot pré-amostrados
│       ├── meta_train/
│       ├── meta_val/
│       └── meta_test/
│
├── src/
│   ├── data/
│   │   ├── loader.py         # Parsing das sequências e modalidades
│   │   ├── episode_sampler.py# Amostragem N-way K-shot (suporte + consulta)
│   │   └── augment.py        # Aumentação espaço-temporal
│   ├── features/
│   │   ├── pose.py           # Extração e normalização de pose 2D
│   │   ├── optical_flow.py   # Cálculo de fluxo óptico
│   │   └── text.py           # Embeddings de descrições textuais dos golpes
│   ├── models/
│   │   ├── protonet.py       # Baseline: Prototypical Networks
│   │   ├── trx.py            # TRX (cross-attention temporal)
│   │   ├── mvp_shot.py       # MVP-Shot (alinhamento multi-velocidade)
│   │   ├── safsar.py         # SAFSAR (vídeo + texto)
│   │   └── vpd.py            # Video Pose Distillation
│   ├── training/
│   │   ├── meta_trainer.py   # Loop de meta-treino episódico
│   │   └── eval_episodic.py  # Avaliação N-way K-shot
│   └── utils/
│       ├── metrics.py        # Acurácia média por episódio, IC 95%, F1
│       └── viz.py            # Visualização de episódios e confusões
│
├── notebooks/
│   ├── 01_eda.ipynb          # Análise exploratória do THETIS
│   ├── 02_episode_design.ipynb # Construção dos splits episódicos
│   └── 03_results.ipynb      # Análise comparativa entre métodos
│
├── experiments/
│   ├── configs/              # Um .yaml por experimento (método × modalidade × N × K)
│   └── logs/                 # Logs de meta-treino (gerados automaticamente)
│
├── outputs/
│   ├── checkpoints/          # Pesos salvos (não versionados)
│   └── results/              # Métricas, plots e tabelas comparativas
│
├── tests/
│   ├── test_data.py
│   ├── test_episode_sampler.py
│   └── test_models.py
│
├── docs/
│   ├── references/           # PDFs dos artigos de FSAR
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

> Para executar comandos Python sem ativar manualmente o ambiente virtual, use `uv run <comando>`.

### 3. Clonar o dataset THETIS

```bash
git clone https://github.com/THETIS-dataset/dataset dataset
```

> O dataset contém vídeos pesados (dezenas de GB). A pasta `dataset/` está no `.gitignore` e **não deve ser versionada**.

### 4. Pré-processar dados e extrair modalidades complementares

```bash
make preprocess
# ou diretamente:
uv run python src/data/loader.py --input dataset/ --output data/ --seed 42
uv run python src/features/pose.py         --input data/processed/ --output data/processed/pose/
uv run python src/features/optical_flow.py --input data/processed/ --output data/processed/flow/
uv run python src/features/text.py         --output data/processed/text/
```

Isso gera, entre outros artefatos:

- `data/processed/manifest.csv`: tabela por amostra (sujeito, ação, modalidade, sequência, caminho).
- `data/processed/integrity_report.json`: relatório de integridade e cobertura por modalidade/classe.
- `data/processed/counts_by_modality_action.csv`: contagens por modalidade e ação.
- `data/processed/pose/`, `data/processed/flow/`, `data/processed/text/`: features das modalidades complementares.

### 5. Construir os splits episódicos

```bash
uv run python src/data/episode_sampler.py \
    --manifest data/processed/manifest.csv \
    --output data/episodes/ \
    --n-way 5 --k-shot 5 --q-query 15 \
    --episodes-per-split 1000 \
    --seed 42
```

Esse comando produz:

- `data/episodes/meta_train/`, `meta_val/`, `meta_test/`: episódios serializados.
- `data/episodes/split_metadata.json`: classes em cada partição, seed e parâmetros (N, K, Q).

A divisão de classes entre meta-train/val/test é fixada por configuração para garantir que classes vistas em meta-treino não apareçam em meta-teste.

---

## Uso

### Meta-treinar um método

```bash
uv run python src/training/meta_trainer.py \
    --config experiments/configs/protonet_skeleton3d_5w5s.yaml
```

### Avaliação episódica

```bash
uv run python src/training/eval_episodic.py \
    --checkpoint outputs/checkpoints/<run>/best.pt \
    --episodes data/episodes/meta_test/
```

A métrica principal é **acurácia média sobre N episódios de teste**, reportada com **intervalo de confiança de 95%**, conforme convenção da literatura de FSAR.

### Rodar os testes

```bash
uv run pytest tests/
```

### Comandos via Makefile

```bash
make preprocess    # extrai features e modalidades complementares
make episodes      # constrói os splits N-way K-shot
make train         # meta-treina com a config padrão
make eval          # avalia o último checkpoint em episódios de meta-teste
make test          # roda a suíte de testes
make clean         # limpa logs e arquivos temporários
```

---

## Convenções de nomenclatura do THETIS

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

## Configuração de experimentos

Cada experimento é definido por um arquivo `.yaml` em `experiments/configs/`. Exemplo:

```yaml
# experiments/configs/protonet_skeleton3d_5w5s.yaml
method: protonet              # protonet | trx | mvp_shot | safsar | vpd | ...
modalities:                   # uma ou mais; SAFSAR usa video + text
  - skeleton_3d
episode:
  n_way: 5
  k_shot: 5
  q_query: 15
  episodes_per_epoch: 200
  episodes_meta_test: 1000
optim:
  epochs: 100
  batch_size: 1               # batch é o episódio; ajuste conforme método
  learning_rate: 0.001
seed: 42
```

Cenários previstos:

- **5-way 1-shot** e **5-way 5-shot**, sobre as 12 classes do THETIS;
- variantes de modalidade: RGB puro, esqueleto, RGB + pose, RGB + fluxo óptico, RGB + texto (SAFSAR), e combinações;
- subgrupo de robustez a velocidade: episódios em que o suporte vem de iniciantes (p1–p31) e a consulta de especialistas (p32–p55), e vice-versa.

---

## Contribuição esperada

- Comparação sistemática entre famílias de métodos de FSAR aplicadas ao tênis.
- Protocolo de avaliação episódica reproduzível sobre o THETIS, com intenção de extensão ao Tennis7.
- Análise das limitações de cada família frente à variação de velocidade entre jogadores e à granularidade fina entre golpes vizinhos.
- Avaliação do ganho trazido por modalidades complementares (pose 2D, fluxo óptico, texto) em relação a baselines puramente RGB.

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
