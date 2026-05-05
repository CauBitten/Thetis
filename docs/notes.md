# Notas de pesquisa — Thetis

Decisões registradas em ordem cronológica reversa. Cada entrada explica
o que foi escolhido, alternativas consideradas e os riscos conhecidos.

---

## 2026-05-05 — Esqueletos tratados como vídeo (não como coordenadas T×J×C)

Inspecionando `dataset/VIDEO_Skelet2D/` e `dataset/VIDEO_Skelet3D/`, todos os
arquivos são `.avi` 640×480 com o esqueleto **renderizado** sobre fundo preto
— não há coordenadas brutas (`.txt`/`.csv`/`.mat`/`.npy`) em lugar algum do
clone do dataset. O README do THETIS confirma: a calibração de pose falhou
em ~38% das sequências (1217/1980), e o que existe é o vídeo de visualização.

**Decisão**: o `ThetisDataset` carrega esqueleto como vídeo, retornando os
frames em chaves explícitas `skeleton_2d_video` e `skeleton_3d_video`
(shape `(T, H, W, 3)` uint8). Coordenadas T×J×C ficam para um futuro
`src/features/pose.py` que extrai pose a partir do RGB (MediaPipe / OpenPose)
e materializa em `data/processed/pose/`. As chaves `skeleton_2d_coords` e
`skeleton_3d_coords` foram **reservadas** no `augment.py` (transforms de
coords são no-op enquanto as chaves não existem no dict).

**Alternativas consideradas**:

- Extrair pose on-the-fly no `__getitem__`: rejeitada — adiciona MediaPipe
  como dep core, é lenta o suficiente para gargalar o data loader.
- Pular esqueletos completamente: rejeitada — perde sinal disponível.

**Riscos**:

- O nome `skeleton_2d`/`skeleton_3d` em `MODALITIES` é intencionalmente o
  mesmo do dir físico, mas a chave de saída é `*_video`. Se um dia houver
  coordenadas, o esquema continua não-ambíguo, mas é preciso lembrar que
  `path_skeleton_2d` no manifest aponta para o **vídeo de visualização**.

---

## 2026-05-05 — Divisão de classes 7/2/3 com seed=42

Mantive o default sugerido pelo orientador (7/2/3) e tornei configurável via
`--train-classes/--val-classes/--test-classes`. Com `seed=42` a divisão
determinística cai em:

- `meta_train` (7): `backhand, backhand_volley, forehand_flat, forehand_openstands, forehand_slice, kick_service, smash`
- `meta_val` (2): `backhand_slice, flat_service`
- `meta_test` (3): `backhand2hands, forehand_volley, slice_service`

**Observação**: a divisão **não** agrupa os três saques em meta-test (o que
seria interessante para testar generalização a uma família de movimentos
diferente). A seed atual espalha saques: `kick_service` em train,
`flat_service` em val, `slice_service` em test. Pode valer a pena fixar uma
seed específica que isole famílias (todos os saques em test, p.ex.) — fica
como decisão futura quando começarem os experimentos comparativos.

**Riscos**:

- Com 12 classes total, 5-way + 7/2/3 só cabe em meta_train (val tem 2,
  test tem 3 — abaixo de 5). O CLI hoje **pula** os splits subdimensionados
  com warning e marca `eligible: false` em `split_metadata.json`. Para
  obter episódios em val/test com 5-way, é preciso ajustar manualmente
  (`--train-classes 5 --val-classes 5 --test-classes 2`, p.ex.) ou usar
  `--n-way` menor.
- Validação reportada com classes diferentes a cada nova seed pode tornar
  comparação entre runs frágil — manter seed fixa em todos os experimentos
  e registrar split_metadata.json no log do experimento.

---

## 2026-05-05 — Episódios serializados como JSONL

`data/episodes/{split}/episodes.jsonl`, uma linha por episódio (objeto JSON).
Schema: `episode_id, split, n_way, k_shot, q_query, classes, support, query,
speed_split_mode, seed_used`. Os `support` e `query` são dicts
`{class_label: [sample_id, ...]}` — só IDs, sem duplicar features.

**Por quê JSONL e não Parquet**:

- Inspecionável com `head` / `grep` / `jq` — útil para debugar splits e
  verificar speed-robustness manualmente.
- Sem dependência adicional (`pyarrow` não entra no projeto).
- Tamanho desprezível: 1000 episódios 5-way 5-shot 15-query ≈ 200 KB.
  Parquet daria ~80 KB; a economia não justifica a perda de inspeção.
- Schema com lista variável de IDs por classe é mais natural em JSON do
  que em colunas Parquet.

**Risco**: se em algum momento os episódios forem materializados com
features (não só IDs), JSONL vira ineficiente — mudar para Parquet ou shards
binários. Hoje, os IDs apontam para o manifest e o `ThetisDataset` recarrega
features sob demanda.

---

## 2026-05-05 — Manifest wide-format (1 linha por sample, paths por modalidade)

Mudança em relação à versão anterior do `loader.py` (commit `461719e`),
que produzia long-format (~8374 linhas, uma por `(sample, modality)`).
Wide-format dá ~1980 linhas, com colunas `path_rgb`, `path_depth`,
`path_mask`, `path_skeleton_2d`, `path_skeleton_3d`. Modalidades faltantes
são string vazia `""`, não `NaN`.

**Por quê**:

- Casa diretamente com a API `ThetisDataset(modalities=[...])`: filtrar
  amostras com modalidades ausentes vira `df[df[col] != ""]` (uma linha
  por amostra simplifica `__len__`).
- O `sample_id` em wide-format é único e legível: `p10_forehand_volley_s1`,
  vs. `p10_forehand_volley_1_rgb_<hash>` em long-format.
- Long-format dava a ilusão de 8374 amostras quando, de fato, são 1980
  amostras observadas em até 5 modalidades.

**Operacional**:

- Todo leitor de manifest usa `pd.read_csv(..., keep_default_na=False)`
  para que `""` permaneça `""` no roundtrip CSV (sem virar `NaN`).
  Tests, `ThetisDataset`, `EpisodeSampler` — todos seguem essa convenção.
- Se precisar long-format para EDA, fazer `df.melt()` no notebook;
  não vale ter dois arquivos como source-of-truth.

---

## 2026-05-05 — `decord` opcional, `opencv-python` no core

Em `pyproject.toml`, `opencv-python>=4.8` é dependência obrigatória e
`decord>=0.6` está em `[project.optional-dependencies] fast`.
`_read_video()` tenta decord primeiro e cai em cv2 (`ImportError`).

**Por quê**:

- Em WSL2 + Python 3.13 (a config local), `decord` não tem wheels e
  build-from-source falha. opencv-python tem wheels manylinux estáveis.
- A diferença de performance importa em treino sério (decord é 3-5×
  mais rápido), mas para a smoke-test e os notebooks de EDA, cv2
  basta. Quem quiser performance extra: `uv add --optional fast decord`.

**Risco**: cv2 lê os frames como BGR; já normalizo para RGB no
`_read_video_cv2`. Se algum lugar do código esquecer disso (futuro
`src/features/pose.py`?), as cores ficam invertidas — vale documentar
no docstring do dataset (já está).

---

## 2026-05-05 — Splits cross-subject e cross-action descartados

A versão antiga do `loader.py` (commit `461719e`) escrevia
`data/splits/cross_subject.csv` e `data/splits/cross_action.csv` para
baselines não-episódicos. Decidimos **descartar** essas funções na nova
implementação.

**Por quê**: o protocolo FSAR é **inerentemente episódico**. Cross-action
splits são reproduzidos pelo `episode_sampler.py` (que particiona classes
em meta-train/val/test). Cross-subject não tem análogo direto em FSAR
mas, se um dia precisar, dá para gerar um manifest extra com a partição
de atores — não há motivo para isso ser responsabilidade do `loader.py`.

**Risco**: se quisermos comparar com algum baseline clássico não-episódico
(supervised closed-set), o split antigo precisa ser regenerado. Está em
`git show 461719e:src/data/loader.py` — função `build_cross_subject_split`
e `build_cross_action_split`.

---

## 2026-05-05 — Pandas pin relaxado para `>=2.2`

`pyproject.toml` original tinha `pandas>=3.0.2`. Pandas 3.x trouxe breaking
changes em copy-on-write default e dtypes nullable. `>=2.2` é mais seguro
e ainda moderno (Arrow-backed dtypes disponíveis quando precisar).

**Risco**: se um dia importarmos código que depende de comportamento
3.x-only (ex: `from pandas.api.types import some_3x_only_thing`), precisa
upgrade. Hoje, `loader.py` e `episode_sampler.py` usam só APIs estáveis
desde 2.0.

---

## 2026-05-05 — Pré-flight de splits subdimensionados (sampler CLI)

Quando o usuário pede `--n-way 5` mas o split tem menos classes (default
val=2, test=3), o `main()` do sampler **pula** o split com warning em vez
de falhar com exception. Splits pulados ficam marcados com
`"eligible": false` no `split_metadata.json`, e os JSONLs simplesmente
não são escritos para aquele split.

**Por quê**: falha hard interrompe o pipeline depois de já ter escrito
parte dos arquivos (estado parcial é pior que nenhum). E a maior parte
do valor já está em meta_train (1000 episódios) — não faz sentido abortar
porque val/test não cabem.

**Risco**: silêncio escondendo erro. Mitigado pela mensagem de warning
explícita ("Use --train/--val/--test-classes or lower --n-way") e pelo
campo `eligible` em metadata. Para CI/script automatizado, adicione
`--strict` (já existe a flag) — mas hoje ela só governa amostragem
intra-classe, não viabilidade de split. Posso estender se virar problema.
