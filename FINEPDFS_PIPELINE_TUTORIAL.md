# FinePDFs Pipeline — Полное руководство
Обзор работы скрипта `run_finepdfs_pipeline.py` (репозиторий HuggingFace FinePDFs).

Этот tutorial объясняет **что именно делает код**, **какие этапы пайплайна есть**, **какие входы/выходы у каждого этапа**, и **где чаще всего возникают ошибки**. Все названия функций/классов/параметров ниже взяты из `run_finepdfs_pipeline.py`.

---

## Общая схема пайплайна

Скрипт запускает пайплайн **последовательно**, вызовами из `main()`:

1. `run_filter_pdfs_and_refetch(crawl_ids)`
2. `run_content_dedup_ocr_organize()`
3. `run_extract(gpus)`
4. `run_postprocess()`
5. `run_language_filter(languages)`
6. `run_exact_dedup(languages)`
7. `run_model_classification(languages, gpus)`
8. `run_minhash(languages)`
9. `run_push_to_hub(languages)`

При этом логика соответствует твоим 10 этапам: OCR-роутинг и извлечение/постпроцессинг разбиты на подветки (OCR vs non-OCR), а language split и две дедупликации идут отдельно.

---

## Глоссарий (обязательный)

Ниже — **специфичные термины из скрипта** и их смысл в контексте FinePDFs. (Здесь больше 20 терминов.)

1. **CommonCrawl (CC)** — публичный веб-архив. В скрипте используется CC index и WARC-данные для скачивания PDF.
2. **Crawl ID (`crawl_id`, например `CC-MAIN-2023-06`)** — идентификатор конкретного “краула” CommonCrawl. Скрипт принимает список через `--crawl-ids`.
3. **CC Index** — индекс URL/метаданных CommonCrawl. В скрипте есть два варианта чтения:
   - старые краулы: `WarcIndexReprocess`
   - новые: `ParquetReader` по `CC_INDEX_INPUT_TEMPLATE`
4. **WARC** — контейнер-формат веб-архива (HTTP ответы). В пайплайне PDF для `non_truncated` читаются из `s3://commoncrawl` через `WarcReaderFast`.
5. **`warc.paths.gz`** — список путей к warc-файлам в CC. Используется `CC_PATHS_TEMPLATE`.
6. **`ParquetReader`** — чтение parquet-частей CC индекса (новые краулы), с `adapter=index_adapter`, чтобы привести строки индекса к формату `Document`.
7. **`WarcReaderFast`** — быстрый reader WARC из CommonCrawl (S3), умеет `preserve_order` и параллелизм `workers`.
8. **`ZstdReader` / `ZstdWriter`** — чтение/запись медиа/документов в zstd-архивах. Скрипт сохраняет скачанные PDF в `PDF_SAVE_DIR` через `ZstdWriter`.
9. **`HTTPFetchReader`** — step, который реально **скачивает контент** по URL (из CC index). Параметры: `workers`, `max_retries`, `timeout`, `download_timeout`.
10. **MIME type** — тип содержимого (`application/pdf`). В скрипте фильтрация через `MimeTypeFilter(mime_types=MIME_TYPES["pdf"])`.
11. **`LambdaFilter`** — универсальный фильтр “оставить/исключить документ” по Python‑функции. Важная деталь: можно задать `exclusion_writer` для сохранения отфильтрованных документов.
12. **Truncated vs non-truncated** — ветвление на два набора документов:
    - `non_truncated`: читаются напрямую из WARC (полные ответы)
    - `truncated`: те, где в индексе/ответе были признаки обрезания; их сохраняют отдельным списком и затем читают из уже скачанных PDF (`ZstdReader(PDF_SAVE_DIR)`)
13. **Split truncation** — сохранение списков в `SPLIT_TRUNCATION_DIR`:
    - `truncated/`
    - `non_truncated/`
    - `failed_pdf_fetch/` и др.
14. **`Document` (datatrove)** — базовая сущность: текст + `metadata` + `media`. В этом пайплайне PDF лежит как `doc.media[0].media_bytes`.
15. **Exact dedup (контентный)** — удаление одинаковых PDF по **байтам** (`ExactDedupSignature/FindDedups/Filter` с `content_getter=_get_media_bytes`).
16. **Signature (dedup signature)** — компактное представление контента для поиска дубликатов (точного или MinHash).
17. **`ExactDedupSignature`** — генерирует сигнатуры для exact‑dedup и сохраняет их.
18. **`ExactFindDedups`** — строит список дубликатов по сигнатурам.
19. **`ExactDedupFilter`** — фильтрует исходные документы по найденным дубликатам, может писать исключения.
20. **OCR routing / scanned classification** — решение “нужно ли OCR”: используется `PDFScannedPredictor` (xgb‑модель) + правило `_filter_ocr`.
21. **`PDFScannedPredictor`** — предиктор, который добавляет в `metadata` признаки/оценки (например `ocr_prob`, `garbled_text_ratio`) и/или исключает документы при фейле.
22. **`_filter_ocr`** — правило: документ попадает в non-OCR ветку, если `ocr_prob < 0.2` и `garbled_text_ratio == 0.0`.
23. **Docling** — ветка извлечения текста “не OCR” (структурный разбор PDF) через `DoclingExtractor`.
24. **`DoclingExtractor`** — step, который извлекает текст/структуру из PDF; имеет `timeout` и `exclusion_writer` для ошибок.
25. **RolmOCR** — модель OCR, запускается как inference через `InferenceRunner` с `model_name_or_path="reducto/RolmOCR"` и `server_type="vllm"`.
26. **`InferenceRunner`** — универсальный inference step datatrove: принимает `rollout_fn`, `InferenceConfig`, и `output_writer`.
27. **`rollout_extract` / `rollout_postprocess`** — функции подготовки запросов/обработки ответов модели (вынесены в `pipeline_utils.*`).
28. **Postprocessing** — блок очистки/нормализации и обогащения: `AddMetadata`, `DropFailedDocuments`, `CoallesceFailedPages`, `TagBoilerplateFormatter`, `Normalize`, `TokensCounter` и др.
29. **Boilerplate** — “шумовой” текст (шапки/футеры/повторы). В скрипте обработка: `TagBoilerplateFormatter(..., drop=True)`.
30. **Language Identification (GlotLID)** — определение языка; используется дважды:
    - `LanguageTagger(..., backend="glotlid")` в `run_postprocess()`
    - `SelectBestLanguage(...)` в `run_language_filter()`
31. **Language bucket (`language_bucket`)** — выбранная “корзина языка” (например `eng_Latn`), по которой документ шардируется в папки `PER_LANGUAGE_DIR_EXACT/...`.
32. **Thresholds (`TH_VALUES_FILE`)** — файл `./thresholds/th_values.json` с порогами уверенности по языкам; минимум 0.05, а `zxx_*` разрешены всегда, если top‑1.
33. **Exact dedup (текстовый)** — точная дедупликация уже по **тексту** (после извлечения), где пробелы удаляются (`create_content_getter_exact()`).
34. **Chunking (`AddTextChunks`)** — добавление “чанков” текста в `metadata["chunks"]` для батчевой классификации качества.
35. **EDU classifier (`finepdfs_edu_classifier_{language}`)** — модель качества/полезности; скрипт проверяет наличие через `model_exists()`.
36. **DCLM classifier (`finepdfs_dclm_classifier_{language}`)** — дополнительный классификатор качества.
37. **Custom inference server** — для классификации используется `InferenceConfig(server_type="custom")` и `server_script=blocks/classification/tf_batching.py`.
38. **MinHash dedup** — “приближённая” дедупликация по шинглам \(n\)-грамм: `MinhashDedupSignature -> Buckets -> Cluster -> Filter`.
39. **`MinhashConfig`** — параметры MinHash: `hash_config` (xxhash, 64), `num_buckets`, `hashes_per_bucket`, `n_grams`.
40. **Cluster ID / Cluster size** — идентификатор и размер кластера дубликатов; в `MinhashDedupCluster(save_cluster_id=True, save_cluster_size=True)`.
41. **HuggingFaceDatasetWriter** — запись в датасет на HuggingFace Hub в формате parquet, с `adapter=push_adapter`.
42. **fw_edu subset** — второй публикуемый датасет с порогом `fw_edu_scores >= 0.5` (после классификации).

---

## Пошаговое описание (обязательный раздел!)

Ниже — описание **в соответствии с тем, как пайплайн размечен в самом `run_finepdfs_pipeline.py`**: **Step 1–7 + push**. Между Step 4 и Step 5 в коде есть отдельный ненумерованный блок `run_language_filter()` — я оставил его отдельным подразделом, потому что он реально запускается из `main()`.

### Общий порядок вызовов (как в `main()`)

```python
run_filter_pdfs_and_refetch(crawl_ids)
run_content_dedup_ocr_organize()
run_extract(args.gpus)
run_postprocess()
run_language_filter(languages=languages)
run_exact_dedup(languages=languages)
run_model_classification(languages=languages, gpus=args.gpus)
run_minhash(languages=languages)
run_push_to_hub(languages=languages)
```

---

### Step 1: `filter_pdfs_and_refetch` (`run_filter_pdfs_and_refetch`)

- **Название на русском**: **Шаг 1 — Отбор PDF в CommonCrawl и повторная загрузка контента (refetch)**
- **Что такое `Document` (очень просто)**:
  - Представь, что `Document` это **карточка одного файла**.
  - В карточке есть:
    - `metadata` — "паспорт": ссылка, язык, служебные пометки и другие поля.
    - `media` — "вложение": сами байты файла (например PDF).
    - `text` — текст документа (на Step 1 чаще всего еще пустой, потому что текст не извлекали).
  - Как он хранится:
    - **во время работы пайплайна**: как объект в памяти Python.
    - **на диске**: обычно как строки JSONL (`.jsonl.gz`) со ссылками и метаданными, а тяжелые PDF-байты отдельно в `.zstd`.
  - Идея простая: метаданные и списки документов храним отдельно от "тяжелых" бинарных файлов. Так быстрее и дешевле.

- **Что делает код (цель, очень подробно и простыми словами)**:
  - Этот шаг берет "огромный список веб-страниц" из CommonCrawl и делает из него "чистый рабочий набор PDF", с которым уже можно безопасно продолжать остальные шаги.

  1) **Сначала выбирается, как читать индекс: старый или новый формат краулов**.
     - Если `crawl_id` старый (раньше `CC-MAIN-2019-47`), используется `WarcIndexReprocess`.
     - Если новый — `ParquetReader`.
     - Почему так: у CommonCrawl со временем поменялся способ хранения индексных данных. Код учитывает это автоматически.
     - Простая аналогия: старые архивы в одном формате папок, новые — в другом; нужен разный "ключ", чтобы их открыть.

  2) **Каждая строка индекса превращается в `Document`**.
     - В новой (parquet) ветке за это отвечает `adapter=index_adapter`.
     - Что делает адаптер: берет "сырой ряд таблицы" и перекладывает его в понятные поля карточки `Document` (URL, служебные признаки, технические поля для последующего чтения).
     - На этом этапе в документе **нет самого PDF**, только информация "где его взять".

  3) **Дальше идет фильтрация в несколько проходов**.
     - `filter_non_pdf`: отсекает то, что не похоже на PDF-кандидат.
     - `filter_non_truncated`: делит документы на две группы:
       - `non_truncated` — "нормальные", не обрезанные записи;
       - `truncated` — потенциально обрезанные/неполные.
     - Технический нюанс в этом коде: `non_truncated` пишется через `exclusion_writer`, то есть в "исключенные" из текущего потока. Это нормально для datatrove, хотя визуально может выглядеть непривычно.
     - После скачивания добавляется дополнительная защита: `MimeTypeFilter`, чтобы реально проверить тип загруженного контента (PDF или нет).
     - Последняя проверка `media_bytes is not None` удаляет случаи, когда скачать файл не получилось.

  4) **Потом выполняется refetch — реальная загрузка файла по ссылке**.
     - Это делает `HTTPFetchReader`.
     - После него `doc.media[0].media_bytes` заполняется байтами PDF (если все успешно).
     - Именно здесь карточка `Document` становится "тяжелой": в ней появляется реальное бинарное вложение.

  5) **PDF-байты сохраняются в `.zstd`**.
     - `.zstd` — это файл, сжатый алгоритмом Zstandard.
     - Что это дает:
       - меньше места на диске;
       - быстрая распаковка;
       - удобно массово читать дальше через `ZstdReader`.
     - Проще говоря: это "архивированный контейнер", оптимизированный для больших потоков данных.

  6) **Почему вообще важны `truncated` и `non_truncated`**.
     - `non_truncated`: документы, которые обычно можно спокойно читать дальше напрямую из WARC в облаке.
     - `truncated`: документы, где чтение из исходного потока может быть неполным; для надежности их держат в сохраненных `.zstd`.
     - Практический эффект: на следующих шагах pipeline знает, откуда надежнее брать байты конкретного документа, и меньше ломается на "битых" входах.

- **Почему это важно архитектурно**: Step 1 — это фундамент качества. Если на этом шаге плохо отфильтровать и плохо разделить потоки, дальше будут лишние затраты на OCR/извлечение, больше пустых результатов и нестабильное качество корпуса.
- **Какие функции/классы используются**:
  - readers: `WarcIndexReprocess` (старые краулы) или `ParquetReader` (новые)
  - фильтры: `LambdaFilter(filter_non_pdf)`, `LambdaFilter(filter_non_truncated, ...)`
  - скачивание: `HTTPFetchReader`
  - MIME фильтр: `MimeTypeFilter`
  - writers: `ZstdWriter`, `JsonlWriter`
  - executor: `LocalPipelineExecutor`
- **Какие входные данные принимает**:
  - `crawl_ids: list[str]` из `--crawl-ids`
  - S3 источники:
    - индекс: `CC_INDEX_INPUT_TEMPLATE` или `CC_PATHS_TEMPLATE`
    - контент: `s3://commoncrawl`
- **Какие параметры настраиваются**:
  - `LIMIT`
  - параметры `HTTPFetchReader`: `workers`, `max_retries`, `timeout`, `download_timeout`
  - `ZstdWriter(max_file_size, output_filename)`
- **Какие выходные данные производит**:
  - `./finepdfs/data/split_truncation/truncated/*.jsonl.gz` — список “truncated”
  - `./finepdfs/data/split_truncation/non_truncated/*.jsonl.gz` — список “non_truncated” (пишется как exclusion из `filter_non_truncated`)
  - `./finepdfs/data/split_truncation/failed_pdf_fetch/*.jsonl.gz` — не удалось скачать bytes
  - `./finepdfs/data/pdf/*.zstd` — скачанные PDF байты
- **Примеры ключевых строк кода с объяснением**:

```python
if crawl_id < "CC-MAIN-2019-47":
    index_reader = WarcIndexReprocess(...)
else:
    index_reader = ParquetReader(..., adapter=index_adapter, ...)
```

```python
HTTPFetchReader(workers=15, max_retries=3, timeout=(60, 60), download_timeout=60 * 10),
MimeTypeFilter(mime_types=MIME_TYPES["pdf"]),
ZstdWriter(output_folder=PDF_SAVE_DIR, ...),
```

```python
LambdaFilter(
    lambda x: x.media[0].media_bytes is not None,
    exclusion_writer=JsonlWriter(output_folder=SPLIT_TRUNCATION_DIR.format(prefix="failed_pdf_fetch")),
),
JsonlWriter(output_folder=SPLIT_TRUNCATION_DIR.format(prefix="truncated")),
```

---

### Step 2: `content_dedup_ocr_organize` (`run_content_dedup_ocr_organize`)

- **Название на русском**: **Шаг 2 — Дедупликация по байтам PDF и маршрутизация в OCR / non‑OCR**
- **Что делает код (цель, очень подробно и простыми словами)**:
  - Если Step 1 — это "собрать PDF", то Step 2 — это "навести порядок перед тяжелой обработкой".
  - Здесь решаются сразу две большие задачи:
    1) убрать повторы одних и тех же PDF;
    2) понять, какой документ отправлять в OCR, а какой — в обычный извлекатель текста.

  - Важная идея: шаг 2 запускается **дважды** — отдельно для `truncated`, отдельно для `non_truncated`.  
    Это нужно, потому что у них разный источник байтов:
    - `non_truncated` читается из WARC;
    - `truncated` читается из сохраненных `.zstd`.

  - Внутри каждой из двух веток код выполняет **3 последовательных мини-пайплайна**.

  1) **Мини-пайплайн 2.1: создаем "отпечатки" PDF (signatures)**  
     Что происходит:
     - `JsonlReader` читает список документов (по сути карточки с ссылками/метаданными).
     - затем reader (`WarcReaderFast` или `ZstdReader`) подгружает реальные PDF-байты в `doc.media[0].media_bytes`.
     - `ExactDedupSignature` считает для каждого документа сигнатуру по байтам.
     
     Простая аналогия: для каждого файла делается "цифровой отпечаток пальца".

  2) **Мини-пайплайн 2.2: ищем повторяющиеся отпечатки (find dups)**  
     Что происходит:
     - `ExactFindDedups` читает папку с сигнатурами;
     - строит список "какие документы одинаковые".
     
     То есть на этом шаге мы еще никого не удаляем, а только составляем "черный список дублей".

  3) **Мини-пайплайн 2.3: удаляем дубли и делаем OCR-роутинг**  
     Что происходит:
     - снова читаем исходный список документов;
     - `ExactDedupFilter` убирает дубликаты по списку из пункта 2.2;
     - reader снова подгружает байты PDF;
     - `PDFScannedPredictor` оценивает, нужен ли OCR (добавляет метрики в metadata);
     - `_filter_ocr` делит поток на:
       - `ocr` (документы, где OCR нужен),
       - `non_ocr` (документы, где OCR не нужен).

  - Почему порядок именно такой:
    - сначала dedup, потом OCR-routing — чтобы не тратить OCR-классификацию на дубликаты;
    - иначе одинаковый PDF мог бы несколько раз попасть в дорогое извлечение.

  - Что происходит с `Document` на этом шаге:
    - в начале есть ссылка/метаданные;
    - после reader появляются `media_bytes`;
    - после `PDFScannedPredictor` в `metadata` появляются поля наподобие `ocr_prob` и `garbled_text_ratio`;
    - дальше документ попадает либо в папку `ocr`, либо в `non_ocr`.

  - Как работает правило роутинга (очень важно):
    - функция `_filter_ocr` возвращает `not (ocr_prob >= 0.2 or garbled_text_ratio > 0.0)`;
    - `LambdaFilter` оставляет то, где функция вернула `True` -> это и есть `non_ocr`;
    - где функция вернула `False`, записывается через `exclusion_writer` в `ocr`.
    - То есть технически в `ocr` идут "исключенные" из `non_ocr` фильтра.
- **Какие функции/классы используются**:
  - readers: `JsonlReader` (списки из Step 1) + `WarcReaderFast`/`ZstdReader` (подкачка media bytes)
  - exact dedup: `ExactDedupSignature`, `ExactFindDedups`, `ExactDedupFilter`
  - OCR routing: `PDFScannedPredictor`, `LambdaFilter(_filter_ocr)`
  - writers: `JsonlWriter`
- **Какие входные данные принимает**:
  - `SPLIT_TRUNCATION_DIR/{truncated|non_truncated}/**/*.jsonl.gz`
  - PDF bytes:
    - `non_truncated`: `WarcReaderFast(s3://commoncrawl)`
    - `truncated`: `ZstdReader(PDF_SAVE_DIR)`
- **Какие параметры настраиваются**:
  - `CONTENT_DEDUP_CONFIG = ExactDedupConfig(content_getter=_get_media_bytes)`
  - `finder_workers` у `ExactDedupSignature` (в коде `100`)
  - `PDF_SCANNED_MODEL_PATH`
  - пороги `_filter_ocr`: `ocr_prob >= 0.2` или `garbled_text_ratio > 0.0`
- **Какие выходные данные производит**:
  - `DEDUP_OUTPUT_DIR/{truncation}/non_ocr/*.jsonl.gz`
  - `DEDUP_OUTPUT_DIR/{truncation}/ocr/*.jsonl.gz`
  - `DEDUP_OUTPUT_DIR/{truncation}/removed/*.jsonl.gz`
  - `DEDUP_OUTPUT_DIR/{truncation}/failed_ocr/*.jsonl.gz`
  - `DEDUP_OUTPUT_DIR/{truncation}/sigs/*` и `.../dups/*` — технические промежуточные файлы exact-dedup
- **Примеры ключевых строк кода с объяснением**:

```python
CONTENT_DEDUP_CONFIG = ExactDedupConfig(content_getter=_get_media_bytes)
```

```python
PDFScannedPredictor(path_to_model=PDF_SCANNED_MODEL_PATH, exclude_failed=True, ...)
LambdaFilter(_filter_ocr, exclusion_writer=JsonlWriter(.../ocr))
JsonlWriter(.../non_ocr)
```

- **Возможные ошибки и как их обрабатывать**:
  - **Нет модели OCR‑классификатора** по `PDF_SCANNED_MODEL_PATH`: обеспечить файл модели; иначе документы будут исключаться (см. `exclude_failed=True`) и попадать в `failed_ocr/`.
  - **Слишком медленный exact dedup**: снизить `finder_workers`, шардинг по задачам/внешняя оркестрация.
  - **Путаница с логикой фильтра**: помнить, что `ocr` пишется как `exclusion_writer` у фильтра `non_ocr` — это нормально для данного паттерна в datatrove.

---

### Step 3: `extract` (`run_extract`)

- **Название на русском (целиком шаг)**: **Шаг 3 — Извлечение текста из PDF: ветка «как обычный документ» (Docling) и ветка OCR (RolmOCR)**

**Что делает этот шаг простыми словами.**  
После Step 2 у вас уже есть два списка на каждое из подмножеств `truncated` / `non_truncated`:

- документы в папке **`.../non_ocr`** — те, для которых пайплайн решил: «можно пытаться вытащить текст как из нормального цифрового PDF»;
- документы в папке **`.../ocr`** — те, для которых решили: «лучше распознавать как скан/картинку через OCR-модель».

Step 3 **ничего не решает заново** — он только **исполняет** это разделение: для `non_ocr` запускается **Docling**, для `ocr` — **модель RolmOCR** через сервер **vLLM**.

**Почему внутри Step 3 два больших цикла.**  
В коде `run_extract()` сначала полностью обрабатываются **все** ветки Docling (и `truncated`, и `non_truncated`), затем **все** ветки RolmOCR. Это два отдельных «под-прохода», но оба относятся к одному логическому шагу «извлечение текста».

**Почему снова фигурируют `truncated` и `non_truncated`.**  
На вход Step 3 приходят **те же карточки документов** (JSONL из Step 2), но **байты PDF** нужно снова подставить в `Document`. Откуда их брать — зависит от ветки:

- **`non_truncated`**: используется `WarcReaderFast` по `s3://commoncrawl` — то есть PDF снова читается из **исходного веб-архива WARC** по сохранённым в карточке координатам (имя warc-файла, смещение, длина и т.д., которые положил `index_adapter` на Step 1).
- **`truncated`**: используется `ZstdReader` по `PDF_SAVE_DIR` — то есть PDF берётся из **уже сохранённых на Step 1 сжатых `.zstd`**.

Идея та же, что и на Step 2: **для «надёжно сохранённых» truncated-документов не гоняем повторно лишний раз через WARC**, а читаем локальный снимок.

---

#### 3.1 Non‑OCR extraction (Docling)

- **Название на русском**: **Шаг 3.1 — Извлечение текста и структуры без OCR (Docling)**

- **Что делает код (подробно, по шагам внутри одного пайплайна):**

  1) **`JsonlReader`** читает из `./finepdfs/data/content_dedup/{truncation}/non_ocr/**/*.jsonl.gz` список документов. На этом этапе в карточке снова в основном **ссылки и метаданные**, как после Step 2 (плюс всё, что успело накопиться в `metadata`).

  2) **`reader`** (`WarcReaderFast` или `ZstdReader`) по этим данным **достаёт PDF-байты** и кладёт их в `doc.media[0].media_bytes`. Без этого Docling не из чего извлекать текст.

  3) **`DoclingExtractor`** — это «тяжёлый» шаг: он открывает PDF как документ и **пытается извлечь текст и структуру** (абзацы, таблицы, порядок блоков — в зависимости от возможностей Docling и самого PDF). Результат попадает в поле **`text`** у `Document` (и при необходимости в дополнительные поля, которые Docling/datatrove проставляют).

  4) **`JsonlWriter`** пишет **успешные** документы в `./finepdfs/data/non_ocr_docs_extracted/{truncation}/extracted/`.

  5) Если извлечение не удалось (таймаут, битый файл и т.п.), **`exclusion_writer`** у `DoclingExtractor` отправляет такой документ в `./finepdfs/data/non_ocr_docs_extracted/{truncation}/failed/`. То есть «провал» не теряется бесследно — он лежит отдельной папкой для разбора.

- **Параметры, которые явно заданы в коде:**
  - `timeout=10 * 60` — до **10 минут** на один документ в Docling; дольше — считается неуспех и уходит в `failed/`.
  - `workers` у reader: для WARC **5**, для Zstd **2** (меньше параллелизма при чтении с диска).

- **Что важно понимать про Docling-ветку:**  
  Это не «распознавание скана», а попытка **прочитать PDF как электронный документ**. Если PDF на самом деле оказался сканом, качество может быть плохим — поэтому на Step 2 такие случаи стараются отправить в `ocr`.

---

#### 3.2 OCR extraction (RolmOCR)

- **Название на русском**: **Шаг 3.2 — Извлечение текста через OCR-модель RolmOCR (сервер vLLM)**

- **Что делает код (подробно):**

  1) **`JsonlReader`** читает `./finepdfs/data/content_dedup/{truncation}/ocr/**/*.jsonl.gz` — только те документы, которые Step 2 отправил в OCR.

  2) Тот же **`reader`** (`WarcReaderFast` / `ZstdReader`) снова **подгружает байты PDF** в `media_bytes`.

  3) **`InferenceRunner`** — обёртка вокруг **языковой/vision-OCR модели**, которая крутится на отдельном сервере. В конфиге указано:
     - **`model_name_or_path="reducto/RolmOCR"`** — какую модель грузить;
     - **`server_type="vllm"`** — какой движок инференса использовать (vLLM — это сервер для быстрого батчевого запуска больших моделей на GPU);
     - **`default_generation_params={"temperature": 0.0}`** — генерация **детерминированная** (без случайных «фантазий» ответа);
     - **`dp=gpus`** — сколько GPU задействовать для параллелизма (параметр функции `run_extract(gpus)` из CLI `--gpus`);
     - **`max_concurrent_generations`**: **50** для `truncated` и **300** для `non_truncated` — сколько запросов к модели можно держать «в воздухе» одновременно; для `non_truncated` лимит выше, потому что там обычно стабильнее вход и/или больше пропускная способность ожидается.

  4) **`rollout_fn=rollout_extract`** (код вынесен в `pipeline_utils/extract_utils.py`) — функция, которая **по одному `Document` готовит запросы к модели** (например, разбиение по страницам, формат промпта, что подавать на вход) и **склеивает ответы** обратно в документ. Без неё `InferenceRunner` не знал бы, *как именно* кормить RolmOCR вашим PDF.

  5) **`output_writer=JsonlWriter(...)`** пишет результат в `./finepdfs/data/ocr_docs_extracted/{truncation}/extracted/`. В отличие от Docling-ветки, отдельного `failed`-writer в этом фрагменте пайплайна нет — неуспехи обрабатываются механизмом inference/datatrove (если документ не дошёл до writer, он не попадёт в `extracted`).

- **Зачем вообще OCR-ветка, если есть Docling:**  
  Для PDF, где текст **картинкой** (скан) или сильно «сломан» на уровне встроенного текста, **OCR-модель** часто даёт читаемый результат там, где обычный парсер видит мусор или пустоту.

- **Комментарий из кода (смысл для понимания):**  
  Авторы пишут, что в продакшене узким местом может быть **синхронная** загрузка PDF внутри шага подготовки запроса; для масштаба лучше было бы **асинхронно** заранее тянуть файлы из бакета. В учебном скрипте выбрана простота.

---

**Сводка по входам и выходам Step 3**

| Подшаг      | Откуда список документов    | Откуда байты PDF | Куда успех | Куда явный провал (если есть) |
|--------     |--------------------------   |------------------|------------|--------------------------------|
| 3.1 Docling | `content_dedup/.../non_ocr` | WARC или `.zstd` | `non_ocr_docs_extracted/.../extracted` | `.../failed` |
| 3.2 RolmOCR | `content_dedup/.../ocr` | WARC или `.zstd` | `ocr_docs_extracted/.../extracted` | (отдельный failed-writer в этом куске не задан) |

**Примеры ключевых строк в коде:**

```python
DoclingExtractor(
    timeout=10 * 60,
    exclusion_writer=JsonlWriter(
        output_folder=OUTPUT_NON_OCR_DIR.format(prefix=f"{truncation}/failed")
    ),
),
JsonlWriter(output_folder=OUTPUT_NON_OCR_DIR.format(prefix=f"{truncation}/extracted")),
```

```python
runner = InferenceRunner(
    rollout_fn=rollout_extract,
    config=InferenceConfig(
        model_name_or_path="reducto/RolmOCR",
        default_generation_params={"temperature": 0.0},
        max_concurrent_generations=50 if truncation == "truncated" else 300,
        server_type="vllm",
        metric_interval=100,
        dp=gpus,
    ),
    output_writer=JsonlWriter(output_folder=OUTPUT_OCR_DIR.format(prefix=f"{truncation}/extracted")),
)
```

---

### Step 4: `postprocess` (`run_postprocess`)

- **Название на русском (целиком шаг)**: **Шаг 4 — Постобработка: привести «сырой текст» к аккуратному виду, убрать мусор, проставить язык и (для OCR) подчистить галлюцинации**

**Что это за шаг простыми словами.**  
Step 3 уже вытащил текст из PDF — но результат часто «грязный»: повторяющиеся шапки и подвалы, странные пробелы, битые страницы, артефакты таблиц, лишние куски от картинок. У **OCR** отдельная проблема: модель иногда «придумывает» текст на **пустых** страницах (галлюцинации). Step 4 как раз **моет, нормализует, помечает метаданными** и для OCR-ветки ещё запускает **дорогую vision‑модель Qwen2.5‑VL**, чтобы вычистить такие случаи (в коде прямо написано, что это осознанно тяжёлый путь).

**Почему в Step 4 снова есть `reader` (WARC / `.zstd`).**  
Точно по той же причине, что и в Step 3: с диска читаются **jsonl‑карточки** из Step 3, а **байты PDF** нужно **снова подгрузить** в `media_bytes`, потому что дальше `rollout_postprocess` использует `document.media[0].media_bytes` (см. `pipeline_utils/postprocess_utils.py`). Без reader у карточки есть текст, но **нет исходного PDF** для визуального шага.

**Две ветки внутри Step 4 (как в коде).**  
1) Цикл по `truncated` / `non_truncated` — **только OCR‑результаты** (RolmOCR).  
2) Один отдельный пайплайн — **только Docling‑результаты** (оба truncation подряд в одном списке шагов).

---

#### 4.1 Общая подготовка: один раз создаётся `LanguageTagger`

Перед циклом по `truncated` / `non_truncated` в `run_postprocess()` создаётся **один** объект:

```python
language_tagger = LanguageTagger(language_threshold=0.01, label_only=True, backend="glotlid")
```

**Зачем создавать его «один раз», а не внутри каждого пайплайна.**  
`LanguageTagger` внутри конструктора поднимает **тяжёлый движок определения языка** (`GlotLID` из `datatrove.utils.lid`, с параметром `k=1000` — сколько гипотез языка держать при предсказании). Загрузка таких моделей **дорогая** по времени и памяти.  
Один и тот же экземпляр потом вставляется в:

- **два** пайплайна OCR (сначала `truncated`, потом `non_truncated`);
- **один** пайплайн Docling.

Итого **три запуска** `LocalPipelineExecutor`, но **модель GlotLID в памяти одна** — это обычная экономия ресурсов.

---

**Что такое `LanguageTagger` простыми словами.**  
Это отдельный шаг пайплайна (`PipelineStep` в терминах datatrove), который **не переписывает сам текст документа ради языка**, а **дописывает в `metadata` поля с языком и оценками уверенности**. Дальше эти поля читает `SelectBestLanguage` на этапе `run_language_filter` (уже с порогами из `th_values.json` и разбиением по `language_bucket`).

Реализация — в `postprocessing/language.py` (класс `LanguageTagger`).

---

**Как именно он считает язык (логика по документу).**

1) **Документ режется на страницы по `page_offsets`.**  
   В коде ожидается, что в `doc.media[0].metadata["page_offsets"]` лежит список смещений в **одной длинной строке** `doc.text` (кумулятивные длины страниц). Это совместимо с тем, как Docling (и OCR‑ветка после извлечения) кладут текст и метаданные.

2) **Для каждой страницы текст чуть «подчищают» перед LID:**  
   убирают строки, похожие на markdown‑таблицы (`| ... |`), выкидывают символы `| - * #`, схлопывают пробелы. Зачем: таблицы и разметка сильно путают детектор языка.

3) **Порог «есть ли вообще смысл определять язык».**  
   Если на странице мало «буквенного» текста или мало латиницы/кириллицы относительно мусора (`min_alpha_length_bytes` по умолчанию 50 байт UTF‑8 и `min_alpha_ratio` 0.2), страница помечается как `("unknown", 0)`.

4) **Два режима внутри одного прохода:**
   - **По страницам:** для каждой страницы вызывается `model.predict`, оценки языков **суммируются** и потом **делятся на число страниц** (усреднение). В metadata попадают, например, `best_page_languages`, `best_page_scores`, `best_page_average_language`, плюс поля вида `page_average_language_<код>_score` для языков выше `language_threshold`.
   - **По всему документу:** берётся склейка всех страниц, **обрезка до 40 000 символов**, снова проверка «достаточно ли букв», затем `predict` на целом куске. В metadata пишутся `language`, `language_score`, и при необходимости `top_language_<код>_score`.

---

**Параметры из вызова в `run_finepdfs_pipeline.py`**

| Параметр | Значение в коде | Смысл |
|----------|-----------------|--------|
| `language_threshold` | `0.01` | Очень **низкий** порог: в список «значимых» языковых оценок попадут даже слабые сигналы (по сравнению с дефолтом `0.65` в конструкторе класса). Для корпуса это «не отфильтровывать язык слишком агрессивно» на этом шаге. |
| `label_only` | `True` | В docstring задумывалось как режим «только метки»; в **текущей** версии `run()` в `language.py` флаг **сохраняется**, но **не переключает отдельную ветку** — фактически шаг всё равно **только дописывает metadata**, сам `doc.text` не заменяет на перевод. |
| `backend` | `"glotlid"` | Вместо старого `ft176` (FastText‑стиль) используется **GlotLID** — отдельная модель/стек для multilang LID, которую datatrove оборачивает в класс `GlotLID`. |

Зависимости шага (для установки окружения): в классе указано `_requires_dependencies` — в т.ч. `fasttext` / `fasteners` для работы LID‑бэкенда.

---

**Связь с «финальным» языком корпуса.**  
Здесь язык — это **богатый набор признаков в `metadata`** для последующих решений.  
Финальное «в какой языковой шард положить документ» делается **позже** в `run_language_filter()` через `SelectBestLanguage` и файл `thresholds/th_values.json`.

---

#### 4.2 OCR‑ветка (цикл `truncated` / `non_truncated`)

Для каждого значения `truncation` строится свой `pipeline_ocr` и вызывается `LocalPipelineExecutor`.

**Откуда читать список документов**

- `JsonlReader` с папкой `./finepdfs/data/ocr_docs_extracted` и шаблоном `{truncation}/extracted/*.jsonl.gz` — то есть отдельно **truncated** и **non_truncated** выходы Step 3.2.

**Откуда снова брать PDF**

- как в Step 3: `non_truncated` → `WarcReaderFast(s3://commoncrawl)`, `truncated` → `ZstdReader(PDF_SAVE_DIR)`.

**Что делает каждый шаг по порядку (очень по‑человечески)**

1. **`AddMetadata(is_docling=False, is_truncated=...)`** — в метаданные документа дописывается: «этот текст из **OCR**» (`is_docling=False`) и «он из **truncated** или **non_truncated** ветки». Дальше любой фильтр/аналитика может это различать.

2. **`DropFailedDocuments(EMPTY_PAGES_DOCLING_DIR)`** — выкидывает документы, где **нет нормального текста** или нет нужной структуры страниц (в реализации смотрит на `document.text` и `page_offsets` в metadata). Имя папки `EMPTY_PAGES_DOCLING_DIR` в коде общее с docling‑веткой, но **используется и здесь** как место, куда складываются «пустые/битые» OCR‑документы.

3. **`CoallesceFailedPages(FAILED_PAGES_OCR_DIR)`** (в коде опечатка *Coallesce*) — шаг для **страниц OCR, которые не удалось обработать**; складывает/объединяет информацию в `./finepdfs/data/postprocessed/ocr_failed`, чтобы не терять диагностику.

4. **`TagBoilerplateFormatter(is_ocr=True, drop=True)`** — пытается найти **повторяющийся «шум»** (колонтитулы, одинаковые блоки) и **`drop=True`** значит: найденное **удалить** из текста.

5. **`Normalize(is_from_docling=False)`** — приводит текст к **единому аккуратному виду** по правилам для **OCR‑текста** (переносы, пробелы, мусорные символы — зависит от реализации `Normalize`).

6. **`language_tagger`** — проставляет языковые признаки через GlotLID (см. выше).

7. **`TokensCounter(...)`** — считает **сколько токенов** в тексте выбранным токенизатором (`hynky/Llama-3.2-1B-no-bos`). Это удобно для статистики длины и дальнейших фильтров.

8. **`InferenceRunner` + `Qwen/Qwen2.5-VL-7B-Instruct` + `rollout_postprocess`** — самый тяжёлый кусок: **мультимодальная модель** смотрит на **страницы PDF** (через подготовку запросов в `rollout_postprocess`, которая снова использует `media_bytes`) и правит текст, чтобы **убрать галлюцинации с пустых страниц**.  
   Параметры в коде: `temperature=0.0`, `max_concurrent_generations=50`, `server_type="vllm"`.

**Куда пишется результат OCR‑ветки**

- `JsonlWriter(output_folder=SAVE_OCR_DIR.format(prefix="extracted"))`  
  Константа `SAVE_OCR_DIR` — это `./finepdfs/data/postprocessed/output_ocr` (в строке **нет** `{prefix}`, вызов `.format` её не меняет).  
  Оба прохода цикла (`truncated` и `non_truncated`) пишут **в одну и ту же выходную папку** — файлы накапливаются там (как несколько партий обработки).

---

#### 4.3 Docling‑ветка (один пайплайн без `reader`)

Здесь **нет** `WarcReaderFast` / `ZstdReader`, потому что для этой цепочки **не вызывается** vision‑шаг на полном PDF в конце — только работа с уже извлечённым текстом и docling‑метаданными.

**Как читаются входы**

- сначала `JsonlReader` на `./finepdfs/data/non_ocr_docs_extracted/non_truncated/extracted/*.jsonl.gz`;
- сразу после него **`AddMetadata(is_docling=True, is_truncated=False)`**;
- затем второй `JsonlReader` на `.../truncated/extracted/*.jsonl.gz`;
- затем **`AddMetadata(is_docling=True, is_truncated=True)`**.

**Простыми словами:** два читателя подряд **склеивают в один поток** сначала все **non_truncated** docling‑документы, потом все **truncated**, и у каждой группы своя метка `is_truncated`.

**Что дальше по шагам**

1. **`RemoveDoclingMetadata()`** — убирает/сжимает **лишние docling‑специфичные поля** в metadata, чтобы не тащить огромные служебные структуры дальше (детали — в реализации шага).

2. **`DropFailedDocuments(EMPTY_PAGES_DOCLING_DIR)`** — то же правило «пустой/битый документ» для docling‑ветки.

3. **`PostprocessPageNumbers()`** — приводит к порядку **номера страниц** в тексте/разметке (после извлечения они часто «плавают»).

4. **`CleanTables()`** — чистит **таблицы** от артефактов извлечения.

5. **`RemoveImageAnnotationsByRatio(ratio_threshold=0.8)`** — если на странице **слишком много** «аннотаций картинок» (по доле), их режут/убирают — чтобы не тянуть в текст мусор от картинок.

6. **`TagBoilerplateFormatter(is_ocr=False, drop=True)`** — boilerplate для **не‑OCR** текста (другие эвристики, чем у OCR).

7. **`Normalize(is_from_docling=True)`** — нормализация под **docling‑текст**.

8. **`language_tagger`** и **`TokensCounter`** — то же, что в OCR‑ветке.

9. **`JsonlWriter(output_folder=SAVE_DOCLING_DIR)`** — итог docling‑ветки в `./finepdfs/data/postprocessed/output_docling`.

---

**Сводка: откуда → куда**

| Ветка | Вход (после Step 3) | Reader PDF? | Выход Step 4 | «Корзины» для проблем |
|-------|---------------------|---------------|----------------|------------------------|
| OCR | `./finepdfs/data/ocr_docs_extracted/{truncation}/extracted/*.jsonl.gz` | да (WARC или `.zstd`) | `./finepdfs/data/postprocessed/output_ocr` | `ocr_failed`, `empty` |
| Docling | `non_ocr_docs_extracted/non_truncated/...` затем `.../truncated/...` | нет | `./finepdfs/data/postprocessed/output_docling` | `empty` |

**Примеры ключевых строк в коде:**

```python
AddMetadata(is_docling=False, is_truncated=truncation == "truncated"),
DropFailedDocuments(EMPTY_PAGES_DOCLING_DIR),
CoallesceFailedPages(FAILED_PAGES_OCR_DIR),
```

```python
InferenceRunner(
    rollout_fn=rollout_postprocess,
    config=InferenceConfig(
        model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
        default_generation_params={"temperature": 0.0},
        max_concurrent_generations=50,
        server_type="vllm",
    ),
    output_writer=JsonlWriter(output_folder=SAVE_OCR_DIR.format(prefix=f"extracted")),
)
```

```python
JsonlReader(data_folder=DOCLING_INPUT_DIR, glob_pattern="non_truncated/extracted/*.jsonl.gz"),
AddMetadata(is_docling=True, is_truncated=False),
JsonlReader(data_folder=DOCLING_INPUT_DIR, glob_pattern="truncated/extracted/*.jsonl.gz"),
AddMetadata(is_docling=True, is_truncated=True),
```

---

### (Ненумерованный блок в коде) Language filter (`run_language_filter`)

- **Название на русском**: **(Между шагами) Языковая маршрутизация: выбор лучшего языка и шардирование по `language_bucket`**
- **Что делает код (цель, подробно)**: этот блок выполняет критически важную операцию “перевода” языковых вероятностей/меток в **один дискретный ярлык языка** (`language_bucket`), по которому дальше строится вся пер‑языковая обработка (exact‑dedup по тексту, классификация, minhash). Он:
  1) **Считывает объединённый поток**: два `JsonlReader` подряд читают `SAVE_DOCLING_DIR` и `SAVE_OCR_DIR`. В терминах datatrove это означает “слить два источника в один последовательный поток документов”.
  2) **Загружает пороги** из `TH_VALUES_FILE` и приводит их к безопасному минимуму (`max(v, 0.05)`), чтобы не было экстремально низких порогов.
  3) **Фиксирует исключения для `zxx_*`**: выставляет порог `-1`, тем самым разрешая этим bucket’ам проходить при top‑1 (комментарий в коде: “zxx is never rerouted”).
  4) **Применяет `SelectBestLanguage`** — step, который выбирает язык с учётом порогов и пишет его в `doc.metadata["language_bucket"]`.
  5) **Опционально ограничивает набор языков** (если передан `--languages`) через `LambdaFilter`.
  6) **Пишет результат в пер‑языковые шард‑папки**: `output_filename="${language_bucket}/${rank}.jsonl.gz"`.
- **Какие функции/классы используются**: `SelectBestLanguage`, `JsonlReader`, `LambdaFilter` (опционально, если `--languages`), `JsonlWriter`.
- **Входные данные**:
  - `SAVE_DOCLING_DIR/*.jsonl.gz`
  - `SAVE_OCR_DIR/*.jsonl.gz`
  - `./thresholds/th_values.json`
- **Какие параметры настраиваются**:
  - пороги по языкам (минимум 0.05)
  - исключения: `zxx_*` пропускаются всегда как top‑1 (порог = -1)
  - whitelist языков через `--languages`
- **Какие выходные данные производит**:
  - `PER_LANGUAGE_DIR_EXACT/${language_bucket}/${rank}.jsonl.gz`
- **Примеры ключевых строк кода с объяснением**:

```python
pipeline = [
    JsonlReader(data_folder=SAVE_DOCLING_DIR, glob_pattern="*.jsonl.gz"),
    JsonlReader(data_folder=SAVE_OCR_DIR, glob_pattern="*.jsonl.gz"),
    SelectBestLanguage(language_thresholds_dict=language_thresholds_dict),
    JsonlWriter(output_folder=PER_LANGUAGE_DIR_EXACT, output_filename="${language_bucket}/${rank}.jsonl.gz"),
]
```

- **Возможные ошибки и как их обрабатывать**:
  - **Нет `TH_VALUES_FILE`**: добавить файл `thresholds/th_values.json` или изменить путь.

---

### Step 5: `exact_dedup` (`run_exact_dedup`)

- **Название на русском**: **Шаг 5 — Точная дедупликация по тексту внутри языка + подготовка чанков**
- **Что делает код (цель, подробно)**: после Step 4 у FinePDFs уже есть “человеко‑читаемый текст”, и теперь задача — убрать точные копии текста (например, одинаковые документы, которые пришли из разных URL или версий). Важная деталь: exact‑dedup выполняется **по нормализованному тексту без пробелов**, т.е. небольшие различия в whitespace не мешают определить дубликаты. После этого шаг готовит данные для моделей качества:
  1) `ExactDedupSignature/FindDedups/Filter` строят и применяют список дубликатов **внутри одного language‑шарда**.
  2) `AddTextChunks(...)` добавляет в `metadata["chunks"]` батч‑пригодные фрагменты текста для последующего inference, а затем документы пишутся в `OUTPUT_DIR_EXACT/output/<language>`.
- **Какие функции/классы используются**:
  - `ExactDedupSignature`, `ExactFindDedups`, `ExactDedupFilter`
  - `AddTextChunks`
  - `create_content_getter_exact()` (удаляет все пробелы)
- **Какие входные данные принимает**: `PER_LANGUAGE_DIR_EXACT/<language>/*.jsonl.gz`
- **Какие параметры настраиваются**:
  - `tasks` (по умолчанию 100)
  - `finder_workers` зависит от `tasks` (`worker_tasks = max(tasks//2, 1)`)
  - токенайзер чанков: ModernBERT для `eng_Latn`, иначе mmBERT
- **Какие выходные данные производит**:
  - `OUTPUT_DIR_EXACT/output/<language>/*.jsonl.gz`
  - `OUTPUT_DIR_EXACT/removed/<language>/*.jsonl.gz`
- **Примеры ключевых строк кода с объяснением**:

```python
def content_getter(doc: Document) -> str:
    return remove_spaces_regex.sub("", doc.text)
```

```python
AddTextChunks(tokenizer_name="answerdotai/ModernBERT-large" if language == "eng_Latn" else "mmbert-colab/mmBERT-base")
```

- **Возможные ошибки и как их обрабатывать**:
  - **Слишком много языков**: можно ограничить `--languages`, чтобы не гонять весь набор.

---

### Step 6: `model_classification` (`run_model_classification`)

- **Название на русском**: **Шаг 6 — Пер‑языковая классификация качества (EDU/DCLM)**
- **Что делает код (цель, подробно)**: этот шаг превращает “просто текст” в “текст + оценки качества”, которые дальше используются как фильтры (особенно для английского) и как сигналы для downstream. Важные особенности реализации:
  1) **Модели выбираются динамически**: для каждого языка формируются repo id `HuggingFaceFW/finepdfs_edu_classifier_{language}` и `..._dclm_classifier_{language}` и проверяются через `model_exists()`. Если моделей нет (или нет доступа), шаг делает passthrough (переписывает JSONL).
  2) **Классификация идёт по чанкам**: `make_rollout_model()` создаёт rollout, который берёт `document.metadata["chunks"]`, отправляет каждый chunk как отдельный запрос и собирает серию оценок (список float/None) в `fw_edu_scores` и/или `dclm_scores`.
  3) **Инференс выполняется через кастомный сервер** (`InferenceConfig(server_type="custom")`) и `server_script=blocks/classification/tf_batching.py`. Это позволяет эффективный batching (см. `batch-size=256`) и большую конкуррентность (`max_concurrent_generations=1024`).
  4) После inference **чанки удаляются** (`document.metadata.pop("chunks", None)`), чтобы не раздувать финальный размер документа.
- **Какие функции/классы используются**:
  - `model_exists()` (через `huggingface_hub.HfApi`)
  - `make_rollout_model(output_fields_in_order)`
  - `InferenceRunner` + `InferenceConfig(server_type="custom")`
- **Какие входные данные принимает**: `INPUT_DIR_MODEL/<language>/*.jsonl.gz` (это `OUTPUT_DIR_EXACT/output/<language>`)
- **Какие параметры настраиваются**:
  - `gpus` влияет на `dp`
  - `model_kwargs` для кастомного сервера (`blocks/classification/tf_batching.py`, batch size, max context и т.д.)
- **Какие выходные данные производит**: `OUTPUT_DIR_MODEL/<language>/*.jsonl.gz`
- **Примеры ключевых строк кода с объяснением**:

```python
edu_model = f"HuggingFaceFW/finepdfs_edu_classifier_{language}"
dclm_model = f"HuggingFaceFW/finepdfs_dclm_classifier_{language}"
```

```python
document.metadata[field] = series
document.metadata.pop("chunks", None)
```

- **Возможные ошибки и как их обрабатывать**:
  - **Сеть/HF API**: если `model_exists()` не может достучаться, классификация “выключится” (код сделает passthrough). Для production лучше логировать причину (в этом скрипте нет).
  - **Непарсящийся ответ**: rollout ставит `None` значения по chunk’ам.

---

### Step 7: `minhash` (`run_minhash`)

- **Название на русском**: **Шаг 7 — Приближённая дедупликация (near‑dedup) MinHash внутри языка**
- **Что делает код (цель, подробно)**: этот шаг удаляет “почти одинаковые” документы (копии с мелкими изменениями, зеркала, репосты, версии с небольшими правками), которые exact‑dedup не поймает. MinHash реализован стандартным конвейером:
  1) `MinhashDedupSignature`: превращает текст в набор хеш‑подписей по \(n\)-граммам (`n_grams=5`) и распределяет их по bucket’ам (`num_buckets=32`).
  2) `MinhashDedupBuckets`: группирует сигнатуры по bucket’ам, подготавливая данные к кластеризации (буферизация `lines_to_buffer=20000` — компромисс между скоростью и памятью).
  3) `MinhashDedupCluster`: строит кластеры near‑duplicates и сохраняет `cluster_id`/`cluster_size`.
  4) `MinhashDedupFilter`: исключает документы, которые попали в “не‑выжившие” позиции внутри кластеров, и пишет оставшиеся в `output/`.
  5) Для `eng_Latn` добавлен **качество‑гейт**: до построения сигнатур применяется `LambdaFilter`, который оставляет документы с `max(fw_edu_scores) >= 0.5`. Это уменьшает объём minhash‑работы и одновременно повышает качество финального англ. поднабора.
- **Какие функции/классы используются**:
  - `MinhashDedupSignature`, `MinhashDedupBuckets`, `MinhashDedupCluster`, `MinhashDedupFilter`
  - `MINHASH_CONFIG` и `HashConfig(xxhash, 64)`
  - `tokenizers_map` для выбора tokenizer language
- **Какие входные данные принимает**: `PER_LANGUAGE_DIR_MINHASH/<language>/*.jsonl.gz` (это `OUTPUT_DIR_MODEL/<language>`)
- **Какие параметры настраиваются**:
  - `MINHASH_CONFIG` (buckets, ngrams, hashes)
  - `lines_to_buffer=20000` в `MinhashDedupBuckets`
  - вычисление `WORKERS` (кратность `num_buckets`)
  - prefilter для `eng_Latn`: `fw_edu_scores >= 0.5`
- **Какие выходные данные производит**:
  - `OUTPUT_DIR_MINHASH/<language>/output/*.jsonl.gz`
  - `.../removed/` + промежуточные `signatures/`, `buckets/`, `clusters/`
- **Примеры ключевых строк кода с объяснением**:

```python
MinhashDedupSignature(..., config=MINHASH_CONFIG, language=tokenizer_name)
MinhashDedupBuckets(..., lines_to_buffer=20000)
MinhashDedupCluster(..., save_cluster_id=True, save_cluster_size=True)
MinhashDedupFilter(..., load_cluster_ids=True, load_cluster_sizes=True)
```

- **Возможные ошибки и как их обрабатывать**:
  - **Очень большие файлы/память**: уменьшать `lines_to_buffer`, снижать параллелизм.

---

### Push: `push_to_hub` (`run_push_to_hub`)

- **Название на русском**: **Публикация — Выгрузка датасетов на HuggingFace Hub**
- **Что делает код (цель, подробно)**: финальный шаг превращает локальные JSONL‑шарды в **публичные parquet‑шарды датасета** на HuggingFace Hub. В терминах MLOps это “материализация датасета”:
  1) Читает финальные документы из `PUSH_INPUT_DIR/<lang>/output`.
  2) Запускает `HuggingFaceDatasetWriter`, который:
     - применяет `adapter=push_adapter`, приводя `Document` к структуре строк parquet (какие поля и как сериализуются — в `pipeline_utils/push_utils.py` и связанных модулях)
     - пишет на диск временную структуру в `local_working_dir`
     - выгружает в dataset repo в путь `data/{lang}/train/${rank}.parquet`
  3) Затем применяет `LambdaFilter` по порогу EDU‑качества и делает второй writer в отдельный dataset `finepdfs_fw_edu_subset`.
  4) Таким образом формируется:
     - “широкий” subset (максимальное покрытие)
     - “строгий” subset (фильтрованный по EDU‑сигналу)
  - `HuggingFaceFW/finepdfs_subset`
  - `HuggingFaceFW/finepdfs_fw_edu_subset` (фильтр по `fw_edu_scores >= 0.5`)
- **Какие функции/классы используются**: `HuggingFaceDatasetWriter`, `LambdaFilter`, `push_adapter`.
- **Какие входные данные принимает**: `PUSH_INPUT_DIR/<lang>/output/*.jsonl.gz`
- **Какие параметры настраиваются**:
  - `dataset=...`
  - `output_filename=f"data/{lang}/train/${{rank}}.parquet"`
  - `local_working_dir=...`
- **Какие выходные данные производит**: parquet‑шарды в репозиториях датасетов на Hub.
- **Примеры ключевых строк кода с объяснением**:

```python
HuggingFaceDatasetWriter(dataset="HuggingFaceFW/finepdfs_subset", adapter=push_adapter, ...)
LambdaFilter(lambda x: max(x.metadata.get("fw_edu_scores", [0])) >= 0.5),
HuggingFaceDatasetWriter(dataset="HuggingFaceFW/finepdfs_fw_edu_subset", adapter=push_adapter, ...)
```

- **Возможные ошибки и как их обрабатывать**:
  - **Нет `HF_TOKEN`/прав**: авторизоваться и убедиться, что есть доступ на запись в dataset repo.
  - **Нет `fw_edu_scores`**: фильтр использует дефолт `[0]`, поэтому такие документы в EDU‑subset не попадут (что обычно ожидаемо).

---

## Практический “как запустить”

Минимальный запуск (пример):

```bash
python run_finepdfs_pipeline.py --crawl-ids CC-MAIN-2023-06
```

Запуск с ограничением языков (после постпроцессинга будут выбраны только эти `language_bucket`):

```bash
python run_finepdfs_pipeline.py --crawl-ids CC-MAIN-2023-06 --languages eng_Latn,rus_Cyrl --gpus 1
```

---

## Чек-лист диагностики (быстро найти где сломалось)

- **Не качаются PDF (этап 2–3)**: проверь `failed_pdf_fetch/`, таймауты и доступ к `s3://commoncrawl`.
- **Падает OCR классификатор (этап 5)**: проверь наличие `./models/xgb_ocr_classifier/xgb_classifier.ubj` и `failed_ocr/`.
- **Падает Docling (этап 4)**: смотри `OUTPUT_NON_OCR_DIR/.../failed/`.
- **Не стартует vLLM (этап 6/10)**: проверь GPU/драйверы/VRAM, снизь `max_concurrent_generations`.
- **Нет языкового шардинга (этап 7)**: проверь `thresholds/th_values.json` и наличие `language_bucket`.
- **Нет моделей EDU/DCLM (этап 8)**: `model_exists()` может возвращать `False` из-за сети/прав; убедись, что HF доступен.
- **MinHash работает слишком долго (этап 9)**: проверь `WORKERS`, размер данных, `lines_to_buffer`.
- **Не пушится на Hub (этап 10)**: проверь `HF_TOKEN` и права на `dataset=...`.

