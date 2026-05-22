import asyncio
import argparse
import glob
import os
import sys
from typing import Any, AsyncGenerator, Optional
from pipeline_utils.language import SelectBestLanguage

# --- Shared third-party imports used across steps ---
from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.io import get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.media.filters.mime_filter import MimeTypeFilter
from datatrove.pipeline.media.media_readers.warc import WarcReaderFast
from datatrove.pipeline.media.media_readers.zstd import ZstdReader
from datatrove.pipeline.media.readers.http_fetch import HTTPFetchReader
from datatrove.pipeline.media.media_writers.zstd import ZstdWriter
from datatrove.pipeline.readers.jsonl import JsonlReader
from datatrove.pipeline.readers.parquet import ParquetReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.pipeline.tokens.counter import TokensCounter
from datatrove.pipeline.inference.run_inference import (
    InferenceConfig,
    InferenceRunner,
    InferenceResult,
)
from datatrove.pipeline.dedup.exact_dedup import (
    ExactDedupFilter,
    ExactDedupSignature,
    ExactFindDedups,
    ExactDedupConfig,
)
from datatrove.pipeline.dedup import (
    MinhashDedupBuckets,
    MinhashDedupFilter,
    MinhashDedupSignature,
    MinhashConfig,
    MinhashDedupCluster,
)
# Docling branch postprocessing
from postprocessing.page_numbers import PostprocessPageNumbers
from postprocessing.boilerplate import TagBoilerplateFormatter
from postprocessing.language import LanguageTagger
from postprocessing.tables import CleanTables
from postprocessing.normalize import Normalize
from postprocessing.remove_image_annots import RemoveImageAnnotationsByRatio
from datatrove.utils.hashing import HashConfig
# --- Project-specific imports ---
from blocks.extractors.opendataloader import OpenDataLoaderExtractor, java_available
from blocks.predictor.ocr_predictor import PDFScannedPredictor
from blocks.readers.warc_reparse import WarcIndexReprocess, _transient_cc_read_error
from blocks.utils import MIME_TYPES, index_adapter, filter_non_pdf, filter_non_truncated
from classification.label_utils import AddTextChunks

# Utilities split into separate modules to keep this file short
from pipeline_utils.extract_utils import rollout_extract
from pipeline_utils.postprocess_utils import (
    AddMetadata,
    RemoveDoclingMetadata,
    DropFailedDocuments,
    CoallesceFailedPages,
    rollout_postprocess,
)

# =====================================================================================
# Constants aggregated from original scripts
# =====================================================================================

CC_INDEX_INPUT_TEMPLATE = "s3://commoncrawl/cc-index/table/cc-main/warc/crawl={crawl_id}/subset=warc"
CC_PATHS_TEMPLATE = "crawl-data/{crawl_id}/warc.paths.gz"
# Local runs: HTTPS (no AWS). On AWS EC2 set FINEPDFS_CC_USE_S3=1 for S3 + parquet index.
CC_HTTP_BASE = "https://data.commoncrawl.org"
CC_S3_STORAGE_OPTIONS = {"anon": True, "client_kwargs": {"region_name": "us-east-1"}}


def _use_cc_s3() -> bool:
    return os.environ.get("FINEPDFS_CC_USE_S3", "0") == "1"


def _cc_s3(path: str) -> tuple[str, dict]:
    return (path, CC_S3_STORAGE_OPTIONS)


def cc_data_root() -> str | tuple[str, dict]:
    if _use_cc_s3():
        return _cc_s3("s3://commoncrawl")
    return CC_HTTP_BASE


def cc_paths_file(crawl_id: str) -> str | tuple[str, dict]:
    rel = CC_PATHS_TEMPLATE.format(crawl_id=crawl_id)
    if _use_cc_s3():
        return _cc_s3(f"s3://commoncrawl/{rel}")
    return f"{CC_HTTP_BASE}/{rel}"


CC_PATHS_CACHE_DIR = "./finepdfs/data/cc_cache"


def ensure_cc_warc_paths_cached(crawl_id: str, max_retries: int = 6, base_delay: float = 5.0) -> str:
    """Download warc.paths.gz with retries; reuse local cache when Common Crawl returns 503."""
    import time
    import urllib.error
    import urllib.request

    cache_dir = os.path.join(CC_PATHS_CACHE_DIR, crawl_id)
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, "warc.paths.gz")
    if os.path.isfile(local_path) and os.path.getsize(local_path) > 100:
        with open(local_path, "rb") as f:
            if f.read(2) == b"\x1f\x8b":
                return local_path

    url = f"{CC_HTTP_BASE}/{CC_PATHS_TEMPLATE.format(crawl_id=crawl_id)}"
    last_err: BaseException | None = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "finepdfs-pipeline/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if len(data) < 100 or data[:2] != b"\x1f\x8b":
                raise OSError(f"Invalid warc.paths.gz from {url} ({len(data)} bytes)")
            with open(local_path, "wb") as f:
                f.write(data)
            return local_path
        except Exception as exc:
            last_err = exc
            if attempt + 1 >= max_retries:
                break
            delay = base_delay * (2**attempt)
            print(
                f"Common Crawl warc.paths.gz unavailable ({exc}); retry in {delay:.0f}s...",
                file=sys.stderr,
            )
            time.sleep(delay)
    assert last_err is not None
    raise last_err
SPLIT_TRUNCATION_DIR = "./finepdfs/data/split_truncation/{prefix}"
PDF_SAVE_DIR = "./finepdfs/data/pdf"

BYTE_CONTENT_DEDUPLICATION_DIR_OUTPUT = "./finepdfs/data/byte_content_deduplication/{prefix}"
OCR_CLASSIFICATION_DIR_OUTPUT = "./finepdfs/data/ocr_classification/{prefix}"
DEDUP_OUTPUT_DIR = "./finepdfs/data/content_dedup/{prefix}"
PDF_SCANNED_MODEL_PATH = "./models/xgb_ocr_classifier/xgb_classifier.ubj"
HERON_MODEL_XML = "./models/heron/heron_int8_quant.xml"
HERON_MODEL_BIN = "./models/heron/heron_int8_quant.bin"

INPUT_DIR_EXTRACT = "./finepdfs/data/content_dedup/{prefix}"
OUTPUT_OCR_DIR = "./finepdfs/data/ocr_docs_extracted/{prefix}"
OUTPUT_NON_OCR_DIR = "./finepdfs/data/non_ocr_docs_extracted/{prefix}"

DOCLING_INPUT_DIR = "./finepdfs/data/non_ocr_docs_extracted"
OCR_INPUT_DIR = "./finepdfs/data/ocr_docs_extracted"
SAVE_DOCLING_DIR = "./finepdfs/data/postprocessed/output_docling"
SAVE_OCR_DIR = "./finepdfs/data/postprocessed/output_ocr"
FAILED_PAGES_OCR_DIR = "./finepdfs/data/postprocessed/ocr_failed"
EMPTY_PAGES_DOCLING_DIR = "./finepdfs/data/postprocessed/empty"

TH_VALUES_FILE = "./thresholds/th_values.json"
PER_LANGUAGE_DIR_EXACT = "./finepdfs/data/glotlid/per_language"
OUTPUT_DIR_EXACT = "./finepdfs/data/exact_dedup/per_language"

INPUT_DIR_MODEL = "./finepdfs/data/exact_dedup/per_language/output"
OUTPUT_DIR_MODEL = "./finepdfs/data/model_labeling/per_language"

PER_LANGUAGE_DIR_MINHASH = "./finepdfs/data/model_labeling/per_language"
OUTPUT_DIR_MINHASH = "./finepdfs/data/minhash/per_language"

# Step 1: WARC records (responses) to scan per crawl — not PDF count.
LIMIT = 5000
# Must match tasks= on ExactFindDedups executor (step 2.2); use 1 on a weak PC.
EXACT_CONTENT_DEDUP_FINDER_WORKERS = 1
# Must match xgb_classifier.ubj training (8 pages). Lower values cause feature_names mismatch.
OCR_PREDICTOR_NUM_PAGES_TO_SAMPLE = 8


def _split_truncation_has_inputs(prefix: str) -> bool:
    folder = SPLIT_TRUNCATION_DIR.format(prefix=prefix)
    return bool(glob.glob(os.path.join(folder, "**", "*.jsonl.gz"), recursive=True))


def _step1_produced_any() -> bool:
    return any(_split_truncation_has_inputs(p) for p in ("truncated", "non_truncated", "failed_pdf_fetch"))


def _is_git_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(64).startswith(b"version https://git-lfs")
    except OSError:
        return False


def _opendataloader_ready() -> bool:
    return java_available()


def _skip_gpu_steps(gpus: int) -> bool:
    if os.environ.get("FINEPDFS_SKIP_GPU_STEPS", "").lower() in ("1", "true", "yes"):
        return True
    if gpus <= 0:
        return True
    try:
        import importlib.util

        return importlib.util.find_spec("vllm") is None
    except Exception:
        return True


def _has_extracted_outputs() -> bool:
    globs = (
        os.path.join(OUTPUT_NON_OCR_DIR.format(prefix="**"), "extracted", "*.jsonl.gz"),
        os.path.join(OUTPUT_OCR_DIR.format(prefix="**"), "extracted", "*.jsonl.gz"),
    )
    return any(glob.glob(p, recursive=True) for p in globs)


def _has_postprocessed_docling() -> bool:
    return bool(glob.glob(os.path.join(SAVE_DOCLING_DIR, "*.jsonl.gz")))


def _has_postprocessed_ocr() -> bool:
    return bool(glob.glob(os.path.join(SAVE_OCR_DIR, "**", "*.jsonl.gz"), recursive=True))


def _language_shards_ready(languages: Optional[list[str]] = None) -> bool:
    if languages:
        return any(
            glob.glob(os.path.join(PER_LANGUAGE_DIR_EXACT, lang, "*.jsonl.gz"))
            for lang in languages
        )
    return bool(
        glob.glob(os.path.join(PER_LANGUAGE_DIR_EXACT, "**", "*.jsonl.gz"), recursive=True)
    )


def _patch_datatrove_jsonl_utf8() -> None:
    """JsonlReader opens text files without encoding; on Windows that is cp1252, not UTF-8."""
    if getattr(JsonlReader, "_finepdfs_utf8", False):
        return
    import base64

    import orjson
    from orjson import JSONDecodeError

    from datatrove.utils.logging import logger

    def read_file(self, filepath: str):
        with self.data_folder.open(
            filepath, "r", compression=self.compression, encoding="utf-8"
        ) as f:
            try:
                for li, line in enumerate(f):
                    with self.track_time():
                        try:
                            line = orjson.loads(line)
                            for media in line.get("media", []):
                                if media["media_bytes"] is not None:
                                    media["media_bytes"] = base64.decodebytes(
                                        media["media_bytes"].encode("ascii")
                                    )
                            document = self.get_document_from_dict(line, filepath, li)
                            if not document:
                                continue
                        except (EOFError, JSONDecodeError) as e:
                            logger.warning(f"Error when reading `{filepath}`: {e}")
                            continue
                    yield document
            except UnicodeDecodeError as e:
                logger.warning(
                    f"File `{filepath}` may be corrupted: raised UnicodeDecodeError ({e})"
                )

    JsonlReader.read_file = read_file
    JsonlReader._finepdfs_utf8 = True


def _content_dedup_has_inputs(truncation: str, branch: str) -> bool:
    folder = DEDUP_OUTPUT_DIR.format(prefix=f"{truncation}/{branch}")
    return bool(glob.glob(os.path.join(folder, "**", "*.jsonl.gz"), recursive=True))


def _any_content_dedup_non_ocr() -> bool:
    return any(_content_dedup_has_inputs(t, "non_ocr") for t in ("truncated", "non_truncated"))


def _tag_non_ocr_metadata(doc: Document) -> bool:
    meta = doc.media[0].metadata or {}
    meta["ocr_prob"] = 0.0
    meta["garbled_text_ratio"] = 0.0
    doc.media[0].metadata = meta
    return True


def _ocr_classifier_step(
    path_to_model: str,
    exclusion_writer: JsonlWriter,
    num_pages_to_sample: int,
    timeout: int,
) -> PipelineStep:
    if _is_git_lfs_pointer(path_to_model):
        print(
            "XGBoost OCR model not downloaded (Git LFS pointer only). "
            "Run: git lfs pull --include=models/xgb_ocr_classifier/xgb_classifier.ubj\n"
            "For now, treating all PDFs as non-OCR (digital).",
            file=sys.stderr,
        )
        return LambdaFilter(_tag_non_ocr_metadata)
    return PDFScannedPredictor(
        path_to_model=path_to_model,
        exclusion_writer=exclusion_writer,
        exclude_failed=True,
        num_pages_to_sample=num_pages_to_sample,
        timeout=timeout,
    )


# =====================================================================================
# Step 1: filter_pdfs_and_refetch
# =====================================================================================

def run_filter_pdfs_and_refetch(crawl_ids: list[str]):
    for crawl_id in crawl_ids:
        # Parquet cc-index on S3 only (AWS). Outside AWS, Common Crawl serves data over HTTPS.
        if _use_cc_s3() and crawl_id >= "CC-MAIN-2019-47":
            index_reader = ParquetReader(
                data_folder=_cc_s3(CC_INDEX_INPUT_TEMPLATE.format(crawl_id=crawl_id)),
                glob_pattern="*.parquet",
                doc_progress=True,
                adapter=index_adapter,
                limit=LIMIT,
            )
        else:
            paths_file = ensure_cc_warc_paths_cached(crawl_id)
            index_reader = WarcIndexReprocess(
                data_folder=cc_data_root(),
                limit=LIMIT,
                paths_file=paths_file,
            )

        pipeline = [
            index_reader,
            LambdaFilter(filter_non_pdf),
            LambdaFilter(
                filter_non_truncated,
                exclusion_writer=JsonlWriter(
                    output_folder=SPLIT_TRUNCATION_DIR.format(prefix="non_truncated"),
                ),
            ),
            # For production purposes, it's not wise to to run this in one pipeline for good resource allocation.
            # We recommend separting the httpfetch reader to a separate pipeline to maximize resource utilization.
            HTTPFetchReader(workers=1, max_retries=5, retry_delay=3, timeout=(20, 20), download_timeout=90),
            MimeTypeFilter(mime_types=MIME_TYPES["pdf"]),
            ZstdWriter(
                max_file_size=128 * 1024 * 1024,
                output_folder=PDF_SAVE_DIR,
                output_filename=f"{crawl_id.replace('-', '_')}_${{rank}}.zstd",
            ),
            LambdaFilter(
                lambda x: x.media[0].media_bytes is not None,
                exclusion_writer=JsonlWriter(
                    output_folder=SPLIT_TRUNCATION_DIR.format(prefix="failed_pdf_fetch"),
                ),
            ),
            JsonlWriter(
                output_folder=SPLIT_TRUNCATION_DIR.format(prefix="truncated"),
            ),
        ]

        try:
            LocalPipelineExecutor(pipeline).run()
        except Exception as exc:
            if _transient_cc_read_error(exc) or "warc.paths" in str(exc).lower():
                print(f"Skipping crawl {crawl_id} after CC network error: {exc}", file=sys.stderr)
                continue
            raise


# =====================================================================================
# Step 2: content_dedup_ocr_organize
# =====================================================================================

def _get_media_bytes(doc: Document) -> bytes:
    return doc.media[0].media_bytes if doc.media[0].media_bytes else b""


CONTENT_DEDUP_CONFIG = ExactDedupConfig(content_getter=_get_media_bytes)


def _filter_ocr(x: Document):
    meta = x.media[0].metadata or {}
    # See the training notebook why we decided for this threshold
    return not (
        meta.get("ocr_prob", 0) >= 0.2 or meta.get("garbled_text_ratio", 0) > 0.0
    )


def run_content_dedup_ocr_organize():
    for truncation in ["truncated", "non_truncated"]:
        if not _split_truncation_has_inputs(truncation):
            print(
                f"Skipping step 2 ({truncation}): no files in "
                f"{SPLIT_TRUNCATION_DIR.format(prefix=truncation)}",
                file=sys.stderr,
            )
            continue
        if truncation == "non_truncated":
            reader = WarcReaderFast(
                data_folder=cc_data_root(),
                preserve_order=True,
                workers=1,
            )
        else:
            reader = ZstdReader(
                data_folder=PDF_SAVE_DIR,
                workers=1,
                preserve_order=True,
            )

        # 2.1 Signatures
        pipeline1 = [
            JsonlReader(
                data_folder=SPLIT_TRUNCATION_DIR.format(prefix=truncation),
                glob_pattern="**/*.jsonl.gz",
                doc_progress=True,
            ),
            reader,
            ExactDedupSignature(
                config=CONTENT_DEDUP_CONFIG,
                output_folder=DEDUP_OUTPUT_DIR.format(prefix=f"{truncation}/sigs"),
                finder_workers=EXACT_CONTENT_DEDUP_FINDER_WORKERS,
            ),
        ]

        # 2.2 Find duplicates
        pipeline2 = [
            ExactFindDedups(
                config=CONTENT_DEDUP_CONFIG,
                data_folder=DEDUP_OUTPUT_DIR.format(prefix=f"{truncation}/sigs"),
                output_folder=DEDUP_OUTPUT_DIR.format(prefix=f"{truncation}/dups"),
            )
        ]

        # 2.3 Filter dups, OCR predicate, split
        pipeline3 = [
            JsonlReader(
                data_folder=SPLIT_TRUNCATION_DIR.format(prefix=truncation),
                glob_pattern="**/*.jsonl.gz",
                doc_progress=True,
            ),
            ExactDedupFilter(
                config=CONTENT_DEDUP_CONFIG,
                data_folder=DEDUP_OUTPUT_DIR.format(prefix=f"{truncation}/dups"),
                exclusion_writer=JsonlWriter(
                    output_folder=DEDUP_OUTPUT_DIR.format(prefix=f"{truncation}/removed")
                ),
            ),
            reader,
            _ocr_classifier_step(
                PDF_SCANNED_MODEL_PATH,
                exclusion_writer=JsonlWriter(
                    output_folder=DEDUP_OUTPUT_DIR.format(prefix=f"{truncation}/failed_ocr")
                ),
                num_pages_to_sample=OCR_PREDICTOR_NUM_PAGES_TO_SAMPLE,
                timeout=20,
            ),
            LambdaFilter(_filter_ocr, exclusion_writer=JsonlWriter(output_folder=DEDUP_OUTPUT_DIR.format(prefix=f"{truncation}/ocr"))),
            JsonlWriter(output_folder=DEDUP_OUTPUT_DIR.format(prefix=f"{truncation}/non_ocr")),
        ]

        LocalPipelineExecutor(pipeline1).run()
        LocalPipelineExecutor(
            pipeline2,
            tasks=EXACT_CONTENT_DEDUP_FINDER_WORKERS,
            workers=EXACT_CONTENT_DEDUP_FINDER_WORKERS,
        ).run()
        LocalPipelineExecutor(pipeline3).run()


# =====================================================================================
# Step 3: extract
# =====================================================================================

def run_extract(gpus: int = 1):
    skip_gpu = _skip_gpu_steps(gpus)
    if skip_gpu:
        print(
            "Skipping RolmOCR (vLLM): no GPU / vllm not installed (expected on Windows CPU install).",
            file=sys.stderr,
        )
    if not _opendataloader_ready():
        print(
            "Skipping OpenDataLoader PDF: Java 11+ not found on PATH.\n"
            "Install JDK from https://adoptium.net/ and ensure `java -version` works.",
            file=sys.stderr,
        )
    # Non-OCR extraction (OpenDataLoader PDF; needs Java, no heron/docling models)
    if _opendataloader_ready():
        for truncation in ["truncated", "non_truncated"]:
            if not _content_dedup_has_inputs(truncation, "non_ocr"):
                print(
                    f"Skipping step 3 OpenDataLoader ({truncation}): no files in "
                    f"{DEDUP_OUTPUT_DIR.format(prefix=f'{truncation}/non_ocr')}",
                    file=sys.stderr,
                )
                continue
            if truncation == "non_truncated":
                reader = WarcReaderFast(
                    data_folder=cc_data_root(),
                    preserve_order=True,
                    workers=1,
                )
            else:
                reader = ZstdReader(
                    data_folder=PDF_SAVE_DIR,
                    workers=1,
                    preserve_order=True,
                )
            pipeline_extract = [
                JsonlReader(
                    data_folder=INPUT_DIR_EXTRACT.format(prefix=f"{truncation}/non_ocr"),
                    glob_pattern="**/*.jsonl.gz",
                    doc_progress=True,
                ),
                reader,
                OpenDataLoaderExtractor(
                    timeout=10 * 60,
                    exclusion_writer=JsonlWriter(
                        output_folder=OUTPUT_NON_OCR_DIR.format(prefix=f"{truncation}/failed")
                    ),
                ),
                JsonlWriter(output_folder=OUTPUT_NON_OCR_DIR.format(prefix=f"{truncation}/extracted")),
            ]
            LocalPipelineExecutor(pipeline_extract).run()

    # OCR extraction via InferenceRunner
    # For production environment, this is too slow as you should asynchronously fetch PDFs from bucket inside the query preparation step.
    # We use synchronous fetching here for simplicity.
    # On h100 you should be able to see ~5 pages/s per worker
    if skip_gpu:
        return
    for truncation in ["truncated", "non_truncated"]:
        if not _content_dedup_has_inputs(truncation, "ocr"):
            print(
                f"Skipping step 3 RolmOCR ({truncation}): no files in "
                f"{DEDUP_OUTPUT_DIR.format(prefix=f'{truncation}/ocr')}",
                file=sys.stderr,
            )
            continue
        if truncation == "non_truncated":
            reader = WarcReaderFast(
                data_folder=cc_data_root(),
                preserve_order=True,
                workers=1,
            )
        else:
            reader = ZstdReader(
                data_folder=PDF_SAVE_DIR,
                workers=1,
                preserve_order=True,
            )

        runner = InferenceRunner(
            rollout_fn=rollout_extract,
            config=InferenceConfig(
                model_name_or_path="reducto/RolmOCR",
                default_generation_params={"temperature": 0.0},
                max_concurrent_generations=1,
                server_type="vllm",
                metric_interval=100,
                dp=gpus,
            ),
            output_writer=JsonlWriter(output_folder=OUTPUT_OCR_DIR.format(prefix=f"{truncation}/extracted")),
        )

        pipeline = [
            JsonlReader(
                data_folder=INPUT_DIR_EXTRACT.format(prefix=f"{truncation}/ocr"),
                glob_pattern="**/*.jsonl.gz",
                doc_progress=True,
            ),
            reader,
            runner,
        ]
        LocalPipelineExecutor(pipeline).run()


# =====================================================================================
# Step 4: postprocess
# =====================================================================================

def run_postprocess(gpus: int = 1):
    language_tagger = LanguageTagger(language_threshold=0.01, label_only=True, backend="glotlid")
    skip_gpu = _skip_gpu_steps(gpus)
    if not skip_gpu:
        # OCR branch postprocessing
        for truncation in ["truncated", "non_truncated"]:
            ocr_glob = os.path.join(
                OCR_INPUT_DIR, truncation, "extracted", "*.jsonl.gz"
            )
            if not glob.glob(ocr_glob):
                continue
            if truncation == "non_truncated":
                reader = WarcReaderFast(
                    data_folder=cc_data_root(),
                    preserve_order=True,
                    workers=1,
                )
            else:
                reader = ZstdReader(
                    data_folder=PDF_SAVE_DIR,
                    workers=1,
                    preserve_order=True,
                )

            pipeline_ocr = [
                JsonlReader(
                    data_folder=OCR_INPUT_DIR,
                    glob_pattern=f"{truncation}/extracted/*.jsonl.gz",
                ),
                reader,
                AddMetadata(is_docling=False, is_truncated=truncation == "truncated"),
                DropFailedDocuments(EMPTY_PAGES_DOCLING_DIR),
                CoallesceFailedPages(FAILED_PAGES_OCR_DIR),
                TagBoilerplateFormatter(is_ocr=True, drop=True),
                Normalize(is_from_docling=False),
                language_tagger,
                TokensCounter(batch_size=16, tokenizer_name_or_path="hynky/Llama-3.2-1B-no-bos"),
                # Removes hallucinations caused by blank pages
                # This is very expensive the way we are doing so, better option
                # would be to finetuned ViT model to classify blank pages, however
                # this we didn't have time for it.
                InferenceRunner(
                    rollout_fn=rollout_postprocess,
                    config=InferenceConfig(
                        model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
                        default_generation_params={"temperature": 0.0},
                        max_concurrent_generations=1,
                        server_type="vllm",
                        metric_interval=100,
                    ),
                    output_writer=JsonlWriter(
                        output_folder=SAVE_OCR_DIR.format(prefix=f"extracted")
                    ),
                ),
            ]
            LocalPipelineExecutor(pipeline_ocr).run()
    else:
        print(
            "Skipping OCR postprocess (Qwen vLLM): no GPU / vllm not installed.",
            file=sys.stderr,
        )



    pipeline_docling: list[PipelineStep] = []
    if glob.glob(os.path.join(DOCLING_INPUT_DIR, "non_truncated", "extracted", "*.jsonl.gz")):
        pipeline_docling.extend(
            [
                JsonlReader(
                    data_folder=DOCLING_INPUT_DIR,
                    glob_pattern="non_truncated/extracted/*.jsonl.gz",
                ),
                AddMetadata(is_docling=False, pdf_extractor="opendataloader", is_truncated=False),
            ]
        )
    if glob.glob(os.path.join(DOCLING_INPUT_DIR, "truncated", "extracted", "*.jsonl.gz")):
        pipeline_docling.extend(
            [
                JsonlReader(
                    data_folder=DOCLING_INPUT_DIR,
                    glob_pattern="truncated/extracted/*.jsonl.gz",
                ),
                AddMetadata(is_docling=False, pdf_extractor="opendataloader", is_truncated=True),
            ]
        )
    if not pipeline_docling:
        print(
            "Skipping step 4 postprocess (OpenDataLoader): no extracted jsonl under "
            f"{DOCLING_INPUT_DIR}",
            file=sys.stderr,
        )
        return
    pipeline_docling.extend(
        [
            RemoveDoclingMetadata(),
            DropFailedDocuments(EMPTY_PAGES_DOCLING_DIR),
            PostprocessPageNumbers(),
            CleanTables(),
            RemoveImageAnnotationsByRatio(ratio_threshold=0.8),
            TagBoilerplateFormatter(is_ocr=False, drop=True),
            Normalize(is_from_docling=False),
            language_tagger,
            TokensCounter(batch_size=16, tokenizer_name_or_path="hynky/Llama-3.2-1B-no-bos"),
            JsonlWriter(output_folder=SAVE_DOCLING_DIR),
        ]
    )
    LocalPipelineExecutor(pipeline_docling).run()

# =====================================================================================
# Language filter (glotlid) -> splits into per-language shards
# =====================================================================================



def run_language_filter(languages: Optional[list[str]] = None):
    import json

    with open(TH_VALUES_FILE, "r") as f:
        language_thresholds_dict = {k: max(float(v), 0.05) for k, v in json.load(f).items()}

    # Always allow zxx_* to pass if they are top-1,
    # Tus zxx is never rerouted
    language_thresholds_dict["zxx_Latn"] = -1
    language_thresholds_dict["zxx_Zzzz"] = -1
    language_thresholds_dict["zxx_Arab"] = -1

    pipeline: list[PipelineStep] = []
    if _has_postprocessed_docling():
        pipeline.append(
            JsonlReader(
                data_folder=SAVE_DOCLING_DIR,
                glob_pattern="*.jsonl.gz",
            )
        )
    if _has_postprocessed_ocr():
        pipeline.append(
            JsonlReader(
                data_folder=SAVE_OCR_DIR,
                glob_pattern="**/*.jsonl.gz",
            )
        )
    if not pipeline:
        print(
            "Skipping language filter: no jsonl in "
            f"{SAVE_DOCLING_DIR} or {SAVE_OCR_DIR}",
            file=sys.stderr,
        )
        return
    pipeline.append(
        SelectBestLanguage(language_thresholds_dict=language_thresholds_dict)
    )

    # Optionally restrict to selected languages
    if languages:
        allowed = set(languages)

        def _keep_selected(doc: Document) -> bool:
            return doc.metadata.get("language_bucket") in allowed

        pipeline.append(LambdaFilter(_keep_selected))

    pipeline.append(
        JsonlWriter(
            output_folder=PER_LANGUAGE_DIR_EXACT,
            output_filename="${language_bucket}/${rank}.jsonl.gz",
        )
    )

    LocalPipelineExecutor(pipeline).run()

# =====================================================================================
# Step 5: exact_dedup
# =====================================================================================

def create_content_getter_exact():
    import re as _re

    remove_spaces_regex = _re.compile(r"\s+")

    def content_getter(doc: Document) -> str:
        return remove_spaces_regex.sub("", doc.text)

    return content_getter


EXACT_CONFIG = ExactDedupConfig(content_getter=create_content_getter_exact())


def run_exact_dedup(languages: Optional[list[str]] = None, tasks: int = 2):
    if not languages:
        languages = [
            lang
            for lang in get_datafolder(PER_LANGUAGE_DIR_EXACT).list_files(
                include_directories=True, recursive=False
            )
            if lang
        ]

    for language in languages:
        worker_tasks = max(tasks // 2, 1)

        pipeline1 = [
            JsonlReader(data_folder=f"{PER_LANGUAGE_DIR_EXACT}/{language}"),
            ExactDedupSignature(
                config=EXACT_CONFIG,
                output_folder=f"{OUTPUT_DIR_EXACT}/sigs/{language}",
                finder_workers=worker_tasks,
            ),
        ]
        pipeline2 = [
            ExactFindDedups(
                config=EXACT_CONFIG,
                data_folder=f"{OUTPUT_DIR_EXACT}/sigs/{language}",
                output_folder=f"{OUTPUT_DIR_EXACT}/dups/{language}",
            )
        ]
        output_writer = JsonlWriter(output_folder=f"{OUTPUT_DIR_EXACT}/output/{language}")
        exclude_writer = JsonlWriter(output_folder=f"{OUTPUT_DIR_EXACT}/removed/{language}")
        pipeline3 = [
            JsonlReader(data_folder=f"{PER_LANGUAGE_DIR_EXACT}/{language}"),
            ExactDedupFilter(
                config=EXACT_CONFIG,
                data_folder=f"{OUTPUT_DIR_EXACT}/dups/{language}",
                exclusion_writer=exclude_writer,
            ),
            # Adds chunk for model classification
            AddTextChunks(tokenizer_name="answerdotai/ModernBERT-large" if language == "eng_Latn" else "mmbert-colab/mmBERT-base"),
            output_writer,
        ]

        LocalPipelineExecutor(pipeline1).run()
        LocalPipelineExecutor(pipeline2, tasks=worker_tasks).run()
        LocalPipelineExecutor(pipeline3).run()


# =====================================================================================
# Step 6: model_classification
# =====================================================================================

def model_exists(repo_id: str) -> bool:
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        api.model_info(repo_id)
        return True
    except Exception:
        return False


def make_rollout_model(output_fields_in_order: list[str]):
    async def rollout_model(document: Document, generate: Any, **kwargs):
        requests = [{"input": chunk} for chunk in document.metadata["chunks"]]
        tasks = [generate(req) for req in requests]
        results = await asyncio.gather(*tasks)

        per_field_scores: dict[str, list[float | None]] = {field: [] for field in output_fields_in_order}
        for chunk_result in results:
            if isinstance(chunk_result, InferenceResult):
                try:
                    values = [float(part.strip()) for part in chunk_result.text.split(",") if part.strip() != ""]
                except Exception:
                    values = []
            else:
                values = []
            for idx, field in enumerate(output_fields_in_order):
                value = values[idx] if idx < len(values) else None
                per_field_scores[field].append(value)
        for field, series in per_field_scores.items():
            document.metadata[field] = series
        document.metadata.pop("chunks", None)
        return document

    return rollout_model


def run_model_classification(languages: Optional[list[str]] = None, gpus: int = 1):
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if not languages:
        languages = [
            lang
            for lang in get_datafolder(INPUT_DIR_MODEL).list_files(
                include_directories=True, recursive=False
            )
            if lang
        ]

    for language in languages:
        edu_model = f"HuggingFaceFW/finepdfs_edu_classifier_{language}"
        dclm_model = f"HuggingFaceFW/finepdfs_dclm_classifier_{language}"
        present_models: list[str] = []
        output_fields_in_order: list[str] = []
        if model_exists(edu_model):
            present_models.append(edu_model)
            output_fields_in_order.append("fw_edu_scores")
        if model_exists(dclm_model):
            present_models.append(dclm_model)
            output_fields_in_order.append("dclm_scores")

        data_folder = f"{INPUT_DIR_MODEL}/{language}"
        output_folder = f"{OUTPUT_DIR_MODEL}/{language}"

        if not present_models:
            pipeline = [
                JsonlReader(data_folder=data_folder, glob_pattern="*.jsonl.gz"),
                JsonlWriter(output_folder=output_folder),
            ]
            LocalPipelineExecutor(pipeline=pipeline).run()
            continue

        model_kwargs = {
            "server_script": f"{CURRENT_DIR}/blocks/classification/tf_batching.py",
            "batch-size": 8,
            "batch-timeout": 10.0,
            "max-context": 1024,
            "model-name-or-path": ";".join(present_models),
            "host": "0.0.0.0",
        }

        config = InferenceConfig(
            server_type="custom",
            model_name_or_path=";".join(present_models),
            max_concurrent_generations=2,
            max_concurrent_documents=2,
            model_kwargs=model_kwargs,
            dp=gpus,
            use_chat=False,
            server_log_folder="./server_logs",
        )

        pipeline = [
            JsonlReader(data_folder=data_folder, glob_pattern="*.jsonl.gz"),
            InferenceRunner(
                rollout_fn=make_rollout_model(output_fields_in_order),
                config=config,
                output_writer=JsonlWriter(output_folder=output_folder),
            ),
        ]
        LocalPipelineExecutor(pipeline=pipeline).run()


# =====================================================================================
# Step 7: minhash
# =====================================================================================

def create_content_getter_minhash():
    import re

    def content_getter(doc: Document) -> str:
        max_len = 1_000_000
        text = doc.text
        if len(text) > max_len:
            # Find the next whitespace (any \s) after max_len using regex
            match = re.search(r"\s", text[max_len:])
            if match is None:
                # No whitespace found, just truncate at max_len
                text = text[:max_len]
            else:
                text = text[:max_len + match.start()]
        return text

    return content_getter


MINHASH_CONFIG = MinhashConfig(
    hash_config=HashConfig(hash_fc="xxhash", precision=64),
    num_buckets=32,
    hashes_per_bucket=10,
    n_grams=5,
)


def run_minhash(languages: Optional[list[str]] = None):
    if not languages:
        languages = [
            lang
            for lang in get_datafolder(PER_LANGUAGE_DIR_MINHASH).list_files(
                include_directories=True, recursive=False
            )
            if lang
        ]

    for language in languages:
        tasks = len(get_datafolder(f"{PER_LANGUAGE_DIR_MINHASH}/{language}").list_files(glob_pattern="*.jsonl.gz"))
        WORKERS = max(tasks // 2, MINHASH_CONFIG.num_buckets) // MINHASH_CONFIG.num_buckets * MINHASH_CONFIG.num_buckets

        output_folder = f"{OUTPUT_DIR_MINHASH}/{language}"
        if language == "eng_Latn":
            input_block = [JsonlReader(data_folder=f"{PER_LANGUAGE_DIR_MINHASH}/{language}"), LambdaFilter(lambda x: max(x.metadata.get("fw_edu_scores", [0])) >= 0.5)]
        else:
            input_block = [JsonlReader(data_folder=f"{PER_LANGUAGE_DIR_MINHASH}/{language}")]

        # Tokenizers map as some languages are too slow/don't have a tokenizer
        tokenizers_map = {
            "lat_Latn": "ita_Latn",
            "kaz_Cyrl": "rus_Cyrl",
            "cym_Latn": "eng_Latn",
            "glg_Latn": "por_Latn",
            "und_Hyra": "jpn_Jpan",
            "unknown": "eng_Latn",
        }

        tokenizer_name = tokenizers_map.get(language, language)

        pipeline1 = [
            *input_block,
            MinhashDedupSignature(output_folder=f"{output_folder}/signatures", config=MINHASH_CONFIG, language=tokenizer_name),
        ]
        pipeline2 = [
            MinhashDedupBuckets(
                input_folder=f"{output_folder}/signatures",
                output_folder=f"{output_folder}/buckets",
                config=MINHASH_CONFIG,
                lines_to_buffer=500,
            ),
        ]
        pipeline3 = [
            MinhashDedupCluster(
                input_folder=f"{output_folder}/buckets",
                output_folder=f"{output_folder}/clusters",
                config=MINHASH_CONFIG,
                save_cluster_id=True,
                save_cluster_size=True,
            ),
        ]
        pipeline4 = [
            *input_block,
            MinhashDedupFilter(
                input_folder=f"{output_folder}/clusters",
                exclusion_writer=JsonlWriter(output_folder=f"{output_folder}/removed"),
                load_cluster_ids=True,
                load_cluster_sizes=True,
            ),
            JsonlWriter(output_folder=f"{output_folder}/output"),
        ]

        LocalPipelineExecutor(pipeline1, tasks=tasks).run()
        LocalPipelineExecutor(pipeline2, tasks=WORKERS).run()
        LocalPipelineExecutor(pipeline3, tasks=1).run()
        LocalPipelineExecutor(pipeline4, tasks=tasks).run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FinePDFs pipeline sequentially")
    parser.add_argument("--crawl-ids", type=str, required=True, help="Comma-separated CommonCrawl crawl IDs for step 1")
    parser.add_argument("--languages", type=str, default=None, help="Comma-separated list of languages for steps 5-8")
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="GPU count for vLLM steps (0 on Windows CPU). Set 1+ only with vllm on Linux+NVIDIA",
    )
    return parser.parse_args()


def main():
    # Windows console: UTF-8 logs; avoid bogus AWS credential lookup for public S3
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    _patch_datatrove_jsonl_utf8()

    args = parse_args()
    languages = [s.strip() for s in args.languages.split(",")] if args.languages else None
    crawl_ids = [s.strip() for s in args.crawl_ids.split(",")] if args.crawl_ids else []

    # Sequentially run all steps
    run_filter_pdfs_and_refetch(crawl_ids)
    if not _step1_produced_any():
        print(
            f"Step 1 finished but found no PDFs in the first {LIMIT} WARC records. "
            "Raise LIMIT in run_finepdfs_pipeline.py or try another --crawl-ids.",
            file=sys.stderr,
        )
        sys.exit(1)
    run_content_dedup_ocr_organize()
    if not _any_content_dedup_non_ocr():
        print(
            "Step 2 produced no documents in content_dedup/*/non_ocr. "
            "If the OCR classifier failed, run: git lfs pull --include=models/xgb_ocr_classifier/xgb_classifier.ubj",
            file=sys.stderr,
        )
        sys.exit(1)
    run_extract(args.gpus)
    if not _has_extracted_outputs():
        print(
            "Step 3 produced no extracted documents.\n"
            "OpenDataLoader: install Java 11+ (https://adoptium.net/)\n"
            "RolmOCR: Linux + NVIDIA GPU + pip install -e '.[gpu]'",
            file=sys.stderr,
        )
        sys.exit(1)
    run_postprocess(args.gpus)
    if not _has_postprocessed_docling() and not _has_postprocessed_ocr():
        print("Step 4 produced no postprocessed jsonl.", file=sys.stderr)
        sys.exit(1)
    run_language_filter(languages=languages)
    if not _language_shards_ready(languages):
        print(
            "Language filter produced no matching documents"
            + (f" for {languages}." if languages else ".")
            + " PDF text may be another language (see step 4). "
            "Retry without --languages or pick the language from glotlid output.",
            file=sys.stderr,
        )
        sys.exit(1)
    run_exact_dedup(languages=languages)
    run_model_classification(languages=languages, gpus=args.gpus)
    run_minhash(languages=languages)


if __name__ == "__main__":
    main()


