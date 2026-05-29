#!/usr/bin/env python3
"""
FinePDFs pipeline with Docling PDF extraction (steps 3–4).

Steps 1–2 and 5–8 reuse run_finepdfs_pipeline.py. Outputs go under
./finepdfs/data_docling/ so they do not overwrite OpenDataLoader results.

Usage (same args as the main script):
  python run_finepdfs_pipeline_docling.py --crawl-ids CC-MAIN-2024-10
"""
from __future__ import annotations

import glob
import os
import sys

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.media.media_readers.warc import WarcReaderFast
from datatrove.pipeline.media.media_readers.zstd import ZstdReader
from datatrove.pipeline.readers.jsonl import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.pipeline.tokens.counter import TokensCounter

from blocks.extractors.docling import DoclingExtractor
from postprocessing.page_numbers import PostprocessPageNumbers
from postprocessing.boilerplate import TagBoilerplateFormatter
from postprocessing.language import LanguageTagger
from postprocessing.tables import CleanTables
from postprocessing.normalize import Normalize
from postprocessing.remove_image_annots import RemoveImageAnnotationsByRatio

import run_finepdfs_pipeline as pipeline
from pipeline_utils.postprocess_utils import (
    AddMetadata,
    RemoveDoclingMetadata,
    DropFailedDocuments,
    CoallesceFailedPages,
)

DATA_ROOT = "./finepdfs/data_docling"


def apply_docling_data_paths() -> None:
    """Separate artifact dirs from the OpenDataLoader pipeline."""
    pipeline.OUTPUT_NON_OCR_DIR = f"{DATA_ROOT}/non_ocr_docs_extracted/{{prefix}}"
    pipeline.DOCLING_INPUT_DIR = f"{DATA_ROOT}/non_ocr_docs_extracted"
    pipeline.SAVE_DOCLING_DIR = f"{DATA_ROOT}/postprocessed/output"
    pipeline.EMPTY_PAGES_DOCLING_DIR = f"{DATA_ROOT}/postprocessed/empty"
    pipeline.PER_LANGUAGE_DIR_EXACT = f"{DATA_ROOT}/glotlid/per_language"
    pipeline.OUTPUT_DIR_EXACT = f"{DATA_ROOT}/exact_dedup/per_language"
    pipeline.INPUT_DIR_MODEL = f"{DATA_ROOT}/exact_dedup/per_language/output"
    pipeline.OUTPUT_DIR_MODEL = f"{DATA_ROOT}/model_labeling/per_language"
    pipeline.PER_LANGUAGE_DIR_MINHASH = f"{DATA_ROOT}/model_labeling/per_language"
    pipeline.OUTPUT_DIR_MINHASH = f"{DATA_ROOT}/minhash/per_language"


def run_extract_docling(gpus: int = 1) -> None:
    skip_gpu = pipeline._skip_gpu_steps(gpus)
    if skip_gpu:
        print(
            "Skipping RolmOCR (vLLM): no GPU / vllm not installed (expected on Windows CPU install).",
            file=sys.stderr,
        )

    for truncation in ["truncated", "non_truncated"]:
        if not pipeline._content_dedup_has_inputs(truncation, "non_ocr"):
            print(
                f"Skipping step 3 Docling ({truncation}): no files in "
                f"{pipeline.DEDUP_OUTPUT_DIR.format(prefix=f'{truncation}/non_ocr')}",
                file=sys.stderr,
            )
            continue
        if truncation == "non_truncated":
            reader = WarcReaderFast(
                data_folder=pipeline.cc_data_root(),
                preserve_order=True,
                workers=1,
            )
        else:
            reader = ZstdReader(
                data_folder=pipeline.PDF_SAVE_DIR,
                workers=1,
                preserve_order=True,
            )
        steps = [
            JsonlReader(
                data_folder=pipeline.INPUT_DIR_EXTRACT.format(prefix=f"{truncation}/non_ocr"),
                glob_pattern="**/*.jsonl.gz",
                doc_progress=True,
            ),
            reader,
            DoclingExtractor(
                timeout=10 * 60,
                exclusion_writer=JsonlWriter(
                    output_folder=pipeline.OUTPUT_NON_OCR_DIR.format(prefix=f"{truncation}/failed")
                ),
            ),
            JsonlWriter(output_folder=pipeline.OUTPUT_NON_OCR_DIR.format(prefix=f"{truncation}/extracted")),
        ]
        LocalPipelineExecutor(steps).run()

    if skip_gpu:
        return

    from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner

    from pipeline_utils.extract_utils import rollout_extract

    for truncation in ["truncated", "non_truncated"]:
        if not pipeline._content_dedup_has_inputs(truncation, "ocr"):
            continue
        if truncation == "non_truncated":
            reader = WarcReaderFast(
                data_folder=pipeline.cc_data_root(),
                preserve_order=True,
                workers=1,
            )
        else:
            reader = ZstdReader(
                data_folder=pipeline.PDF_SAVE_DIR,
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
            output_writer=JsonlWriter(
                output_folder=pipeline.OUTPUT_OCR_DIR.format(prefix=f"{truncation}/extracted")
            ),
        )
        LocalPipelineExecutor(
            [
                JsonlReader(
                    data_folder=pipeline.INPUT_DIR_EXTRACT.format(prefix=f"{truncation}/ocr"),
                    glob_pattern="**/*.jsonl.gz",
                    doc_progress=True,
                ),
                reader,
                runner,
            ]
        ).run()


def run_postprocess_docling(gpus: int = 1) -> None:
    language_tagger = LanguageTagger(language_threshold=0.01, label_only=True, backend="glotlid")
    skip_gpu = pipeline._skip_gpu_steps(gpus)

    if not skip_gpu:
        from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner

        from pipeline_utils.postprocess_utils import rollout_postprocess

        for truncation in ["truncated", "non_truncated"]:
            ocr_glob = os.path.join(
                pipeline.OCR_INPUT_DIR, truncation, "extracted", "*.jsonl.gz"
            )
            if not glob.glob(ocr_glob):
                continue
            if truncation == "non_truncated":
                reader = WarcReaderFast(
                    data_folder=pipeline.cc_data_root(),
                    preserve_order=True,
                    workers=1,
                )
            else:
                reader = ZstdReader(
                    data_folder=pipeline.PDF_SAVE_DIR,
                    workers=1,
                    preserve_order=True,
                )
            steps = [
                JsonlReader(
                    data_folder=pipeline.OCR_INPUT_DIR,
                    glob_pattern=f"{truncation}/extracted/*.jsonl.gz",
                ),
                reader,
                AddMetadata(is_docling=False, is_truncated=truncation == "truncated"),
                DropFailedDocuments(pipeline.EMPTY_PAGES_DOCLING_DIR),
                CoallesceFailedPages(pipeline.FAILED_PAGES_OCR_DIR),
                TagBoilerplateFormatter(is_ocr=True, drop=True),
                Normalize(is_from_docling=False),
                language_tagger,
                TokensCounter(batch_size=16, tokenizer_name_or_path="hynky/Llama-3.2-1B-no-bos"),
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
                        output_folder=pipeline.SAVE_OCR_DIR.format(prefix="extracted")
                    ),
                ),
            ]
            LocalPipelineExecutor(steps).run()
    else:
        print(
            "Skipping OCR postprocess (Qwen vLLM): no GPU / vllm not installed.",
            file=sys.stderr,
        )

    steps: list = []
    if glob.glob(os.path.join(pipeline.DOCLING_INPUT_DIR, "non_truncated", "extracted", "*.jsonl.gz")):
        steps.extend(
            [
                JsonlReader(
                    data_folder=pipeline.DOCLING_INPUT_DIR,
                    glob_pattern="non_truncated/extracted/*.jsonl.gz",
                ),
                AddMetadata(is_docling=True, pdf_extractor="docling", is_truncated=False),
            ]
        )
    if glob.glob(os.path.join(pipeline.DOCLING_INPUT_DIR, "truncated", "extracted", "*.jsonl.gz")):
        steps.extend(
            [
                JsonlReader(
                    data_folder=pipeline.DOCLING_INPUT_DIR,
                    glob_pattern="truncated/extracted/*.jsonl.gz",
                ),
                AddMetadata(is_docling=True, pdf_extractor="docling", is_truncated=True),
            ]
        )
    if not steps:
        print(
            f"Skipping step 4 postprocess (Docling): no extracted jsonl under "
            f"{pipeline.DOCLING_INPUT_DIR}",
            file=sys.stderr,
        )
        return
    steps.extend(
        [
            RemoveDoclingMetadata(),
            DropFailedDocuments(pipeline.EMPTY_PAGES_DOCLING_DIR),
            PostprocessPageNumbers(),
            CleanTables(),
            RemoveImageAnnotationsByRatio(ratio_threshold=0.8),
            TagBoilerplateFormatter(is_ocr=False, drop=True),
            Normalize(is_from_docling=True),
            language_tagger,
            TokensCounter(batch_size=16, tokenizer_name_or_path="hynky/Llama-3.2-1B-no-bos"),
            JsonlWriter(output_folder=pipeline.SAVE_DOCLING_DIR),
        ]
    )
    LocalPipelineExecutor(steps).run()


def main() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    pipeline._patch_datatrove_jsonl_utf8()

    args = pipeline.parse_args()
    languages = [s.strip() for s in args.languages.split(",")] if args.languages else None
    crawl_ids = [s.strip() for s in args.crawl_ids.split(",")] if args.crawl_ids else []

    apply_docling_data_paths()

    pipeline.run_filter_pdfs_and_refetch(crawl_ids)
    if not pipeline._step1_produced_any():
        print(
            f"Step 1 finished but found no PDFs in the first {pipeline.LIMIT} WARC records. "
            "Raise LIMIT in run_finepdfs_pipeline.py or try another --crawl-ids.",
            file=sys.stderr,
        )
        sys.exit(1)
    pipeline.run_content_dedup_ocr_organize()
    if not pipeline._any_content_dedup_non_ocr():
        print(
            "Step 2 produced no documents in content_dedup/*/non_ocr. "
            "If the OCR classifier failed, run: git lfs pull --include=models/xgb_ocr_classifier/xgb_classifier.ubj",
            file=sys.stderr,
        )
        sys.exit(1)

    run_extract_docling(args.gpus)
    if not pipeline._has_extracted_outputs():
        print("Step 3 (Docling) produced no extracted documents.", file=sys.stderr)
        sys.exit(1)
    run_postprocess_docling(args.gpus)
    if not pipeline._has_postprocessed_docling() and not pipeline._has_postprocessed_ocr():
        print("Step 4 (Docling) produced no postprocessed jsonl.", file=sys.stderr)
        sys.exit(1)
    pipeline.run_language_filter(languages=languages)
    if not pipeline._language_shards_ready(languages):
        print(
            "Language filter produced no matching documents"
            + (f" for {languages}." if languages else ".")
            + " PDF text may be another language (see step 4). "
            "Retry without --languages or pick the language from glotlid output.",
            file=sys.stderr,
        )
        sys.exit(1)
    pipeline.run_exact_dedup(languages=languages)
    pipeline.run_model_classification(languages=languages, gpus=args.gpus)
    pipeline.run_minhash(languages=languages)


if __name__ == "__main__":
    main()
