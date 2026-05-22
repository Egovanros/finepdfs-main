#!/usr/bin/env python3
"""
Compare Docling vs OpenDataLoader on the same crawl inputs (content_dedup/*/non_ocr).

Same PDF bytes, same timeout as pipeline step 3. Reports yield (how many docs
each extractor succeeds on) and per-document timing.
"""
from __future__ import annotations

import gzip
import glob
import json
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from datatrove.data import Document, Media
from datatrove.pipeline.media.media_readers.warc import WarcReaderFast
from datatrove.pipeline.media.media_readers.zstd import ZstdReader

from blocks.extractors.docling import DoclingExtractor
from blocks.extractors.opendataloader import OpenDataLoaderExtractor, java_version_ok
from run_finepdfs_pipeline import (
    INPUT_DIR_EXTRACT,
    LIMIT,
    OUTPUT_NON_OCR_DIR,
    PDF_SAVE_DIR,
    cc_data_root,
)

TIMEOUT = 10 * 60
REPORT_PATH = Path("benchmark_docling_vs_odl_results.json")


@dataclass
class SampleResult:
    doc_id: str
    branch: str
    url: str
    pdf_bytes: int
    fetch_ok: bool
    odl_sec: float | None
    odl_ok: bool
    odl_text_len: int
    odl_error: str | None
    docling_sec: float | None
    docling_ok: bool
    docling_text_len: int
    docling_error: str | None
    speedup_docling_over_odl: float | None


def _json_to_document(raw: dict) -> Document:
    media_list = []
    for m in raw.get("media") or []:
        media_list.append(
            Media(
                id=m["id"],
                type=m["type"],
                url=m.get("url") or "",
                alt=m.get("alt"),
                path=m.get("path"),
                offset=m.get("offset"),
                length=m.get("length"),
                media_bytes=m.get("media_bytes"),
                metadata=m.get("metadata") or {},
            )
        )
    return Document(
        text=raw.get("text") or "",
        id=raw["id"],
        media=media_list,
        metadata=raw.get("metadata") or {},
    )


def load_all_branch_docs(branch: str) -> list[Document]:
    pattern = INPUT_DIR_EXTRACT.format(prefix=f"{branch}/non_ocr") + "/**/*.jsonl.gz"
    paths = sorted(glob.glob(pattern, recursive=True))
    docs: list[Document] = []
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                docs.append(_json_to_document(json.loads(line)))
    return docs


def _is_zstd_media(media: Media) -> bool:
    path = (media.path or "").lower()
    return path.endswith(".zstd") or path.endswith(".zst")


def fetch_pdf_bytes(branch: str, docs: list[Document]) -> tuple[list[Document], int]:
    """Return docs with media_bytes set; second value = fetch failures."""
    if branch == "truncated" or (docs and docs[0].media and _is_zstd_media(docs[0].media[0])):
        reader = ZstdReader(data_folder=PDF_SAVE_DIR, workers=1, preserve_order=True)
        reader.thread_local = threading.local()
        reader._init_thread_local()
    else:
        reader = WarcReaderFast(data_folder=cc_data_root(), preserve_order=True, workers=1)

    out: list[Document] = []
    failed_ids: list[str] = []
    for doc in docs:
        if not doc.media:
            failed_ids.append(doc.id)
            continue
        media = doc.media[0]
        try:
            data = reader.read_media_record(media)
        except Exception as exc:
            print(f"  fetch error {doc.id}: {exc}", file=sys.stderr)
            failed_ids.append(doc.id)
            continue
        if not data:
            failed_ids.append(doc.id)
            continue
        media.media_bytes = data
        out.append(doc)
    return out, failed_ids


def timed_extract(extractor, media_bytes: bytes, metadata: dict) -> tuple[float, str, dict]:
    t0 = time.perf_counter()
    text, meta = extractor.extract(media_bytes, metadata)
    elapsed = time.perf_counter() - t0
    err = meta.get("extraction_error")
    ok = bool(text.strip()) and not err
    return elapsed, text, {"ok": ok, "error": err}


def count_pipeline_extracted() -> dict[str, dict[str, int]]:
    """Documents already written by last pipeline run (OpenDataLoader step 3)."""
    out: dict[str, dict[str, int]] = {}
    for branch in ("non_truncated", "truncated"):
        pattern = OUTPUT_NON_OCR_DIR.format(prefix=f"{branch}/extracted") + "/**/*.jsonl.gz"
        ids: set[str] = set()
        for path in glob.glob(pattern, recursive=True):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    ids.add(json.loads(line)["id"])
        failed_ids: set[str] = set()
        fail_pattern = OUTPUT_NON_OCR_DIR.format(prefix=f"{branch}/failed") + "/**/*.jsonl.gz"
        for path in glob.glob(fail_pattern, recursive=True):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    failed_ids.add(json.loads(line)["id"])
        out[branch] = {"extracted": len(ids), "failed": len(failed_ids)}
    return out


def yield_summary(results: list[SampleResult]) -> dict:
    fetched = [r for r in results if r.fetch_ok]
    odl_ok = [r for r in fetched if r.odl_ok]
    docling_ok = [r for r in fetched if r.docling_ok]
    both = [r for r in fetched if r.odl_ok and r.docling_ok]
    only_odl = [r for r in fetched if r.odl_ok and not r.docling_ok]
    only_docling = [r for r in fetched if r.docling_ok and not r.odl_ok]
    neither = [r for r in fetched if not r.odl_ok and not r.docling_ok]

    def _times(ok_list, attr):
        vals = [getattr(r, attr) for r in ok_list if getattr(r, attr)]
        return round(sum(vals), 2), round(sum(vals) / len(vals), 2) if vals else None

    odl_total, odl_avg = _times(odl_ok, "odl_sec")
    doc_total, doc_avg = _times(docling_ok, "docling_sec")
    both_speedups = [
        r.speedup_docling_over_odl
        for r in both
        if r.speedup_docling_over_odl
    ]

    return {
        "candidates": len(results),
        "pdf_fetched": len(fetched),
        "fetch_failed": len(results) - len(fetched),
        "opendataloader_ok": len(odl_ok),
        "docling_ok": len(docling_ok),
        "both_ok": len(both),
        "only_opendataloader": len(only_odl),
        "only_docling": len(only_docling),
        "neither_ok": len(neither),
        "odl_total_sec": odl_total,
        "odl_avg_sec_per_ok": odl_avg,
        "docling_total_sec": doc_total,
        "docling_avg_sec_per_ok": doc_avg,
        "speedup_docling_vs_odl_total": round(doc_total / odl_total, 2) if odl_total else None,
        "speedup_mean_per_doc": round(sum(both_speedups) / len(both_speedups), 2) if both_speedups else None,
    }


def main() -> None:
    java_ok, java_msg = java_version_ok()
    if not java_ok:
        print(f"OpenDataLoader unavailable: {java_msg}", file=sys.stderr)
        sys.exit(1)

    print("Input: content_dedup/*/non_ocr (same as pipeline step 3)")
    print(f"WARC LIMIT (step 1) = {LIMIT}, extract timeout = {TIMEOUT}s\n")

    batches: list[tuple[str, list[Document]]] = []
    for branch in ("non_truncated", "truncated"):
        docs = load_all_branch_docs(branch)
        print(f"{branch}: {len(docs)} candidates in content_dedup")
        batches.append((branch, docs))

    odl = OpenDataLoaderExtractor(timeout=TIMEOUT)
    print("\nInitializing Docling...")
    t0 = time.perf_counter()
    docling = DoclingExtractor(timeout=TIMEOUT)
    print(f"DoclingExtractor ready in {time.perf_counter() - t0:.1f}s\n")

    results: list[SampleResult] = []

    for branch, docs in batches:
        print(f"=== {branch}: fetching PDF bytes ===")
        fetched_docs, fetch_failed_ids = fetch_pdf_bytes(branch, docs)
        print(f"  fetched {len(fetched_docs)}, failed {len(fetch_failed_ids)}")

        for i, doc in enumerate(fetched_docs, 1):
            media = doc.media[0]
            url = media.url or ""
            nbytes = len(media.media_bytes or b"")
            meta = dict(media.metadata or {})
            print(f"[{branch} {i}/{len(fetched_docs)}] {nbytes/1024/1024:.2f} MiB")

            odl_sec, odl_text, odl_meta = timed_extract(odl, media.media_bytes, meta)
            d_sec, d_text, d_meta = timed_extract(docling, media.media_bytes, meta)
            print(
                f"  ODL {odl_sec:.1f}s ok={odl_meta['ok']} len={len(odl_text)} | "
                f"Docling {d_sec:.1f}s ok={d_meta['ok']} len={len(d_text)}"
            )

            speedup = round(d_sec / odl_sec, 2) if odl_sec and odl_meta["ok"] and d_meta["ok"] else None
            results.append(
                SampleResult(
                    doc_id=doc.id,
                    branch=branch,
                    url=url,
                    pdf_bytes=nbytes,
                    fetch_ok=True,
                    odl_sec=round(odl_sec, 3),
                    odl_ok=bool(odl_meta["ok"]),
                    odl_text_len=len(odl_text),
                    odl_error=odl_meta.get("error"),
                    docling_sec=round(d_sec, 3),
                    docling_ok=bool(d_meta["ok"]),
                    docling_text_len=len(d_text),
                    docling_error=d_meta.get("error"),
                    speedup_docling_over_odl=speedup,
                )
            )

        for fid in fetch_failed_ids:
            results.append(
                SampleResult(
                    doc_id=fid,
                    branch=branch,
                    url="",
                    pdf_bytes=0,
                    fetch_ok=False,
                    odl_sec=None,
                    odl_ok=False,
                    odl_text_len=0,
                    odl_error="fetch_failed",
                    docling_sec=None,
                    docling_ok=False,
                    docling_text_len=0,
                    docling_error="fetch_failed",
                    speedup_docling_over_odl=None,
                )
            )

    total = yield_summary(results)
    by_branch = {
        branch: yield_summary([r for r in results if r.branch == branch])
        for branch in ("non_truncated", "truncated")
    }
    pipeline_odl = count_pipeline_extracted()

    report = {
        "settings": {
            "limit_warc_records_step1": LIMIT,
            "extract_timeout_sec": TIMEOUT,
            "java": java_msg,
            "pdf_save_dir": PDF_SAVE_DIR,
            "cc_data_root": str(cc_data_root()),
        },
        "pipeline_opendataloader_outputs": pipeline_odl,
        "benchmark_total": total,
        "benchmark_by_branch": by_branch,
        "fetch_failed_ids": [r.doc_id for r in results if not r.fetch_ok],
        "results": [asdict(r) for r in results if r.fetch_ok],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print("YIELD COMPARISON (same PDF bytes, same timeout)")
    print("=" * 70)
    print(json.dumps({"total": total, "by_branch": by_branch, "pipeline_odl": pipeline_odl}, indent=2))
    print(f"\nReport: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
