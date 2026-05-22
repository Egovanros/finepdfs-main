import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from blocks.predictor.base_extractor import BaseMediaExtractor
from datatrove.pipeline.writers.disk_base import DiskWriter

PAGE_BREAK = "<--- page break ---"


def _java_executables() -> list[str]:
    """Prefer explicit JDK 17+ over Oracle java8path shims on Windows PATH."""
    seen: set[str] = set()
    candidates: list[str] = []

    def add(path: str | None) -> None:
        if not path:
            return
        p = os.path.normpath(path)
        if p in seen:
            return
        seen.add(p)
        if os.path.isfile(p):
            candidates.append(p)

    for env in ("FINEPDFS_JAVA_HOME", "JAVA_HOME"):
        home = os.environ.get(env)
        if not home:
            continue
        add(os.path.join(home, "bin", "java.exe"))
        add(os.path.join(home, "java.exe"))

    for root in (
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
    ):
        java_dir = os.path.join(root, "Java")
        if not os.path.isdir(java_dir):
            continue
        for name in sorted(os.listdir(java_dir), reverse=True):
            if name.lower().startswith("jdk"):
                add(os.path.join(java_dir, name, "bin", "java.exe"))
        adoptium = os.path.join(root, "Eclipse Adoptium")
        if os.path.isdir(adoptium):
            for name in sorted(os.listdir(adoptium), reverse=True):
                add(os.path.join(adoptium, name, "bin", "java.exe"))

    add(shutil.which("java"))
    return candidates


def java_version_ok() -> tuple[bool, str]:
    executables = _java_executables()
    if not executables:
        return False, "Java not found (install JDK 11+ from https://adoptium.net/)"

    for java in executables:
        proc = subprocess.run(
            [java, "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        out = f"{proc.stderr or ''}{proc.stdout or ''}".strip()
        match = re.search(r'version "(\d+)(?:\.(\d+))?', out)
        if not match:
            continue
        major = int(match.group(1))
        if major == 1 and match.group(2):
            major = int(match.group(2))
        if major < 11:
            continue
        return True, f"{out.splitlines()[0]} ({java})"

    path_java = shutil.which("java") or "unknown"
    return (
        False,
        f"Only Java 8 found on PATH ({path_java}). JDK 17 is at "
        r"C:\Program Files\Java\jdk-17 — set JAVA_HOME or move jdk-17\bin above Oracle java8path in PATH.",
    )


def java_available() -> bool:
    ok, _ = java_version_ok()
    return ok


class OpenDataLoaderExtractor(BaseMediaExtractor):
    """Extract PDF text via opendataloader-pdf (bundled JVM CLI; requires Java 11+ on PATH)."""

    def __init__(
        self,
        timeout: int = 10 * 60,
        hybrid: str | None = None,
        hybrid_url: str | None = None,
        quiet: bool = True,
        exclusion_writer: Optional[DiskWriter] = None,
    ):
        self.hybrid = hybrid or os.environ.get("FINEPDFS_ODL_HYBRID")
        self.hybrid_url = hybrid_url or os.environ.get("FINEPDFS_ODL_HYBRID_URL")
        self.quiet = quiet
        super().__init__(timeout=timeout, exclusion_writer=exclusion_writer)

    def extract(self, media_bytes: bytes | None, document_metadata: dict) -> tuple[str, dict]:
        if media_bytes is None:
            return "", {"extraction_error": "Media bytes are None"}
        ok, java_msg = java_version_ok()
        if not ok:
            return "", {"extraction_error": java_msg}

        import opendataloader_pdf

        java_exe = _java_executables()[0]
        java_bin = os.path.dirname(java_exe)
        env = os.environ.copy()
        env["JAVA_HOME"] = os.path.dirname(java_bin)
        env["PATH"] = f"{java_bin}{os.pathsep}{env.get('PATH', '')}"

        work = tempfile.mkdtemp(prefix="finepdfs_odl_")
        try:
            pdf_path = os.path.join(work, f"{uuid.uuid4().hex}.pdf")
            with open(pdf_path, "wb") as f:
                f.write(media_bytes)
            out_dir = os.path.join(work, "out")
            os.makedirs(out_dir, exist_ok=True)

            convert_kwargs = {
                "input_path": pdf_path,
                "output_dir": out_dir,
                "format": "markdown",
                "quiet": self.quiet,
                "markdown_page_separator": PAGE_BREAK,
            }
            if self.hybrid:
                convert_kwargs["hybrid"] = self.hybrid
            if self.hybrid_url:
                convert_kwargs["hybrid_url"] = self.hybrid_url

            prev_path = os.environ.get("PATH")
            prev_home = os.environ.get("JAVA_HOME")
            try:
                os.environ["PATH"] = env["PATH"]
                os.environ["JAVA_HOME"] = env["JAVA_HOME"]
                opendataloader_pdf.convert(**convert_kwargs)
            finally:
                if prev_path is None:
                    os.environ.pop("PATH", None)
                else:
                    os.environ["PATH"] = prev_path
                if prev_home is None:
                    os.environ.pop("JAVA_HOME", None)
                else:
                    os.environ["JAVA_HOME"] = prev_home

            md_files = sorted(Path(out_dir).rglob("*.md"))
            if not md_files:
                return "", {"extraction_error": f"No markdown output under {out_dir}"}

            full_text = md_files[0].read_text(encoding="utf-8", errors="replace")
            if PAGE_BREAK in full_text:
                page_list = full_text.split(PAGE_BREAK)
            else:
                page_list = [full_text]

            if not any(p.strip() for p in page_list):
                return "", {"extraction_error": "OpenDataLoader returned empty text"}

            metadata = {
                "num_pages": len(page_list),
                "page_offsets": np.cumsum([len(p) for p in page_list]).tolist(),
                "conversion_status": "success",
                "version": "opendataloader-pdf",
                "pdf_extractor": "opendataloader",
            }
            json_files = list(Path(out_dir).rglob("*.json"))
            if json_files:
                metadata["opendataloader_json_file"] = json_files[0].name

            return "".join(page_list), metadata
        except Exception as exc:
            return "", {"extraction_error": str(exc)}
        finally:
            shutil.rmtree(work, ignore_errors=True)
