from __future__ import annotations

import os
from typing import Any
import cs336_data.utils as utils
from cs336_data.gopher import GopherFilter
import cs336_data.dedup as dedup


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return utils.html_to_txt(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    return utils.LanguageDetector().detect_language(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return utils.PIIFilter().mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return utils.PIIFilter().mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return utils.PIIFilter().mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return utils.NSFWDetector().filter_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return utils.ToxicDetector().filter_toxic(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_gopher_quality_filter(text: str) -> bool:
    return GopherFilter().filter(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    return dedup.exact_dedup(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    raise NotImplementedError
