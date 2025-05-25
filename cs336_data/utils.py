"""
Web content processing and filtering utilities.

This module provides functionality for extracting and filtering text content
from WARC files, including PII masking, language detection, and content filtering.
"""

from fastwarc.warc import ArchiveIterator, WarcRecordType
from fastwarc.stream_io import GZipStream, FileStream
from resiliparse.parse.encoding import detect_encoding
import fasttext
from resiliparse.extract.html2text import extract_plain_text
import os
from typing import Tuple
import random
import re
from cs336_data.gopher import GopherFilter

BASE_DIR = "/Users/christineye/cs336/assignment4-data/data"

def html_to_txt(html: bytes) -> str:
    encoding = detect_encoding(html)
    if encoding is None:
        raise ValueError("Could not detect encoding")
    html_str = html.decode(encoding)
    
    return extract_plain_text(html_str)

def warc_to_txt(warc_file: str, n_records: int = 10, record_id: int = 0):
    stream = GZipStream(FileStream(warc_file, 'rb'))
    for i, record in enumerate(ArchiveIterator(stream, record_types=WarcRecordType.response)):
        if i < record_id:
            continue
        if n_records <= 0:
            break
        n_records -= 1
        
        yield html_to_txt(record.reader.read())

def mask_emails(text: str) -> str:
    email_regex = r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
    masked_text, count = re.subn(email_regex, "|||EMAIL_ADDRESS|||", text, flags = re.IGNORECASE)
    return masked_text, count

def mask_phone_numbers(text: str) -> str:
    phone_regex = r"(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"
    masked_text, count = re.subn(phone_regex, "|||PHONE_NUMBER|||", text)
    return masked_text, count

def mask_ips(text: str) -> str:
    ipv4_regex = r"(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])"
    masked_text, count = re.subn(ipv4_regex, "|||IP_ADDRESS|||", text)
    return masked_text, count

def filter_fasttext(text: str, classifier_id: str) -> str:
    # strip newlines
    text = text.replace("\n", " ")
    
    # load classifier
    classifier_path = os.path.join(BASE_DIR, classifier_id)
    classifier = fasttext.load_model(classifier_path)
    prediction = classifier.predict(text)

    label = prediction[0][0].replace("__label__", "")
    confidence = prediction[1][0]
    return label, confidence

def detect_language(text: str, classifier_id: str = "classifiers/lid.176.bin") -> Tuple[str, float]:
    return filter_fasttext(text, classifier_id)

def filter_nsfw(text: str, classifier_id: str = "classifiers/jigsaw_fasttext_bigrams_nsfw_final.bin") -> str:
    return filter_fasttext(text, classifier_id)

def filter_toxic(text: str, classifier_id: str = "classifiers/jigsaw_fasttext_bigrams_hatespeech_final.bin") -> str:
    return filter_fasttext(text, classifier_id)

def filter_pii(text: str) -> str:
    # strip newlines
    text = text.replace("\n", " ")

if __name__ == "__main__":
    test_task = "gopher"
    WARC_path = os.path.join(BASE_DIR, "CC/example.warc.gz")

    if test_task == "html_to_txt":
        for txt in warc_to_txt(WARC_path, n_records = 1, record_id = 2):
            print(txt)
            print('content length: ', len(txt))
    
    elif test_task == "detect_language":
        id = random.randint(0, 10000)
        for txt in warc_to_txt(WARC_path, n_records = 1, record_id = id):
            print(txt)
            print('Classified language:', detect_language(txt))
    
    elif test_task == "filter_harmful":
        id = random.randint(0, 10000)
        for txt in warc_to_txt(WARC_path, n_records = 1, record_id = id):
            print(txt)
            print('Classified NSFW:', filter_nsfw(txt))
            print('Classified toxic:', filter_toxic(txt))

    elif test_task == "mask":
        num_replacements = 0

        id = random.randint(1, 10000)
        for txt in warc_to_txt(WARC_path, n_records = 10, record_id = id):
            masked_text, email_count = mask_emails(txt)
            masked_text, phone_count = mask_phone_numbers(masked_text) 
            masked_text, ip_count = mask_ips(masked_text)
            total_count = email_count + phone_count + ip_count

            if total_count > 0:
                print('MASKED TEXT:')
                print('-' * 100)
                print(masked_text)
                print('-' * 100)
                print('ORIGINAL TEXT:')
                print('-' * 100)
                print(txt)
                print('-' * 100)
                print('Number of replacements:', total_count)
    
    elif test_task == "gopher":
        id = random.randint(0, 10000)
        for txt in warc_to_txt(WARC_path, n_records = 1, record_id = id):
            print(txt)
            print('Gopher quality:', GopherFilter(verbose = True).filter(txt))