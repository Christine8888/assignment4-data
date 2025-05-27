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
import chardet

#BASE_DIR = "/Users/christineye/cs336/assignment4-data/data"
#BASE_DIR = "/home/c-cye/assignment4-data/cs336_data"
BASE_DIR = "/data"
NSFW_FILTER = "classifiers/dolma_fasttext_nsfw_jigsaw_model.bin"
TOXIC_FILTER = "classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin"
LANGUAGE_FILTER = "classifiers/lid.176.bin"
QUALITY_FILTER = "/home/c-cye/assignment4-data/cs336_data/quality_classifier.bin"

def html_to_txt(html: bytes) -> str:
    encoding = detect_encoding(html)
    if encoding is None:
        raise ValueError("Could not detect encoding")
    try:
        html_str = html.decode(encoding)
        return extract_plain_text(html_str)
    
    except Exception as e:
        try:
            result = chardet.detect(html[:10000]) 
            if result['encoding'] is not None:
                html_str = html.decode(result['encoding'])
                return extract_plain_text(html_str)
            else:
                print(f"Error decoding HTML: {result}")
                return ""
        except Exception as e:
            print(f"Error decoding HTML: {e}")
            return ""

def warc_to_txt(warc_file: str, n_records: int = 10, record_id: int = 0):
    stream = GZipStream(FileStream(warc_file, 'rb'))
    for i, record in enumerate(ArchiveIterator(stream, record_types=WarcRecordType.response)):
        if i < record_id:
            continue
        if n_records <= 0:
            break
        n_records -= 1
        
        yield html_to_txt(record.reader.read())

class PIIFilter():
    def __init__(self):
        self.email_regex = r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
        self.phone_regex = r"(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"
        self.ipv4_regex = r"(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])"

        # compile regexes for speedup
        self.email_regex = re.compile(self.email_regex, re.IGNORECASE)
        self.phone_regex = re.compile(self.phone_regex)
        self.ipv4_regex = re.compile(self.ipv4_regex)

    def mask_emails(self, text: str) -> str:
        masked_text, count = self.email_regex.subn("|||EMAIL_ADDRESS|||", text)
        return masked_text, count

    def mask_phone_numbers(self, text: str) -> str:
        masked_text, count = self.phone_regex.subn("|||PHONE_NUMBER|||", text)
        return masked_text, count

    def mask_ips(self, text: str) -> str:
        masked_text, count = self.ipv4_regex.subn("|||IP_ADDRESS|||", text)
        return masked_text, count

def load_classifier(classifier_id: str) -> fasttext.FastText:
    classifier_path = os.path.join(BASE_DIR, classifier_id)
    return fasttext.load_model(classifier_path)

def filter_fasttext(text: str, classifier: fasttext.FastText) -> str:
    # strip newlines
    text = text.replace("\n", " ")
    prediction = classifier.predict(text)

    label = prediction[0][0].replace("__label__", "")
    confidence = prediction[1][0]
    return label, confidence

class QualityFilter():
    def __init__(self, classifier_id: str = QUALITY_FILTER):
        self.classifier = load_classifier(classifier_id)

    def filter_quality(self, text: str) -> str:
        return filter_fasttext(text, self.classifier)

class LanguageDetector():
    def __init__(self, classifier_id: str = LANGUAGE_FILTER):
        self.classifier = load_classifier(classifier_id)

    def detect_language(self, text: str) -> Tuple[str, float]:
        return filter_fasttext(text, self.classifier)

class NSFWDetector():
    def __init__(self, classifier_id: str = NSFW_FILTER):
        self.classifier = load_classifier(classifier_id)

    def filter_nsfw(self, text: str) -> str:
        return filter_fasttext(text, self.classifier)

class ToxicDetector():
    def __init__(self, classifier_id: str = TOXIC_FILTER):
        self.classifier = load_classifier(classifier_id)

    def filter_toxic(self, text: str) -> str:
        return filter_fasttext(text, self.classifier)

if __name__ == "__main__":
    test_task = "html_to_txt"
    # WARC_path = os.path.join(BASE_DIR, "CC/example.warc.gz")
    # WARC_path = os.path.join(BASE_DIR, "batch_0000.warc.gz")
    WARC_path = "/data/CC/example.warc.gz"
    language_detector = LanguageDetector()
    nsfw_detector = NSFWDetector()
    toxic_detector = ToxicDetector()
    piifilter = PIIFilter()

    if test_task == "html_to_txt":
        for txt in warc_to_txt(WARC_path, n_records = 1, record_id = 2000):
            print(txt)
            print('content length: ', len(txt))
    
    elif test_task == "detect_language":
        id = random.randint(0, 10000)
        for txt in warc_to_txt(WARC_path, n_records = 1, record_id = id):
            print(txt)
            print('Classified language:', language_detector.detect_language(txt))
    
    elif test_task == "filter_harmful":
        id = random.randint(0, 10000)
        for txt in warc_to_txt(WARC_path, n_records = 1, record_id = id):
            print(txt)
            print('Classified NSFW:', nsfw_detector.filter_nsfw(txt))
            print('Classified toxic:', toxic_detector.filter_toxic(txt))

    elif test_task == "mask":
        num_replacements = 0

        id = random.randint(1, 10000)
        for txt in warc_to_txt(WARC_path, n_records = 10, record_id = id):
            masked_text, email_count = piifilter.mask_emails(txt)
            masked_text, phone_count = piifilter.mask_phone_numbers(masked_text) 
            masked_text, ip_count = piifilter.mask_ips(masked_text)
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