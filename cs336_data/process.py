"""
WARC to FastText training data converter.

This module reads text from WARC files and formats it for FastText training,
with configurable labels and optional line limits.
"""

import os
import glob
from typing import List, Optional, Union
from pathlib import Path
import random

from cs336_data.utils import warc_to_txt, LanguageDetector, NSFWDetector, ToxicDetector
from cs336_data.gopher import GopherFilter

bullet_point_characters = tuple(["*", "-", "•", "•"])


class WarcToFastTextConverter:
    """Convert WARC files to FastText training format."""
    
    def __init__(self, output_file: str, labels: List[str], max_lines: Optional[int] = None):
        """
        initialize the converter
        
        Args:
            output_file: Path to output text file
            labels: List of labels to apply to documents (without __label__ prefix)
            max_lines: Optional limit on number of lines to write
        """
        self.output_file = output_file
        self.labels = [f"__label__{label}" for label in labels]
        self.max_lines = max_lines
        self.lines_written = 0
        self.gopherfilter = GopherFilter()

        self.language_detector = LanguageDetector()
        self.nsfw_detector = NSFWDetector()
        self.toxic_detector = ToxicDetector()
    
    def _filter_text(self, text: str, threshold: float = 0.6) -> str:
        # apply language, gopher, and quality filters
        lang, langconf = self.language_detector.detect_language(text)
        nsfw, nsfwconf = self.nsfw_detector.filter_nsfw(text)
        toxic, toxicconf = self.toxic_detector.filter_toxic(text)
        gopher = self.gopherfilter.filter(text)

        if lang != "en" or langconf < threshold:
            # print(f"language filter failed: {lang} with confidence {langconf}")
            return False

        if nsfw != "non-nsfw" or nsfwconf < threshold:
            # print(f"nsfw filter failed: {nsfw} with confidence {nsfwconf}")
            return False

        if toxic != "non-toxic" or toxicconf < threshold:
            # print(f"toxic filter failed: {toxic} with confidence {toxicconf}")
            return False

        return gopher

    def _clean_content(self, text: str) -> str:
        # remove all lines starting with bullet points
        text = "\n".join([line for line in text.split("\n") if not line.strip().startswith(bullet_point_characters)])
        return text
        
    def _format_line(self, text: str, clean_content: bool = False) -> str:
        """format a single line for FastText training."""

        if clean_content:
            text = self._clean_content(text)

        # remove newlines and extra whitespace
        cleaned_text = " ".join(text.split())
        if len(cleaned_text) < 10:
            return ""
        
        # combine labels with text
        final_line = ""
        for label in self.labels:
            final_line += f"{label} "
        
        final_line += f"{cleaned_text}\n"

        return final_line
    
    def _should_continue(self) -> bool:
        """check for line limit."""
        if self.max_lines is None:
            return True
        
        return self.lines_written < self.max_lines
    
    def process_warc_file(self, warc_file: str, n_records: Optional[int] = None, 
                         sample: bool = False, filter_content: bool = True, clean_content: bool = False) -> int:
        """
        Process a single WARC file.
        
        Args:
            warc_file: Path to WARC file
            n_records: Optional limit on records to process from this file
            sample: Whether to randomly sample records from the WARC file
            filter_content: Whether to apply filters to the text
            clean_content: Whether to clean the content of the text
            
        Returns:
            Number of lines written from this file
        """
        lines_from_file = 0
        
        
        if sample:
            # count total records
            total_records = sum(1 for _ in warc_to_txt(warc_file, n_records = 1))
            indices = random.sample(range(0, total_records), n_records * 10)
        else:
            indices = []

        for i, txt in enumerate(warc_to_txt(warc_file, n_records = float('inf'))):
            if i % 50 == 0:
                print(f"processing record {i} of {n_records} from {warc_file}")
            
            if not self._should_continue(): break

            if sample and i not in indices: continue
            
            try:
                if filter_content and not self._filter_text(txt): continue
                
                # format and write the line
                formatted_line = self._format_line(txt, clean_content)
                if not formatted_line: continue
                
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    f.write(formatted_line)
                
                self.lines_written += 1
                lines_from_file += 1
                
                if self.lines_written % 1000 == 0:
                    print(f"Processed {self.lines_written} lines...")
                        
            except Exception as e:
                print(f"Error processing {warc_file}: {e}")
            
        return lines_from_file
    
    def process_warc_files(self, warc_path: Union[str, List[str]], 
                          n_records: Optional[int] = None,
                          sample: bool = False,
                          filter_content: bool = True,
                          clean_content: bool = False) -> None:
        """
        Process WARC files from a path or list of paths.
        
        Args:
            warc_path: Single file path, directory path, or list of file paths
            n_records: Optional limit on records per file
            sample: Whether to randomly sample records from the WARC file
            filter_content: Whether to apply filters to the text
        """
        
        # get list of WARC files to process
        if isinstance(warc_path, list):
            warc_files = warc_path
        elif os.path.isfile(warc_path):
            warc_files = [warc_path]
        elif os.path.isdir(warc_path):
            # find all WARC files in directory
            print('finding warc files in directory: ', warc_path)
            warc_files = []
            for pattern in ['**/*.warc', '**/*.warc.gz']:
                warc_files.extend(glob.glob(os.path.join(warc_path, pattern), recursive=True))
            warc_files.sort()
        else:
            raise ValueError(f"Invalid warc_path: {warc_path}")
        
        if not warc_files:
            raise ValueError(f"No WARC files found at: {warc_path}")
        
        print(f"Found {len(warc_files)} WARC files to process")
        print(f"Writing to: {self.output_file}")
        print(f"Labels: {self.labels}")
        if self.max_lines:
            print(f"Max lines: {self.max_lines}")
        if sample:
            print(f"Sampling {n_records} records per file")
        
        # process each file
        for i, warc_file in enumerate(warc_files):
            if not self._should_continue():
                break
                
            print(f"processing file {i+1}/{len(warc_files)}: {os.path.basename(warc_file)}")
            lines_from_file = self.process_warc_file(
                warc_file, 
                n_records=n_records,
                sample=sample,
                filter_content=filter_content,
                clean_content=clean_content
            )
            print(f"  -> {lines_from_file} lines written")
        
        print(f"conversion complete! Total lines written: {self.lines_written}")


def convert_warc_to_fasttext(warc_path: Union[str, List[str]], 
                           output_file: str,
                           labels: List[str],
                           max_lines: Optional[int] = None,
                           records_per_file: Optional[int] = None,
                           sample: bool = False,
                           filter_content: bool = True,
                           clean_content: bool = False) -> None:
    """
    convenience function to convert WARC files to FastText format.
    
    args:
        warc_path: Single file path, directory path, or list of file paths  
        output_file: Path to output text file
        labels: List of labels to apply (without __label__ prefix)
        max_lines: Optional limit on total lines to write
        records_per_file: Optional limit on records per WARC file
        sample: Whether to randomly sample records from the WARC file
    """
    converter = WarcToFastTextConverter(output_file, labels, max_lines)
    converter.process_warc_files(warc_path = warc_path, n_records = records_per_file, sample = sample, filter_content = filter_content, clean_content = clean_content)


if __name__ == "__main__":
    # procss CC for negative examples
    # convert_warc_to_fasttext(
    #     warc_path="/data/CC/example.warc.gz",
    #     output_file="negative_data.txt", 
    #     labels=["low-quality"],
    #     max_lines = 35000,
    #     filter_content = False
    # )

    # process quality data for positive examples
    convert_warc_to_fasttext(
        warc_path="/data/c-cye/assignment4-data/quality_text_4",
        output_file="positive_data_cleaner.txt", 
        labels=["high-quality"],
        filter_content = True,
        max_lines = 15000,
        clean_content = True
    )