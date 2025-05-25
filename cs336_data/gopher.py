"""Gopher quality filters from Rae et al. 2021"""

import nltk.tokenize

class GopherFilter():
    bullet_point_characters = ["*", "-", "•"]
    stop_words = ["the", "be", "to", "of", "and", "that", "have", "with"]

    def __init__(self, filter_length: bool = True, 
                 filter_mean_length: bool = True, 
                 filter_ellipsis: bool = True,
                 filter_word_alphabet: bool = True,
                 filter_bullet_point: bool = True,
                 filter_stop_word: bool = True,
                 verbose: bool = False):

        self.filter_length = filter_length
        self.filter_mean_length = filter_mean_length
        self.filter_ellipsis = filter_ellipsis
        self.filter_word_alphabet = filter_word_alphabet
        self.filter_bullet_point = filter_bullet_point
        self.filter_stop_word = filter_stop_word
        self.verbose = verbose

    def filter(self, text: str) -> str:
        # clean whitespace from text
        text = text.strip()

        tokenized = self.tokenize(text)
        # remove whitespace tokens
        tokenized = [token for token in tokenized if token.strip()]
        # remove bullet point tokens
        tokenized = [token for token in tokenized if token not in GopherFilter.bullet_point_characters]

        splitlines = text.split("\n")
        # strip whitespace from splitlines
        splitlines = [line.strip() for line in splitlines]

        if self.filter_mean_length:
            if not self.mean_length_filter(tokenized):
                if self.verbose: print("mean length filter failed")
                return False

        if self.filter_ellipsis:
            if not self.ellipsis_filter(splitlines):
                if self.verbose: print("ellipsis filter failed")
                return False

        if self.filter_word_alphabet:
            if not self.word_alphabet_filter(tokenized):
                if self.verbose: print("word alphabet filter failed")
                return False
        
        if self.filter_bullet_point:
            if not self.bullet_point_filter(splitlines):
                if self.verbose: print("bullet point filter failed")
                return False
        
        if self.filter_stop_word:
            if not self.stop_word_filter(tokenized):
                if self.verbose: print("stop word filter failed")
                return False

        return True

    @staticmethod
    def word_alphabet_filter(tokenized: list[str]) -> bool:
        # 80% of words must have at least one alphabetic character
        alpha_count = sum(1 for word in tokenized if any(c.isalpha() for c in word))
        print(alpha_count, len(tokenized))
        alpha_frac = alpha_count / len(tokenized)
        return alpha_frac >= 0.8 and alpha_count >= 50 and alpha_count <= 100000
    
    @staticmethod
    def tokenize(text: str) -> list[str]:
        return nltk.tokenize.word_tokenize(text)

    @staticmethod
    def length_filter(tokenized: list[str]) -> bool:
        return len(tokenized) >= 50 and len(tokenized) <= 100000

    @staticmethod
    def mean_length_filter(tokenized: list[str]) -> bool:
        # compute mean word length
        mean_length = sum(len(word) for word in tokenized) / len(tokenized)
        return mean_length >= 3 and mean_length <= 10
    
    @staticmethod
    def ellipsis_filter(splitlines: list[str]) -> bool:
        ellipsis_count = sum(1 for line in splitlines if line.endswith("..."))
        return (ellipsis_count / len(splitlines)) <= 0.3
    
    @staticmethod
    def bullet_point_filter(tokenized: list[str]) -> bool:
        # count number of bullet points
        bullet_point_count = sum(1 for line in tokenized if line.startswith(tuple(GopherFilter.bullet_point_characters)))
        return (bullet_point_count / len(tokenized)) <= 0.9

    @staticmethod
    def stop_word_filter(tokenized: list[str]) -> bool:
        # count number of stop words
        stop_word_count = sum(1 for word in tokenized if word in GopherFilter.stop_words)
        return stop_word_count >= 2