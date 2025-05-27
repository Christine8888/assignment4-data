
import hashlib
import os
import re
from unicodedata import normalize
import mmh3
from collections import defaultdict
import shutil
import random

def exact_dedup(file_list, output_dir, hash_func = hashlib.md5):
    """
    deduplicate a list of files by exact match.
    
    file_list: list of files to deduplicate
    output_dir: directory to save deduplicated files
    hash_func: function to hash lines, defaults to md5
    """
    line_counts = {}
    
    # count lines
    for file in file_list:
        with open(file, 'r') as f:
            for line in f:
                line_hash = hash_func(line.encode()).hexdigest()
                line_counts[line_hash] = line_counts.get(line_hash, 0) + 1
    
    for file in file_list:
        with open(file, 'r') as f_in:
            output_lines = []
            for line in f_in:
                line_hash = hash_func(line.encode()).hexdigest()
                
                if line_counts[line_hash] == 1:
                    # keep only unique lines
                    output_lines.append(line)
            
            # save deduplicated lines to file
            with open(os.path.join(output_dir, os.path.basename(file)), 'w') as f_out:
                f_out.write(''.join(output_lines)) # newlines are already in the file (?)

class MinHashDedup():
    def __init__(self, num_hashes = 100, num_bands = 10, ngrams = 3, jaccard_threshold = 0.5, verbose = False):
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.ngrams = ngrams
        self.jaccard_threshold = jaccard_threshold
        self.hashes = [i for i in range(self.num_hashes)] # mmh3 seeds
        self.verbose = verbose

    def minhash_dedup(self, files: list[os.PathLike], output_directory: os.PathLike):
        # data structure
        band_size = self.num_hashes // self.num_bands
        hashlist = {}
        for band in range(self.num_bands):
            hashlist[band] = {}
        candidate_duplicates = set() # store (band, band_hash) pairs

        # hash files and count duplicates
        for file in files:
            with open(file, 'r') as f:
                text = f.read()
            text = self.normalize_text(text)
            minhashes = self._minhash(text)

            # split minhashes into bands, use strings as keys
            for i in range(self.num_bands):
                band = minhashes[i * band_size: (i + 1) * band_size]
                bandstr = "".join(str(h) for h in band)

                if bandstr in hashlist[i]:
                    # candidate duplicate: matching in some band
                    candidate_duplicates.add((i, bandstr))
                    hashlist[i][bandstr].append(file)
                else:
                    hashlist[i][bandstr] = [file]

        # construct pairs from buckets
        # deduplicate by pairs
        clusters = []
        for band, band_hash in candidate_duplicates:
            candidate_files = hashlist[band][band_hash]
            for i, file1 in enumerate(candidate_files):
                for j in range(i + 1, len(candidate_files)):
                    file2 = candidate_files[j]
                    if self._jaccard(file1, file2) >= self.jaccard_threshold:
                        if self.verbose:
                            print('Found duplicate: ', file1, file2)
                        clusters.append((file1, file2))
        
        # save deduplicated files
        duplicate_groups = self._merge_clusters(clusters)
        self._save_deduplicated_files(files, output_directory, duplicate_groups)

    def _merge_clusters(self, clusters):
        all_files = set()
        uf = UnionFind()
        for file1, file2 in clusters:
            uf.union(file1, file2)
            all_files.add(file1)
            all_files.add(file2)

        # group into clusters
        final_clusters = defaultdict(list)
        for file in all_files:
            root = uf.find(file)
            final_clusters[root].append(file)

        duplicate_groups = [group for group in final_clusters.values() if len(group) > 1]
        print('Found', len(duplicate_groups), 'duplicate clusters')
        return duplicate_groups
    
    def _save_deduplicated_files(self, files: list[os.PathLike], output_directory: os.PathLike, duplicate_groups: list[list[os.PathLike]]):
        files_to_skip = set()
        output_abs = os.path.abspath(output_directory)
        
        for group in duplicate_groups:
            files_to_skip.update(group)
            print(f'Removing {len(group)} duplicate files')
            keep_file = random.choice(group)
            shutil.copy(keep_file, os.path.join(output_directory, os.path.basename(keep_file)))

        # copy all files to output directory
        for file in files:
            if file not in files_to_skip:
                shutil.copy(file, os.path.join(output_directory, os.path.basename(file)))
        
    def _jaccard(self, file1, file2):
        # read files
        with open(file1, 'r') as f1:
            text1 = f1.read()
            text1 = self.normalize_text(text1)
        with open(file2, 'r') as f2:
            text2 = f2.read()
            text2 = self.normalize_text(text2)
        
        # split into ngrams
        ngrams1 = [text1[i: i + self.ngrams] for i in range(len(text1) - self.ngrams + 1)]
        ngrams2 = [text2[i: i + self.ngrams] for i in range(len(text2) - self.ngrams + 1)]
        # count ngram intersections
        intersection = len(set(ngrams1) & set(ngrams2))
        return intersection / len(set(ngrams1) | set(ngrams2))

    def _minhash(self, text: str):
        # split text into ngrams
        ngrams = [text[i:i+self.ngrams] for i in range(len(text) - self.ngrams + 1)]
        minhashes = []
        for seed in self.hashes:
            hashes = [mmh3.hash(ngram, seed) for ngram in ngrams]
            minhashes.append(min(hashes))
        
        return minhashes
        
    @staticmethod
    def normalize_text(text):
        # lowercase
        text = text.lower()
        # replace newlines, tabs, etc. with spaces
        text = re.sub(r'\s+', ' ', text)
        # remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # remove accents, apply nfd unicode normalization
        text = normalize('NFD', text)
        return text

class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        
        # union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1