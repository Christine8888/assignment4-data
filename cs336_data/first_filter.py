import os
import pathlib
import submitit
from tqdm import tqdm
import glob
from fastwarc.warc import WarcRecordType, ArchiveIterator
from fastwarc.stream_io import GZipStream, FileStream
from cs336_data.utils import html_to_txt, LanguageDetector, QualityFilter, NSFWDetector, ToxicDetector, PIIFilter
from cs336_data.gopher import GopherFilter
from cs336_data.dedup import MinHashDedup
import json
import nltk
import math

VERBOSE = False
DEDUP = False

LANGUAGE_THRESHOLD = 0.5
NSFW_THRESHOLD = 0.5
TOXIC_THRESHOLD = 0.5
QUALITY_THRESHOLD = 0.6

NSFW_FILTER = "/data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin"
TOXIC_FILTER = "/data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin"
LANGUAGE_FILTER = "/data/classifiers/lid.176.bin"
QUALITY_FILTER = "/home/c-cye/assignment4-data/cs336_data/quality_classifier.bin"

def process_single_wet_file(input_path: str, output_path: str, work_dir: str):
    # load filters
    language_detector = LanguageDetector(LANGUAGE_FILTER)
    quality_filter = QualityFilter(QUALITY_FILTER)
    nsfw_detector = NSFWDetector(NSFW_FILTER)
    toxic_detector = ToxicDetector(TOXIC_FILTER)
    gopher_filter = GopherFilter(verbose=VERBOSE)
    dedup = MinHashDedup()

    print('Loaded filters successfully')

    # set up stats
    stats = {
        'total_records': 0,
        'after_language_filter': 0,
        'after_gopher_filter': 0,
        'after_nsfw_filter': 0,
        'after_toxic_filter': 0,
        'after_quality_filter': 0,
        'after_dedup': 0
    }
    os.makedirs(work_dir, exist_ok=True)

    # load WET file stream
    warc_file = input_path
    stream = GZipStream(FileStream(warc_file, 'rb'))
    wet_iterable = ArchiveIterator(stream)

    # iterate over records
    filelist = []
    output_file = open(output_path, "w")
    for i, record in enumerate(wet_iterable):
        if i % 100 == 0:
            print(f"Processing record {i}")
            print(f"Stats: {stats}")
        
        # check record type
        if record.record_type not in [WarcRecordType.conversion, WarcRecordType.response]:
            continue
        
        stats['total_records'] += 1

        # always decode as utf-8
        try:
            text = record.reader.read().decode('utf-8')
        except Exception as e:
            print(f"Error decoding record {i}: {e}")
            continue
            
        if VERBOSE:
            print(f"FULL TEXT\n{text}\n")

        # filter on language
        language, langconf = language_detector.detect_language(text)
        if VERBOSE: print(f"Language: {language} with confidence {langconf}")
        if language != "en" or langconf < LANGUAGE_THRESHOLD: continue
        stats['after_language_filter'] += 1

        # filter with gopher
        gopher = gopher_filter.filter(text)
        if VERBOSE: print(f"Gopher: {gopher}")
        if gopher != True: continue
        stats['after_gopher_filter'] += 1

        # filter on nsfw
        nsfw, nsfw_conf = nsfw_detector.filter_nsfw(text)
        if VERBOSE: print(f"NSFW: {nsfw} with confidence {nsfw_conf}")
        if nsfw != "non-nsfw" or nsfw_conf < NSFW_THRESHOLD: continue
        stats['after_nsfw_filter'] += 1

        # filter on toxic
        toxic, toxic_conf = toxic_detector.filter_toxic(text)
        if VERBOSE: print(f"Toxic: {toxic} with confidence {toxic_conf}")
        if toxic != "non-toxic" or toxic_conf < TOXIC_THRESHOLD: continue
        stats['after_toxic_filter'] += 1

        # filter on quality
        quality, quality_conf = quality_filter.filter_quality(text)
        if VERBOSE: print(f"Quality: {quality} with confidence {quality_conf}")
        if quality == "high-quality" or quality_conf < QUALITY_THRESHOLD:
            # allow both high-quality and low-quality with low-confidence
            stats['after_quality_filter'] += 1
        else:
            continue

        # delete empty or short lines from text
        text = "\n".join([line for line in text.split("\n") if line.strip() and len(nltk.tokenize.word_tokenize(line.strip())) > 4])
        if VERBOSE: print(f"AFTER FILTERING\n{text}\n")

        # save text to file in working directory
        if DEDUP:
            print(f"Saving text to file {i}")
            with open(os.path.join(work_dir, f"{i}.txt"), "w") as f:
                f.write(text)
                # full path to file
                filelist.append(os.path.join(work_dir, f"{i}.txt"))
        else:
            output_file.write(text)
            output_file.write("<|endoftext|>")
            output_file.write("\n") 
            stats['after_dedup'] += 1
    
    if DEDUP:
        # deduplicate in working directory
        dedup_dir = os.path.join(work_dir, "dedup")
        os.makedirs(dedup_dir, exist_ok=True)
        dedup.minhash_dedup(filelist, dedup_dir)
        
        # count files after deduplication
        dedup_files = os.listdir(dedup_dir)
        stats['after_dedup'] = len(dedup_files)
    
        # write to output file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        for file in dedup_files:
            with open(os.path.join(dedup_dir, file), "r") as in_f:
                output_file.write(in_f.read())
                output_file.write("<|endoftext|>")

    stats_path = output_path.replace(".txt", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f)
    
    print(stats)
    output_file.close()
    return output_path

def process_batch_of_wet_files(wet_filepaths_batch: list, output_directory_path: str, work_dir_base: str, batch_id: int):
    """Process a batch of WET files in a single job"""
    print(f"Starting batch {batch_id} with {len(wet_filepaths_batch)} files")
    
    results = []
    for wet_filepath in wet_filepaths_batch:
        try:
            wet_filename = str(pathlib.Path(wet_filepath).name)
            wet_filename = wet_filename.split('.')[0]
            
            output_path = os.path.join(output_directory_path, f"{wet_filename}.txt")
            work_dir = os.path.join(work_dir_base, f"batch_{batch_id}", f"{wet_filename}_work")
            
            # check if output file already exists
            if os.path.exists(output_path):
                print(f"Skipping {wet_filename} because it already exists")
                continue
            
            result = process_single_wet_file(wet_filepath, output_path, work_dir)
            results.append(result)
            print(f"Batch {batch_id}: Completed {wet_filename}")
            
        except Exception as e:
            print(f"Batch {batch_id}: Error processing {wet_filepath}: {str(e)}")
            continue
    
    print(f"Batch {batch_id} completed with {len(results)} successful files")
    return results

def partition_list(lst, n):
    """Partition a list into n roughly equal chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

N_WORKERS = 128
wet_filepaths = json.loads(open("wetlist.json", "r").read())
N_FILES = len(wet_filepaths)
print(f"Found {N_FILES} files")
N_FILES_PER_WORKER = math.ceil(N_FILES / N_WORKERS)
print(f"Will process {N_FILES_PER_WORKER} files per worker")

file_batches = partition_list(wet_filepaths, N_WORKERS)

output_directory_path = "/data/c-cye/assignment4-data/cc_filtered"
work_dir = "/data/c-cye/assignment4-data/cc_filtered_work"

# set up the submitit executor
executor = submitit.AutoExecutor(folder="slurm_logs")
executor.update_parameters(
    slurm_array_parallelism=N_WORKERS,  
    timeout_min = 30,           
    mem_gb = 4,             
    cpus_per_task = 1,
    slurm_account="student",
    slurm_partition="a4-cpu", 
    slurm_qos="a4-cpu-qos",
)

# submit jobs   
print(f"Submitting {len(file_batches)} batch jobs...")
futures = []

for batch_id, file_batch in enumerate(file_batches):
    print(f"Submitting batch {batch_id} with {len(file_batch)} files")
    future = executor.submit(
        process_batch_of_wet_files,
        file_batch,
        output_directory_path,
        work_dir,
        batch_id
    )
    futures.append(future)

# monitor progress
print("Monitoring job progress...")
completed_batches = 0
for future in tqdm(submitit.helpers.as_completed(futures), total=len(file_batches)):
    try:
        result = future.result()
        completed_batches += 1
        print(f"Batch completed ({completed_batches}/{len(file_batches)}). Processed {len(result)} files successfully.")
    except Exception as e:
        print(f"Batch failed with error: {str(e)}")

print(f"All jobs completed! {completed_batches}/{len(file_batches)} batches finished successfully.")