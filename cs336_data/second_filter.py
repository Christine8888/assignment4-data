import os
import pathlib
import submitit
from tqdm import tqdm
from cs336_data.utils import QualityFilter
import json
import numpy as np
import math

VERBOSE = False
DEDUP = False

PALOMA_FILTER = "/home/c-cye/assignment4-data/cs336_data/paloma.bin"
PALOMA_PERCENTILE = 0.50

def process_single_txt_file(input_path: str, output_path: str):
    # load paloma filter
    paloma_filter = QualityFilter(classifier_id=PALOMA_FILTER)

    # load text file
    text = open(input_path, "r").read()
    docs = text.split("<|endoftext|>")
    docs = [doc.strip() for doc in docs if doc.strip()]
    
    # apply paloma filter
    labels, confs = paloma_filter.classifier.predict(docs)
    
    # keep all paloma and all non-paloma with confs under 50th percentile
    paloma_docs = [i for i, label in enumerate(labels) if label[0] == "__label__paloma"]
    paloma_percentile = np.percentile(confs, PALOMA_PERCENTILE)
    non_paloma_docs = [i for i, label in enumerate(labels) if label[0] != "__label__paloma" and confs[i][0] < paloma_percentile]
    filtered_docs = paloma_docs + non_paloma_docs
    
    # write to output file
    output_file = open(output_path, "w")

    for idx in filtered_docs:
        # write to output file
        output_file.write(docs[idx])
        output_file.write("<|endoftext|>")
    
    output_file.close()

def process_batch_of_txt_files(txt_filepaths_batch: list, output_directory_path: str, work_dir_base: str, batch_id: int):
    """Process a batch of txt files in a single job"""
    print(f"Starting batch {batch_id} with {len(txt_filepaths_batch)} files")
    
    results = []
    for txt_filepath in txt_filepaths_batch:
        try:
            txt_filename = str(pathlib.Path(txt_filepath).name)
            txt_filename = txt_filename.split('.')[0]
            
            output_path = os.path.join(output_directory_path, f"{txt_filename}_paloma.txt")
            
            # check if output file already exists
            if os.path.exists(output_path):
                print(f"Skipping {txt_filename} because it already exists")
                continue
            
            result = process_single_txt_file(txt_filepath, output_path)
            results.append(result)
            print(f"Batch {batch_id}: Completed {txt_filename}")
            
        except Exception as e:
            print(f"Batch {batch_id}: Error processing {txt_filepath}: {str(e)}")
            continue
    
    print(f"Batch {batch_id} completed with {len(results)} successful files")
    return results

def partition_list(lst, n):
    """Partition a list into n roughly equal chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

N_WORKERS = 128
txt_filepaths = json.loads(open("txtlist.json", "r").read())
N_FILES = len(txt_filepaths)
print(f"Found {N_FILES} text files")
N_FILES_PER_WORKER = math.ceil(N_FILES / N_WORKERS)
print(f"Will process {N_FILES_PER_WORKER} files per worker")

file_batches = partition_list(txt_filepaths, N_WORKERS)
output_directory_path = "/data/c-cye/assignment4-data/cc_filtered_paloma"

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
        process_batch_of_txt_files,
        file_batch,
        output_directory_path,
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