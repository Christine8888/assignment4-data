"""
Parallel URL scraper with wget/submitit.
"""

import argparse
import gzip
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import List

try:
    import submitit
    SUBMITIT_AVAILABLE = True
except ImportError:
    SUBMITIT_AVAILABLE = False
    print("Warning: submitit not available. Falling back to local multiprocessing.")


def sample_urls(input_file: str, n_samples: int, output_file: str) -> None:
    """Sample n_samples URLs from the input file."""
    print(f"Sampling {n_samples} URLs from {input_file}")
    
    # count total lines
    if input_file.endswith('.gz'):
        with gzip.open(input_file, 'rt', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
    
    print(f"Total URLs available: {total_lines}")
    
    if n_samples > total_lines:
        print(f"Warning: Requested {n_samples} samples but only {total_lines} available")
        n_samples = total_lines
    
    # sample random line indices
    sampled_indices = set(random.sample(range(total_lines), n_samples))
    
    # extract sampled URLs
    sampled_urls = []
    if input_file.endswith('.gz'):
        with gzip.open(input_file, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i in sampled_indices:
                    sampled_urls.append(line.strip())
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i in sampled_indices:
                    sampled_urls.append(line.strip())
    
    # write sampled URLs
    with open(output_file, 'w', encoding='utf-8') as f:
        for url in sampled_urls:
            f.write(f"{url}\n")
    
    print(f"Sampled {len(sampled_urls)} URLs to {output_file}")


def split_urls_into_batches(url_file: str, batch_size: int, output_dir: str) -> List[str]:
    """Split URLs into batches and return list of batch files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    batch_files = []
    
    with open(url_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    print(f"Splitting {len(urls)} URLs into batches of {batch_size}")
    
    for i in range(0, len(urls), batch_size):
        batch_num = i // batch_size
        batch_file = output_path / f"batch_{batch_num:04d}.txt"
        batch_urls = urls[i:i + batch_size]
        
        with open(batch_file, 'w', encoding='utf-8') as f:
            for url in batch_urls:
                f.write(f"{url}\n")
        
        batch_files.append(str(batch_file))
        print(f"Created batch {batch_num}: {len(batch_urls)} URLs -> {batch_file}")
    
    return batch_files


def download_batch(batch_file: str, output_dir: str, timeout: int = 5) -> str:
    """Download a batch of URLs using wget."""
    batch_path = Path(batch_file)
    batch_name = batch_path.stem
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    warc_file = output_path / f"{batch_name}.warc"
    log_file = output_path / f"{batch_name}.log"
    
    # wget command
    cmd = [
        'wget',
        f'--timeout={timeout}',
        '--tries=3',
        '--waitretry=1',
        '--random-wait',
        '--user-agent=Mozilla/5.0 (compatible; research bot)',
        '-i', str(batch_file),
        f'--warc-file={warc_file.with_suffix("")}',  # wget adds .warc.gz
        '-O', '/dev/null',
        '--no-verbose'
    ]
    
    print(f"Starting download for {batch_file}")
    
    try:
        with open(log_file, 'w') as log:
            result = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=3600  # 1 hour timeout for the entire batch
            )
        
        if result.returncode == 0:
            status = f"SUCCESS: {batch_file} -> {warc_file}.gz"
        else:
            status = f"PARTIAL: {batch_file} completed with errors (code {result.returncode})"
        
    except subprocess.TimeoutExpired:
        status = f"TIMEOUT: {batch_file} timed out after 1 hour"
    except Exception as e:
        status = f"ERROR: {batch_file} failed with exception: {e}"
    
    print(status)
    return status


def process_batch_chunk(batch_files_chunk: List[str], output_dir: str, n_workers: int = 4) -> List[str]:
    """Process a chunk of batch files using local multiprocessing on a single node."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    node_id = os.environ.get('SLURM_PROCID', 'local')
    print(f"Node {node_id}: Processing {len(batch_files_chunk)} batches with {n_workers} workers")
    
    results = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # submit all jobs for this chunk
        future_to_batch = {
            executor.submit(download_batch, batch_file, output_dir): batch_file 
            for batch_file in batch_files_chunk
        }
        
        # collect results as they complete
        for future in as_completed(future_to_batch):
            batch_file = future_to_batch[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Node {node_id}: Completed {batch_file}")
            except Exception as e:
                error_msg = f"ERROR: {batch_file} failed with {e}"
                results.append(error_msg)
                print(f"Node {node_id}: {error_msg}")
    
    print(f"Node {node_id}: Completed all {len(batch_files_chunk)} batches")
    return results


def distribute_batches_across_jobs(batch_files: List[str], n_slurm_jobs: int) -> List[List[str]]:
    """Distribute batch files evenly across SLURM jobs."""
    chunks = [[] for _ in range(n_slurm_jobs)]
    
    # round-robin distribution to ensure even load
    for i, batch_file in enumerate(batch_files):
        chunks[i % n_slurm_jobs].append(batch_file)
    
    # print distribution summary
    for i, chunk in enumerate(chunks):
        print(f"SLURM job {i}: {len(chunk)} batches")
    
    return chunks


def run_with_submitit(batch_files: List[str], 
                    output_dir: str, 
                     slurm_partition: str = "a4-cpu", 
                     slurm_time: int = 30,
                     slurm_mem: str = "16GB",
                     slurm_qos: str = "a4-cpu-qos",
                     n_slurm_jobs: int = 4,
                     local_workers: int = 4) -> None:
    """Run downloads using submitit on SLURM with fixed number of jobs."""
    if not SUBMITIT_AVAILABLE:
        raise ImportError("submitit is required for SLURM execution")
    
    # distribute batches across SLURM jobs
    batch_chunks = distribute_batches_across_jobs(batch_files, n_slurm_jobs)
    print(f"Distributed {len(batch_files)} batches across {n_slurm_jobs} SLURM jobs")
    
    # setup submitit executor
    executor = submitit.AutoExecutor(folder=f"{output_dir}/submitit_logs")
    executor.update_parameters(
        partition=slurm_partition,
        time=slurm_time,  # minutes
        mem=slurm_mem,
        job_name="url_scraper",
        qos=slurm_qos,
        account="student",
    )
    
    print(f"Submitting {n_slurm_jobs} SLURM jobs to partition '{slurm_partition}'")
    print(f"Each job will use {local_workers} local workers")
    
    # submit all jobs
    jobs = []
    for i, batch_chunk in enumerate(batch_chunks):
        if not batch_chunk:  # skip empty chunks
            continue
            
        job = executor.submit(process_batch_chunk, batch_chunk, output_dir, local_workers)
        jobs.append(job)
        print(f"Submitted SLURM job {job.job_id} for chunk {i} ({len(batch_chunk)} batches)")
    
    # wait for completion and collect results
    print(f"Waiting for {len(jobs)} SLURM jobs to complete...")
    all_results = []
    for i, job in enumerate(jobs):
        try:
            results = job.result()  # this blocks until job completes
            all_results.extend(results)
            print(f"SLURM job {job.job_id} completed successfully")
        except Exception as e:
            error_msg = f"SLURM JOB ERROR: {job.job_id} failed with {e}"
            all_results.append(error_msg)
            print(error_msg)
    
    # print summary
    success_count = sum(1 for r in all_results if r.startswith("SUCCESS"))
    partial_count = sum(1 for r in all_results if r.startswith("PARTIAL"))
    error_count = len(all_results) - success_count - partial_count
    
    print(f"\nSUMMARY:")
    print(f"  Successful batches: {success_count}")
    print(f"  Partial batches: {partial_count}")
    print(f"  Failed batches: {error_count}")
    print(f"  Total batches: {len(batch_files)}")


def run_with_local_multiprocessing(batch_files: List[str], output_dir: str, 
                                 n_workers: int = 4) -> None:
    """Run downloads using local multiprocessing."""
    results = process_batch_chunk(batch_files, output_dir, n_workers)
    
    # print summary
    success_count = sum(1 for r in results if r.startswith("SUCCESS"))
    partial_count = sum(1 for r in results if r.startswith("PARTIAL"))
    error_count = len(results) - success_count - partial_count
    
    print(f"\nSUMMARY:")
    print(f"  Successful batches: {success_count}")
    print(f"  Partial batches: {partial_count}")
    print(f"  Failed batches: {error_count}")
    print(f"  Total batches: {len(batch_files)}")