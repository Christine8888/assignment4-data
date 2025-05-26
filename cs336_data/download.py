from cs336_data.url import sample_urls, split_urls_into_batches, run_with_submitit, run_with_local_multiprocessing
import argparse
import random
from pathlib import Path
import sys

try:
    import submitit
    SUBMITIT_AVAILABLE = True
except ImportError:
    SUBMITIT_AVAILABLE = False
    print("Warning: submitit not available. Falling back to local multiprocessing.")

def main():
    parser = argparse.ArgumentParser(description="Parallel Wikipedia URL scraper")
    parser.add_argument("input_file", help="Input file with URLs (can be .gz)")
    parser.add_argument("--output-dir", "-o", required=True, 
                       help="Output directory for downloaded files")
    parser.add_argument("--n-samples", "-n", type=int, default=100000,
                       help="Number of URLs to sample (default: 100k)")
    parser.add_argument("--batch-size", "-b", type=int, default=1000,
                       help="URLs per batch (default: 1000)")
    parser.add_argument("--use-slurm", action="store_true",
                       help="Use SLURM via submitit (default: local multiprocessing)")
    parser.add_argument("--slurm-partition", default="a4-cpu",
                       help="SLURM partition (default: a4-cpu)")
    parser.add_argument("--slurm-qos", default="a4-cpu-qos",
                       help="SLURM QOS (default: a4-cpu-qos)")
    parser.add_argument("--slurm-time", type=int, default=30,
                       help="SLURM time limit in minutes (default: 30)")
    parser.add_argument("--slurm-mem", default="16GB",
                       help="SLURM memory per job (default: 16GB)")
    parser.add_argument("--n-slurm-jobs", type=int, default=4,
                       help="Number of SLURM jobs to launch (default: 4)")
    parser.add_argument("--local-workers", type=int, default=4,
                       help="Number of local workers per SLURM job (default: 4)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for sampling (default: None)")
    
    args = parser.parse_args()
    
    # set random seed
    if args.seed is not None:   
        random.seed(args.seed)
    
    # create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # sample URLs
    sampled_urls_file = output_path / "sampled_positive_urls.txt" # this is the file that contains the sampled URLs
    batch_dir = output_path / "batch_files"
    
    if sampled_urls_file.exists():
        print(f"Sampled URLs file {sampled_urls_file} already exists. Skipping sampling.")
    else:
        print(f"Sampled URLs file {sampled_urls_file} does not exist. Sampling URLs.")
        sample_urls(args.input_file, args.n_samples, sampled_urls_file)
    
    batch_files = split_urls_into_batches(sampled_urls_file, args.batch_size, batch_dir)
    
    # count number of batch files
    print(f"Number of batches: {len(batch_files)}")

    # download in parallel
    download_dir = output_path / "scraped_data"
    
    if args.use_slurm:
        if not SUBMITIT_AVAILABLE:
            print("ERROR: submitit not available but --use-slurm specified")
            sys.exit(1)
        run_with_submitit(batch_files, download_dir, 
                         args.slurm_partition, args.slurm_time, args.slurm_mem,
                         args.slurm_qos, args.n_slurm_jobs, args.local_workers)
    else:
        run_with_local_multiprocessing(batch_files, download_dir, args.local_workers)
    
    print(f"\nAll downloads complete! Check {download_dir} for WARC files.")
    print(f"Logs available in {download_dir}/*.log")


if __name__ == "__main__":
    main()