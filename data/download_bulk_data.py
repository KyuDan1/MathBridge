
"""
Prerequisition:

1. You have to prepare AWS access key and secret access key. And also give authority.
2. You have to prepare manifest.xml file for pdf and src.
3. You have to get download tool 'aria2c.exe' in https://aria2.github.io/.
Then you have to embed in system file (PATH).

execute below instruction.

python script/download_bulk_data.py pdf downloads
or
python script/download_bulk_data.py src downloads
"""



import os
import json
import boto3
import requests
from tqdm import tqdm
import xml.etree.ElementTree as ET
import subprocess
import sys
import time

bucket_name = 'arxiv'
REGION_NAME = 'us-west-1'
NUM_THREADS = 16  # MAX = 16 Number of threads to use for downloading chunks of each file

s3 = boto3.client('s3', aws_access_key_id='', 
                  aws_secret_access_key= '', region_name=REGION_NAME)



def parse_manifest(manifest):
    root = ET.fromstring(manifest)
    return [{c.tag: f.find(c.tag).text for c in list(f)} for f in root.findall('file')]

def move_file_to_final_location(cache_file, final_file):
    os.makedirs(os.path.dirname(final_file), exist_ok=True)
    os.rename(cache_file, final_file)

def download_file_with_aria2(filename, cache_file, num_threads=NUM_THREADS):
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": filename, "RequestPayer": 'requester'},
        ExpiresIn=3600
    )

    aria2c_cmd = [
        ".\\aria2c",
        "--file-allocation=none",
        "--allow-overwrite=true",
        f"--max-connection-per-server={num_threads}",
        "--split=10",
        "--min-split-size=1M",
        "--dir={}".format(os.path.dirname(cache_file)),
        "--out={}".format(os.path.basename(cache_file)),
        url
    ]

    result = subprocess.run(aria2c_cmd, capture_output=True)
    if result.returncode != 0:
        tqdm.write(f"Error downloading {filename} with aria2: {result.stderr.decode()}")
        return False
    return True

def download_check_tarfiles(file_info_list, output_dir, cache_dir):
    total_files = len(file_info_list)
    start_time = time.time()

    # Filter out already completed files
    completed_files = set(f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f)))
    file_info_list = [f for f in file_info_list if os.path.basename(f['filename']) not in completed_files]

    print(f"Total files to download: {total_files}")
    print(f"Completed files: {len(completed_files)}")
    print(f"Remaining files: {len(file_info_list)}")

    # Sort files by filename to ensure orderly processing
    file_info_list.sort(key=lambda x: x['filename'])

    with tqdm(total=len(file_info_list), desc="Total Progress", unit='file') as pbar_total:
        for fileinfo in file_info_list:
            filename = fileinfo['filename']
            final_file = os.path.join(output_dir, os.path.basename(filename))
            cache_file = os.path.join(cache_dir, os.path.basename(filename))

            if os.path.exists(final_file):
                pbar_total.update(1)
                continue

            result = download_file_with_aria2(filename, cache_file)
            if result:
                move_file_to_final_location(cache_file, final_file)
            pbar_total.update(1)

            current_time = time.time()
            elapsed_time = current_time - start_time
            avg_time_per_file = elapsed_time / (pbar_total.n + 1)
            remaining_files = len(file_info_list) - pbar_total.n
            estimated_time_remaining = avg_time_per_file * remaining_files

            tqdm.write(f"{filename} - Estimated time remaining: {estimated_time_remaining / 60:.2f} minutes")

def download_manifest_file(manifest_key, output_dir):
    manifest_path = os.path.join(output_dir, os.path.basename(manifest_key))
    if not os.path.exists(manifest_path):
        download_file(manifest_key, manifest_path, redownload=True)
    return manifest_path

def download_file(filename, outfile, chunk_size=2**20, redownload=False, dryrun=False):
    """
    Downloads filename from the ArXiv AWS S3 bucket.
    """
    if os.path.exists(outfile) and not redownload:
        return True

    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": filename, "RequestPayer": 'requester'},
        ExpiresIn=3600
    )

    try:
        if not dryrun:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            file_size = int(response.headers.get('content-length', 0))
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=filename, initial=0, ascii=True) as pbar:
                response_iter = response.iter_content(chunk_size=chunk_size)
                with open(outfile, 'wb') as fout:
                    for chunk in response_iter:
                        fout.write(chunk)
                        pbar.update(len(chunk))
        return os.path.getsize(outfile) == file_size
    except requests.exceptions.RequestException as e:
        tqdm.write(f"Error downloading {filename}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python download_bulk_data.py <data_type> <work_dir>")
        print("<data_type>: all, pdf, or src")
        sys.exit(1)

    data_type = sys.argv[1]
    work_dir = sys.argv[2]

    if data_type not in ['all', 'pdf', 'src']:
        print("Invalid data type. Choose from 'all', 'pdf', or 'src'.")
        sys.exit(1)

    pdf_output_dir = os.path.join(work_dir, 'pdfs')
    src_output_dir = os.path.join(work_dir, 'src')
    cache_dir = os.path.join(work_dir, 'cache')

    if data_type in ['all', 'pdf'] and not os.path.exists(pdf_output_dir):
        os.makedirs(pdf_output_dir)
    if data_type in ['all', 'src'] and not os.path.exists(src_output_dir):
        os.makedirs(src_output_dir)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if data_type in ['all', 'pdf']:
        pdf_manifest_key = 'pdf/arXiv_pdf_manifest.xml'
        pdf_manifest_file = download_manifest_file(pdf_manifest_key, work_dir)
        print("Loading PDF manifest...")
        file_info_list = parse_manifest(open(pdf_manifest_file, 'r').read())
        file_info_list.sort(key=lambda x: x['filename'])
        download_check_tarfiles(file_info_list, pdf_output_dir, cache_dir)

    if data_type in ['all', 'src']:
        src_manifest_key = 'src/arXiv_src_manifest.xml'
        src_manifest_file = download_manifest_file(src_manifest_key, work_dir)
        print("Loading Source manifest...")
        file_info_list = parse_manifest(open(src_manifest_file, 'r').read())
        file_info_list.sort(key=lambda x: x['filename'])
        download_check_tarfiles(file_info_list, src_output_dir, cache_dir)
