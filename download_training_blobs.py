from azure_blob_user import create_blob_storage_client_via_key, AzureBlobUser
from tqdm import tqdm
from time import sleep
from datetime import datetime
import os
import zipfile
import shutil

from dotenv import load_dotenv
load_dotenv()

target_container = r"buffalo-crop"
output_dir = os.getenv("DOWNLOAD_FOLDER")

MAX_DOWNLOAD_ATTEMPTS = 3  # 1 initial + 2 retries on corruption

def is_zip_valid(zip_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            bad = zf.testzip()
            return bad is None
    except (zipfile.BadZipFile, Exception):
        return False


def download_blob(blob_user, blob_name, dest_path):
    blob_user.establish_blob_client(blob_name)
    blob_user.download_blob_client_contents(dest_path)


def download_and_unzip_blob(blob_user, blob_name, output_dir):
    zip_path = os.path.join(output_dir, os.path.basename(blob_name))
    stem = os.path.splitext(os.path.basename(blob_name))[0]
    unzip_dir = os.path.join(output_dir, stem)

    if os.path.isdir(unzip_dir):
        return True  # already done

    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            download_blob(blob_user, blob_name, zip_path)
        except Exception as e:
            print(f"  Download error on attempt {attempt}/{MAX_DOWNLOAD_ATTEMPTS} for {blob_name}: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            if attempt < MAX_DOWNLOAD_ATTEMPTS:
                sleep(10)
            continue

        if not is_zip_valid(zip_path):
            print(f"  Corrupted zip on attempt {attempt}/{MAX_DOWNLOAD_ATTEMPTS}: {blob_name}")
            os.remove(zip_path)
            if attempt < MAX_DOWNLOAD_ATTEMPTS:
                sleep(5)
            continue

        # Valid zip — unzip then delete
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(unzip_dir)
            os.remove(zip_path)
            return True
        except Exception as e:
            print(f"  Unzip failed for {blob_name}: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            if os.path.isdir(unzip_dir):
                shutil.rmtree(unzip_dir)
            return False

    print(f"  Giving up on {blob_name} after {MAX_DOWNLOAD_ATTEMPTS} attempts.")
    return False


def main():
    os.makedirs(output_dir, exist_ok=True)

    blob_storage_client, expiry = create_blob_storage_client_via_key()
    blob_user = AzureBlobUser(blob_storage_client, expiry)

    blob_user.establish_blob_container(target_container, list_blobs=False)
    print(f"Listing blobs in '{target_container}'...")
    blobs = blob_user.list_blobs_in_container(verbose=False, return_blobs=True)
    blobs = blobs[:3]
    print(f"Found {len(blobs)} blobs.")

    # Skip blobs whose unzip directory already exists
    to_download = []
    for b in blobs:
        stem = os.path.splitext(os.path.basename(b.name))[0]
        if not os.path.isdir(os.path.join(output_dir, stem)):
            to_download.append(b)

    print(f"Skipping {len(blobs) - len(to_download)} already-unzipped. Downloading {len(to_download)}.")

    failed = []
    for blob in tqdm(to_download):
        blob_user.check_if_client_needs_reset()
        blob_user.establish_blob_container(target_container, list_blobs=False)
        success = download_and_unzip_blob(blob_user, blob.name, output_dir)
        if not success:
            failed.append(blob.name)

    succeeded = len(to_download) - len(failed)
    print(f"\nDone. {succeeded} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed blobs:")
        for name in failed:
            print(f"  {name}")
    write_log(output_dir, total=len(blobs), skipped=len(blobs) - len(to_download),
              succeeded=succeeded, failed=failed)


def write_log(output_dir, total, skipped, succeeded, failed):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"download_log_{timestamp}.txt")
    with open(log_path, "w") as f:
        f.write(f"Download log — {datetime.now().isoformat()}\n")
        f.write(f"Container : {target_container}\n")
        f.write(f"Output dir: {output_dir}\n\n")
        f.write(f"Total blobs  : {total}\n")
        f.write(f"Skipped      : {skipped}\n")
        f.write(f"Succeeded    : {succeeded}\n")
        f.write(f"Failed       : {len(failed)}\n")
        if failed:
            f.write("\nFailed blobs:\n")
            for name in failed:
                f.write(f"  {name}\n")
    print(f"Log written to {log_path}")


if __name__ == "__main__":
    main()
