import os
import subprocess
import urllib.request
import shutil

tar_path = r"E:\Downloads\ILSVRC2012_img_val.tar"
extract_dir = r"E:\DL_proj\imagenet\val"

print("Step 1: Extracting tar file (this might take a few minutes)...")
os.makedirs(extract_dir, exist_ok=True)
try:
    subprocess.run(["tar", "-xf", tar_path, "-C", extract_dir], check=True)
except Exception as e:
    print(f"Error extracting tar file: {e}")
    exit(1)

print("Step 2: Downloading valprep.sh for organization mapping...")
valprep_url = "https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh"
valprep_path = os.path.join(extract_dir, "valprep.sh")
try:
    urllib.request.urlretrieve(valprep_url, valprep_path)
except Exception as e:
    print(f"Error downloading valprep.sh: {e}")
    exit(1)

print("Step 3: Organizing images into subfolders...")
with open(valprep_path, 'r') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if line.startswith("mkdir -p"):
        folder = line.split()[-1]
        os.makedirs(os.path.join(extract_dir, folder), exist_ok=True)
    elif line.startswith("mv"):
        parts = line.split()
        if len(parts) >= 3:
            img_file = parts[1]
            dest_folder = parts[2].rstrip('/')
            
            src_path = os.path.join(extract_dir, img_file)
            dest_path = os.path.join(extract_dir, dest_folder, img_file)
            
            if os.path.exists(src_path):
                shutil.move(src_path, dest_path)

print("All done! The ImageNet validation set is properly structured in E:\\DL_proj\\imagenet\\val")
