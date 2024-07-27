import ipdb
import requests
from pathlib import Path
import sys
import pandas as pd 
import zipfile
import rarfile
import shutil
import re
import os
import filetype
import subprocess
from urllib.parse import unquote

sys.path.insert(0, '.')
dir_this_file = Path(__file__).parent
urls_form_responses = {
	"0": "https://docs.google.com/spreadsheets/d/e/2PACX-1vTDny8zzlWTbQ-kdHNu68bl07Pi-HB-JDzHP4PnoA1Q79OkIc9_mLtWYVYkBm4YQ9Te3qZqaRX4df9a/pub?gid=626873631&single=true&output=csv",
}

def download_form_responses(idx, url):
	dir_data = dir_this_file / f"formdata_{idx}"
	dir_data.mkdir(exist_ok=True, parents=True)
	f_csv = dir_data / "responses.csv"
	download_csv(url, f_csv)
	df = pd.read_csv(f_csv)    
	df = df.dropna(subset=["Your email"])
	print("Unique emails: ", len(df["Your email"].unique()))
	download_images_from_csv(dir_data, df)

def download_csv(url, output_path):
	response = requests.get(url)
	response.raise_for_status()
	with open(output_path, 'wb') as f:
		f.write(response.content)
	print("CSV downloaded and saved at:", output_path)

def determine_file_type(file_path):
	kind = filetype.guess(file_path)
	if kind is None:
		print(f"Unknown file type for {file_path}")
		return "unknown"
	print(f"Detected MIME type for {file_path}: {kind.mime}")
	return kind.mime

def process_archive_file(archive_path, allowed_extensions):
	print(f"Processing archive: {archive_path}")
	
	if not os.path.exists(archive_path):
		raise FileNotFoundError(f"The file {archive_path} does not exist.")
	
	file_size = os.path.getsize(archive_path)
	print(f"File size: {file_size} bytes")
	
	if file_size == 0:
		print("Warning: File is empty.")
		return
	
	file_type = determine_file_type(archive_path)
	parent_dir = archive_path.parent
	
	if file_type == "application/zip":
		with zipfile.ZipFile(archive_path, 'r') as zip_ref:
			for zip_info in zip_ref.infolist():
				if zip_info.filename[-1] == '/':
					continue  # skip directories
				zip_info.filename = os.path.basename(zip_info.filename)
				zip_ref.extract(zip_info, parent_dir)
		print("Extracted ZIP archive")
	elif file_type == "application/x-rar-compressed":
		unar_path = shutil.which('unar')
		if unar_path is None:
			raise RuntimeError("unar command-line tool not found. Please install it using 'brew install unar'.")
		
		# Extract to a temporary directory
		temp_dir = parent_dir / "temp_extract"
		temp_dir.mkdir(exist_ok=True)
		command = [unar_path, "-force-overwrite", "-o", str(temp_dir), str(archive_path)]
		result = subprocess.run(command, capture_output=True, text=True)
		if result.returncode != 0:
			raise RuntimeError(f"Failed to extract RAR archive. Error: {result.stderr}")
		
		# Check if there's only one directory in temp_dir
		temp_contents = list(temp_dir.iterdir())
		if len(temp_contents) == 1 and temp_contents[0].is_dir():
			# If so, use that as our source
			source_dir = temp_contents[0]
		else:
			# Otherwise, use temp_dir itself
			source_dir = temp_dir
		
		# Move files from source directory to parent directory
		for item in source_dir.iterdir():
			if item.is_file():
				shutil.move(str(item), str(parent_dir))
		
		# Remove the temporary directory
		shutil.rmtree(temp_dir)
		print("Extracted RAR archive")
	else:
		raise ValueError(f"Unsupported archive type: {file_type}")
	
	# Delete the original archive file
	archive_path.unlink()
	print(f"Deleted original archive: {archive_path}")
	
	# Check for unsupported file types
	for file in parent_dir.iterdir():
		if file.is_file() and file.suffix.lower() not in allowed_extensions:
			print(f"Warning: Unsupported file type found after extraction: {file}")


def extract_file_id(url):
	# for google drive
	file_id_match = re.search(r"(?:\/d\/|id=|open\?id=)([a-zA-Z0-9_-]+)", url)
	return file_id_match.group(1) if file_id_match else None

def download_images_from_csv(dir_data, df):
	dir_data_images = dir_data / "images"
	dir_data_images.mkdir(exist_ok=True, parents=True)
	col_imgs = [s for s in df.columns if s.lower()[:17] == "image / image set"]
	assert len(col_imgs) == 1, "exactly one column must start with name 'image'"
	drive_links = df[col_imgs].iloc[:, 0]
	allowed_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif'}
	
	session = requests.Session()
	
	for row_index, drive_link in drive_links.items():
		# if row_index != 12:
		#     continue
		row_dir = dir_data_images / f"idx_{row_index}"
		row_dir.mkdir(parents=True, exist_ok=True)
		
		print(f"Processing link: {drive_link}")
		
		file_id = extract_file_id(drive_link)
		if not file_id:
			raise ValueError(f"Could not extract file ID from link: {drive_link}")
		
		download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

		response = session.get(download_url, stream=True)
		response.raise_for_status()
		
		if "Content-Disposition" in response.headers:
			print(True)
			filename = re.findall("filename=\"(.+)\"", response.headers["Content-Disposition"])[0]
		else:
			print(False)
			confirm_token = next((value for key, value in response.cookies.items() if key.startswith('download_warning')), None)
			if confirm_token:
				download_url = f"{download_url}&confirm={confirm_token}"
				response = session.get(download_url, stream=True)
				response.raise_for_status()
				filename = re.findall("filename=\"(.+)\"", response.headers["Content-Disposition"])[0]
			else:
				filename = f"file_{row_index}"
		
		output_path = row_dir / filename
		print(f"Downloading to: {output_path}")
		
		with open(output_path, 'wb') as f:
			for chunk in response.iter_content(chunk_size=8192):
				if chunk:
					f.write(chunk)
		
		print(f"Downloaded: {filename}")
		
		file_type = determine_file_type(output_path)
		if file_type in ["application/zip", "application/x-rar-compressed"]:
			process_archive_file(output_path, allowed_extensions)
		elif output_path.suffix.lower() not in allowed_extensions:
			print(f"Warning: Unsupported file type downloaded: {output_path}")
	
	print("Image download process completed.")
	ipdb.set_trace()

if __name__ == "__main__":
	for idx, url in urls_form_responses.items():
		download_form_responses(idx, url)