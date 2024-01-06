import os
import sys
import chardet
from pathlib import Path

def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read())
    return result["encoding"]

def convert_to_utf8(src_file_path, dest_file_path, original_encoding):

    try:
        with open(src_file_path, "r", encoding=original_encoding) as f:
            content = f.read()

        with open(dest_file_path, "w", encoding="utf-8") as f:
            f.write(content)

    except UnicodeDecodeError as e:
        print(f"Error decoding file {src_file_path}: {e}")
    except UnicodeEncodeError as e:
        print(f"Error encoding file {src_file_path}: {e}")
    except Exception as e:
        print(f"Error converting file {src_file_path}: {e}")

def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".cpp", ".h", ".hpp", ".cs")):
                file_path = Path(root) / file
                original_encoding = detect_encoding(file_path)
                if original_encoding.lower() != "utf-8":
                    backup_file_path = file_path.with_suffix(".back")
                    os.rename(file_path, backup_file_path)
                    convert_to_utf8(backup_file_path, file_path, original_encoding)
                    print(f"Converted {original_encoding}  {file_path} to utf-8")
                    #os.remove(backup_file_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_utf8.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        sys.exit(1)

    process_directory(directory)
