import os
import sys
import chardet
from pathlib import Path

def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read())
    return result["encoding"]

def convert_to_utf8(src_file_path, dest_file_path, original_encoding, idx):
    try:
        with open(src_file_path, "r", encoding=original_encoding) as f:
            content = f.read()

        with open(dest_file_path, "w", encoding="utf-8") as f:
            f.write(content)
            print(f"    [{idx}] write {src_file_path} {dest_file_path}")

        return True
    except (UnicodeDecodeError, UnicodeEncodeError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error converting file {src_file_path}: {e}")

    return False

def process_directory(directory):
    idx = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".cpp", ".h", ".hpp", ".cs", ".inl")):
                file_path = Path(root) / file
                original_encoding = detect_encoding(file_path)
                if original_encoding and original_encoding.lower() != "utf-8":
                    tmp_file_path = file_path.with_suffix(file_path.suffix + ".tmp")
                    
                    # 尝试转换，若成功
                    if convert_to_utf8(file_path, tmp_file_path, original_encoding, idx):
                        backup_file_path = file_path.with_suffix(file_path.suffix + ".back")
                        # 将原文件改名为back
                        os.rename(file_path, backup_file_path)

                        # 将tmp改名为原文件
                        os.rename(tmp_file_path, file_path)

                        print(f"[{idx}] Converted {original_encoding} {file_path} to utf-8")
                        idx += 1
                    #else:
                        # 转换失败，恢复原文件名
                        #os.rename(tmp_file_path, file_path)
                        #print(f"Failed to convert {file_path} from {original_encoding} to utf-8")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auto_fix_utf8.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        sys.exit(1)

    process_directory(directory)
