import os
import sys

def count_lines(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    return len(lines)

def process_directory(directory):
    total_lines = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".cs"):
                file_path = os.path.join(root, file)
                lines = count_lines(file_path)
                total_lines += lines
                print(f"{file_path}: {lines} lines")
    print(f"Total lines: {total_lines}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_lines.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        sys.exit(1)

    process_directory(directory)
