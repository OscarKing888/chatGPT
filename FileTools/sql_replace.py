import os
import sys
import json

def load_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def build_lookup_table(data):
    lookup_table = {}
    for entry in data:
        filename = entry['filename']
        # Normalize the filename to handle different path separators
        normalized_filename = os.path.normpath(filename)
        code_blocks = entry['code_blocks']

        if normalized_filename in lookup_table:
            print(f"Error: Duplicate filename detected in JSON data: {normalized_filename}")
        else:
            lookup_table[normalized_filename] = code_blocks

    return lookup_table

def process_lua_file(file_path, code_blocks):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"Error: {file_path} is not encoded in UTF-8. Skipping this file.")
        return

    original_content = content  # Keep original content for comparison
    modified = False  # Flag to check if any replacements were made

    for block in code_blocks:
        old_code = block['old']
        new_code = block['new'] if block['new'] is not None else ''
        # Generate the replacement string
        replacement = f"--[[\n{old_code}\n--]]\n{new_code}"
        # Check if old_code exists in the content
        if old_code in content:
            content = content.replace(old_code, replacement)
            modified = True
        else:
            print(f"Warning: In file {file_path}, could not find the code block to replace:\n{old_code}\n")

    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Processed and modified file: {file_path}")
    else:
        print(f"No modifications made to file: {file_path}")

def process_directory(directory, lookup_table):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.lua'):
                file_path = os.path.join(root, file)
                # Get the relative path from the base directory
                rel_path = os.path.relpath(file_path, directory)
                # Normalize the relative path
                normalized_rel_path = os.path.normpath(rel_path)
                if normalized_rel_path in lookup_table:
                    code_blocks = lookup_table[normalized_rel_path]
                    process_lua_file(file_path, code_blocks)
                else:
                    # Filename not in lookup table
                    continue

if __name__ == '__main__':

    if len(sys.argv) < 2:
        directory = '.'
    else:
        directory = sys.argv[1]
        
    json_file = 'sql_extract_ok.json'

    if os.path.isdir(directory):
        try:
            data = load_json(json_file)
            lookup_table = build_lookup_table(data)
            process_directory(directory, lookup_table)
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print(f"The directory {directory} does not exist.")
