import re
import os
import sys
import json

def extract_SQLQueryUtil_Query_blocks(code_text):
    pattern = r'\.Query\('

    positions = []
    index = 0
    length = len(code_text)

    while index < length:
        # 查找下一个匹配的位置
        match = re.search(pattern, code_text[index:])
        if not match:
            break
        start_index = index + match.start()
        index = start_index + len(match.group())

        # 从匹配行的第一个字符开始
        line_start = code_text.rfind('\n', 0, start_index) + 1

        # 初始化括号计数器和其他变量
        paren_count = 1  # 已经找到一个 '('
        end_index = index
        in_string = False
        string_char = ''
        escape_char = False

        while end_index < length and paren_count > 0:
            char = code_text[end_index]

            if in_string:
                if escape_char:
                    escape_char = False
                elif char == '\\':
                    escape_char = True
                elif char == string_char:
                    in_string = False
            else:
                if char == '"' or char == "'":
                    in_string = True
                    string_char = char
                elif char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1

            end_index += 1

        # 提取代码块
        code_block = code_text[line_start:end_index]
        positions.append(code_block)

        # 更新索引，继续搜索
        index = end_index

    return positions


total_count = 0
find_count = 0

def extract_lua(file_Path):
    with open(file_Path, 'r', encoding='utf-8') as f:
        code_text = f.read()

    # 提取代码块
    code_blocks = extract_SQLQueryUtil_Query_blocks(code_text)

    global total_count

    total_count += 1

    code_blocks_json = []
    # If code_blocks is not empty, return the dict
    if len(code_blocks) > 0:
        # Optionally, print the filename and code_blocks
        print(f"{file_Path}====:")
        for block in code_blocks:
            print(block)            
            global find_count
            find_count += 1
            code_blocks_json.append({'old':block, 'new':None})
        return {'filename': file_Path, 'code_blocks': code_blocks_json}
    else:
        return None


def process_directory(directory):
    results = []
    total_lines = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".lua"):
                file_path = os.path.join(root, file)
                result = extract_lua(file_path)
                if result:
                    results.append(result)
    return results


if __name__ == '__main__':
    if len(sys.argv) < 2:
        directory = '.'
    else:
        directory = sys.argv[1]
    
    if os.path.isdir(directory):
        results = process_directory(directory)
        with open('sql_results.json', 'w', encoding='utf-8') as output_file:
            json.dump(results, output_file, ensure_ascii=False, indent=4)

    print(f"Total count: {find_count}/{total_count}")
