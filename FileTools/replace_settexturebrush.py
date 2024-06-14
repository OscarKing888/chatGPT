import os
import re

def replace_set_brush_from_texture_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Pattern to find the old SetBrushFromTexture calls
    pattern = re.compile(r'self\.(C_[\w]+):SetBrushFromTexture\((Info\.[\w]+)\)')
    
    # Replace function
    def replacement(match):
        widget_name = match.group(1)
        image_name = match.group(2)
        return f'SomeBPFunctionLib.SetSetBrushFromTexture(self, self.{widget_name}, {image_name})'

    # Perform replacement
    new_content = pattern.sub(replacement, content)

    # Write the modified content back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(new_content)

def recursively_process_directory(directory='.'):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.lua'):
                file_path = os.path.join(root, file)
                replace_set_brush_from_texture_in_file(file_path)
                print(f'Processed: {file_path}')

# Start processing from the current directory
recursively_process_directory()
