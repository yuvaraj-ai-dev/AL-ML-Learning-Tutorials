import os
import json
import ast

def extract_imports_from_code(code):
    try:
        tree = ast.parse(code)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
        return imports
    except SyntaxError:
        return set()

folder = 'Machine Learning/05_Image_Video_Processing/'
all_imports = set()

for file in os.listdir(folder):
    if file.endswith('.ipynb'):
        path = os.path.join(folder, file)
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                all_imports.update(extract_imports_from_code(source))

print("Unique packages:")
for pkg in sorted(all_imports):
    print(pkg)