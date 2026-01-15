import sys
import os

# Fix for transformers import issue with torch version
import torch
torch_version_str = str(torch.__version__)

# Monkey-patch the packaging.version.parse function
import packaging.version

original_parse = packaging.version.parse

def patched_parse(version):
    # Convert torch version objects to string
    if hasattr(version, '__str__'):
        version = str(version)
    return original_parse(version)

packaging.version.parse = patched_parse

# Also patch the transformers.utils.import_utils module before it's imported
import importlib.util
import types

# Create a module that will be loaded instead of the real one
spec = importlib.util.spec_from_file_location(
    'import_utils',
    '/nas_train/app.e0031982/miniforge3/envs/py310/lib/python3.10/site-packages/transformers/utils/import_utils.py'
)

# Read the original file
with open('/nas_train/app.e0031982/miniforge3/envs/py310/lib/python3.10/site-packages/transformers/utils/import_utils.py', 'r') as f:
    source = f.read()

# Replace the problematic line
source = source.replace(
    'torch_version = version.parse(_torch_version)',
    f'torch_version = version.parse("{torch_version_str}")'
)

# Execute the modified source
module = types.ModuleType('transformers.utils.import_utils')
exec(source, module.__dict__)

# Replace in sys.modules
sys.modules['transformers.utils.import_utils'] = module

print(f"Patched transformers.utils.import_utils with torch_version_str: {torch_version_str}")
