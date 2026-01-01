#!/usr/bin/env python3
"""Fix HTML files to use Tailwind CDN instead of inline CSS"""

import re

def fix_html_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace the style block with Tailwind CDN script
    # Match from <style> to </style> including all content between
    pattern = r'<style>.*?</style>'
    replacement = '<script src="https://cdn.tailwindcss.com"></script>'
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    with open(filepath, 'w') as f:
        f.write(new_content)
    
    print(f"Fixed {filepath}")

if __name__ == "__main__":
    fix_html_file('frontend/index.html')
    fix_html_file('frontend/pipeline.html')
    print("Done! Tailwind CDN restored in both files.")
