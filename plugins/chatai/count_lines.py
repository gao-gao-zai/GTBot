import os
from pathlib import Path

def count_lines_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return len(f.readlines())
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0

def count_lines_in_directory(directory):
    file_line_counts = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                line_count = count_lines_in_file(file_path)
                file_line_counts.append((str(file_path), line_count))
    
    file_line_counts.sort(key=lambda x: x[1], reverse=True)
    
    return file_line_counts

def main():
    current_directory = Path(__file__).parent
    print(f"Directory: {current_directory}")
    
    file_line_counts = count_lines_in_directory(current_directory)
    
    print("\nPython files line count (sorted by line count):")
    print("-" * 60)
    total_lines = 0
    
    for file_path, line_count in file_line_counts:
        relative_path = os.path.relpath(file_path, current_directory)
        print(f"{line_count:>6} lines | {relative_path}")
        total_lines += line_count
    
    print("-" * 60)
    print(f"{total_lines:>6} lines | TOTAL")

if __name__ == "__main__":
    main()