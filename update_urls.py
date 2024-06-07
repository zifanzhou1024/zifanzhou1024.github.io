import os

# Directories and files to check
directories = [
    '_layouts',
    'assets',
    '_includes'
]

# Files to scan
files_to_scan = ['html', 'css', 'js', 'yml']

# Function to update URLs
def update_urls_in_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    updated_content = content.replace('http://zifanzhou.com', 'https://zifanzhou.com')

    with open(file_path, 'w') as file:
        file.write(updated_content)

def scan_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.split('.')[-1] in files_to_scan:
                file_path = os.path.join(root, file)
                update_urls_in_file(file_path)

if __name__ == "__main__":
    base_path = '/home/sakura/zifanzhou1024.github.io'
    
    for directory in directories:
        full_path = os.path.join(base_path, directory)
        if os.path.exists(full_path):
            scan_directory(full_path)
        else:
            print(f"Directory {full_path} does not exist.")
