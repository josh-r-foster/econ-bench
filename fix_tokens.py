import os
import glob

tasks_dir = "/Users/jfoster/Documents/GitHub/econ-bench/src/tasks"
for file_path in glob.glob(os.path.join(tasks_dir, "*.py")):
    with open(file_path, "r") as f:
        content = f.read()
    
    # Replace common max_new_tokens defaults in task files
    new_content = content.replace("max_new_tokens=1000", "max_new_tokens=8192")
    new_content = new_content.replace("max_new_tokens: int = 1024", "max_new_tokens: int = 8192")
    
    if new_content != content:
        with open(file_path, "w") as f:
            f.write(new_content)
        print(f"Updated {file_path}")
