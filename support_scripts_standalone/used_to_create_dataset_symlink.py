import os

dataset_source_dir = r'D:\programming\PyCharm Community Edition\Solutions\GestureClassifier\private\data' 
# Where you want the link to appear
link_name = 'data' 

try:
    os.symlink(dataset_source_dir, link_name, target_is_directory=True)
    print("Symlink created successfully!")
except FileExistsError:
    print("Link already exists.")
except OSError as e:
    print(f"Error: {e}. Note: On Windows, you may need admin rights.")