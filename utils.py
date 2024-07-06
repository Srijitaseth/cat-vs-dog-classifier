import os
import shutil

# Path to the test directory
test_dir = '/Users/srijitaseth/cat_dog/cats_vs_dogs/test'

def move_non_image_files():
    # Create a directory for XML files
    xml_dir = 'cats_vs_dogs/xml_files'
    os.makedirs(xml_dir, exist_ok=True)

    # Move XML files to the new directory
    for file_name in os.listdir(test_dir):
        if file_name.lower().endswith('.xml'):
            file_path = os.path.join(test_dir, file_name)
            shutil.move(file_path, os.path.join(xml_dir, file_name))

# Call this function to move XML files
if __name__ == "__main__":
    move_non_image_files()
