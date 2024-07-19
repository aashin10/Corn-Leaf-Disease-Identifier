import os
import random
import shutil


def data_preprocess(train_dir, test_dir, percent=15):
    os.makedirs(test_dir, exist_ok=True)
    for disease_name in os.listdir(train_dir):
        disease_train_path = os.path.join(train_dir, disease_name)
        if not os.path.isdir(disease_train_path):
            continue
        # make folder for each disease
        disease_test_path = os.path.join(test_dir, disease_name)
        os.makedirs(disease_test_path, exist_ok=True)

        # select 15% of images for testing randomly
        files = os.listdir(disease_train_path)
        num_images = len(files)
        num_to_select = int(num_images * (percent / 100))
        selected_images = random.sample(files, num_to_select)

        # move selected images to test_dir
        for img in selected_images:
            src_path = os.path.join(disease_train_path, img)
            dest_path = os.path.join(disease_test_path, img)
            shutil.move(src_path, dest_path)


# by default every image captured will be stored into train_dir
train_dir = 'assets/user_dataset/Train/'
test_dir = 'assets/user_dataset/Test/'
val_dir = 'assets/user_dataset/Validate'
data_preprocess(train_dir, test_dir)
data_preprocess(train_dir, val_dir)
print("Data Splitting Done Successfully!")