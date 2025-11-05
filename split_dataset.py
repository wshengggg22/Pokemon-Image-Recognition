import os
import random
import shutil

# Paths
original_dataset_dir = "PokemonData"  
base_dir = "pokemon_split"         
splits = ['train', 'val', 'test']
split_ratios = [0.7, 0.15, 0.15]
random.seed(42)

# Create directories
for split in splits:
    split_dir = os.path.join(base_dir, split)
    os.makedirs(split_dir, exist_ok=True)

# For each Pokémon class folder
for class_name in os.listdir(original_dataset_dir):
    class_dir = os.path.join(original_dataset_dir, class_name)
    if not os.path.isdir(class_dir):
        continue

    # List all image files in the class folder
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)

    # Compute split sizes
    n_total = len(images)
    n_train = int(split_ratios[0] * n_total)
    n_val = int(split_ratios[1] * n_total)

    # Split
    train_files = images[:n_train]
    val_files = images[n_train:n_train + n_val]
    test_files = images[n_train + n_val:]

    # Copy files to new folders
    for split, files in zip(splits, [train_files, val_files, test_files]):
        split_class_dir = os.path.join(base_dir, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)
        for f in files:
            src = os.path.join(class_dir, f)
            dst = os.path.join(split_class_dir, f)
            shutil.copy(src, dst)

    print(f"{class_name}: {n_total} images → train {len(train_files)}, val {len(val_files)}, test {len(test_files)}")

print("Dataset successfully split into train/val/test folders")
