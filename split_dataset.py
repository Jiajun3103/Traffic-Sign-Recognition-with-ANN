import json
import os
import shutil
from collections import defaultdict
import random
from tqdm import tqdm 

# Configuration
SOURCE_BASE_DIR = 'test.v1i.coco' 
TARGET_BASE_DIR = 'data_stratified' 

TRAIN_SPLIT_RATIO = 0.86
VAL_SPLIT_RATIO = 0.13 
TEST_SPLIT_RATIO = 0.01

# Ensure ratios sum to 1
assert TRAIN_SPLIT_RATIO + VAL_SPLIT_RATIO + TEST_SPLIT_RATIO == 1.0, \
    "Split ratios must sum to 1.0!"

# Helper Functions
def load_coco_json(json_path):
    """Loads a COCO JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def save_coco_json(data, json_path):
    """Saves data to a COCO JSON file."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def merge_datasets(source_dirs, source_base_dir):
    """
    Merges multiple COCO annotation JSON files and their corresponding images
    into a single dataset representation.
    """
    merged_images = []
    merged_annotations = []
    merged_categories = []
    
    seen_image_ids = set()
    seen_ann_ids = set()
    seen_category_ids = set()

    for split_dir in source_dirs:
        json_path = os.path.join(source_base_dir, split_dir, '_annotations.coco.json')
        image_dir = os.path.join(source_base_dir, split_dir)

        if not os.path.exists(json_path):
            print(f"Warning: JSON file not found at {json_path}. Skipping.")
            continue
        
        data = load_coco_json(json_path)

        # Merge categories, ensuring no duplicates
        for cat in data['categories']:
            if cat['id'] not in seen_category_ids:
                merged_categories.append(cat)
                seen_category_ids.add(cat['id'])
        
        # Merge images, adjusting paths and ensuring unique IDs
        for img in data['images']:
            # Create a unique ID for the image across all original datasets
            original_image_id = img['id']
            new_image_id = f"{split_dir}_{original_image_id}" 
            img['id'] = new_image_id
            img['file_name'] = os.path.join(split_dir, img['file_name']) # Store original path for copying later
            
            if new_image_id not in seen_image_ids:
                merged_images.append(img)
                seen_image_ids.add(new_image_id)
        
        # Merge annotations, ensuring unique IDs and linking to new image IDs
        # Need to map original image IDs to new image IDs for annotations
        original_to_new_image_id_map = {f"{split_dir}_{img['id']}": img['id'] for img in data['images']}

        for ann in data['annotations']:
            # Create a unique annotation ID
            original_ann_id = ann['id']
            new_ann_id = f"{split_dir}_{original_ann_id}"
            ann['id'] = new_ann_id
            
            # Map annotation's image_id to the new global image_id
            # This requires careful handling if original IDs aren't unique across splits
            # For robustness, we recreate the image_id using the split_dir prefix
            ann['image_id'] = f"{split_dir}_{ann['image_id']}"

            if new_ann_id not in seen_ann_ids:
                merged_annotations.append(ann)
                seen_ann_ids.add(new_ann_id)

    # Re-index all IDs to be contiguous from 0 after merging if desired, 
    # but for simplicity, we'll use the prefixed IDs.
    # If using for training, ensure your dataset loader can handle non-contiguous IDs
    # or re-index them before saving. For this example, we keep prefixed IDs.

    print(f"Merged {len(merged_images)} images and {len(merged_annotations)} annotations from {len(source_dirs)} splits.")
    return {
        'images': merged_images,
        'annotations': merged_annotations,
        'categories': merged_categories,
        'info': data.get('info', {}),
        'licenses': data.get('licenses', [])
    }

def perform_stratified_split(all_data):
    """
    Performs a stratified split of the merged dataset based on image-level categories.
    Each image is assigned a primary category based on its annotations.
    """
    images_by_category = defaultdict(list)
    
    # Map image_id to its annotations for easy lookup
    annotations_by_image_id = defaultdict(list)
    for ann in all_data['annotations']:
        annotations_by_image_id[ann['image_id']].append(ann)

    # Assign each image to a primary category. If multiple, pick the first.
    # Images with no annotations are put into a "no_annotation" category.
    for img in all_data['images']:
        img_annotations = annotations_by_image_id[img['id']]
        if img_annotations:
            # For multi-class (single-label), we assume one dominant label per image.
            # Take the first annotation's category ID.
            primary_category_id = img_annotations[0]['category_id']
            images_by_category[primary_category_id].append(img)
        else:
            images_by_category['no_annotation'].append(img) # For images without annotations

    train_images = []
    val_images = []
    test_images = []

    for category_id, images in images_by_category.items():
        random.shuffle(images)
        total_images = len(images)
        
        train_end = int(total_images * TRAIN_SPLIT_RATIO)
        val_end = train_end + int(total_images * VAL_SPLIT_RATIO)

        train_images.extend(images[:train_end])
        val_images.extend(images[train_end:val_end])
        test_images.extend(images[val_end:])
    
    print(f"Split distribution: Train {len(train_images)}, Valid {len(val_images)}, Test {len(test_images)}")

    # Collect annotations for each split
    train_image_ids = {img['id'] for img in train_images}
    val_image_ids = {img['id'] for img in val_images}
    test_image_ids = {img['id'] for img in test_images}

    train_annotations = [ann for ann in all_data['annotations'] if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in all_data['annotations'] if ann['image_id'] in val_image_ids]
    test_annotations = [ann for ann in all_data['annotations'] if ann['image_id'] in test_image_ids]

    return {
        "train": {
            "images": train_images,
            "annotations": train_annotations,
            "categories": all_data['categories'],
            "info": all_data['info'], "licenses": all_data['licenses']
        },
        "valid": {
            "images": val_images,
            "annotations": val_annotations,
            "categories": all_data['categories'],
            "info": all_data['info'], "licenses": all_data['licenses']
        },
        "test": {
            "images": test_images,
            "annotations": test_annotations,
            "categories": all_data['categories'],
            "info": all_data['info'], "licenses": all_data['licenses']
        }
    }

def copy_images_for_splits(all_merged_data, new_splits_data):
    """
    Copies images from their original locations to the new split directories.
    Assumes original file_name in all_merged_data['images'] contains the path relative to SOURCE_BASE_DIR.
    """
    print("Copying images to new split directories...")
    # Create target directories
    os.makedirs(os.path.join(TARGET_BASE_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(TARGET_BASE_DIR, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(TARGET_BASE_DIR, 'test'), exist_ok=True)

    # Create a map from new image ID to original file path
    image_id_to_original_path = {img['id']: img['file_name'] for img in all_merged_data['images']}

    splits_to_copy = {
        "train": new_splits_data["train"]["images"],
        "valid": new_splits_data["valid"]["images"],
        "test": new_splits_data["test"]["images"]
    }

    for split_name, images_in_split in splits_to_copy.items():
        target_dir = os.path.join(TARGET_BASE_DIR, split_name)
        print(f"Copying images for {split_name} split to {target_dir}...")
        for img_info in tqdm(images_in_split):
            original_relative_path = image_id_to_original_path[img_info['id']]
            source_path = os.path.join(SOURCE_BASE_DIR, original_relative_path)
            
            # The destination file name should only be the actual image file name
            destination_file_name = os.path.basename(original_relative_path)
            destination_path = os.path.join(target_dir, destination_file_name)
            
            if os.path.exists(source_path):
                shutil.copy(source_path, destination_path)
            else:
                print(f"Warning: Source image not found at {source_path}. Skipping copy for {img_info['file_name']}.")
            

# Main Execution
if __name__ == "__main__":
    print("Starting dataset merging and splitting...")

    # 1. Merge all original datasets
    source_splits = ['train', 'valid', 'test'] 
    all_merged_data = merge_datasets(source_splits, SOURCE_BASE_DIR)

    # 2. Perform stratified split
    new_splits_data = perform_stratified_split(all_merged_data)

    # 3. Save new COCO JSON files
    print("Saving new COCO JSON files...")
    save_coco_json(new_splits_data["train"], os.path.join(TARGET_BASE_DIR, 'train', '_annotations.coco.json'))
    save_coco_json(new_splits_data["valid"], os.path.join(TARGET_BASE_DIR, 'valid', '_annotations.coco.json')) 
    save_coco_json(new_splits_data["test"], os.path.join(TARGET_BASE_DIR, 'test', '_annotations.coco.json'))

    # 4. Copy images to new directories
    copy_images_for_splits(all_merged_data, new_splits_data)

    print("Dataset splitting complete. Update your main.py to use the new paths:")
    print(f"train_json_path = '{TARGET_BASE_DIR}/train/_annotations.coco.json'")
    print(f"valid_json_path = '{TARGET_BASE_DIR}/valid/_annotations.coco.json'")
    print(f"test_json_path = '{TARGET_BASE_DIR}/test/_annotations.coco.json'")
