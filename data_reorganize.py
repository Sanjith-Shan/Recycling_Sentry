import os
import shutil
from pathlib import Path
from collections import defaultdict
import re

def extract_class_from_filename(filename):
    name = filename.lower()
    
    class_patterns = {
        'cardboard': 'Cardboard',
        'foil': 'Foil',
        'food wrapper': 'Food Wrapper',
        'food_wrapper': 'Food Wrapper',
        'glass bottle': 'Glass Bottle',
        'glass_bottle': 'Glass Bottle',
        'glass jar': 'Glass Jar',
        'glass_jar': 'Glass Jar',
        'milk carton': 'Milk Carton',
        'milk_carton': 'Milk Carton',
        'pet': 'PET (Water bottle)',
        'water bottle': 'PET (Water bottle)',
        'water_bottle': 'PET (Water bottle)',
        'magazine': 'Magazine',
        'paper': 'Paper',
        'pet_carton': 'PET_Carton (plastic box)',
        'plastic box': 'PET_Carton (plastic box)',
        'plastic_box': 'PET_Carton (plastic box)',
        'metal can': 'Metal Can',
        'metal_can': 'Metal Can',
        'metal': 'Metal',
        'null': 'Null',
    }
    
    for pattern, class_name in class_patterns.items():
        if name.startswith(pattern):
            return class_name
    
    match = re.match(r'^([a-zA-Z\s]+)', name)
    if match:
        extracted = match.group(1).strip().title()
        return extracted
    
    return 'Unknown'

def reorganize_dataset(data_dir):
    data_dir = Path(data_dir)
    
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"Warning: {split} directory not found")
            continue
        
        print(f"\nProcessing {split}...")
        
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_files.extend(split_dir.glob(f'*{ext}'))
        
        class_counts = defaultdict(int)
        for img_file in image_files:
            class_name = extract_class_from_filename(img_file.name)
            class_counts[class_name] += 1
        
        print(f"Found {len(image_files)} images")
        print("Classes detected:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count} images")
        
        moved = 0
        for img_file in image_files:
            class_name = extract_class_from_filename(img_file.name)
            
            class_dir = split_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            dest = class_dir / img_file.name
            if not dest.exists():
                shutil.move(str(img_file), str(dest))
                moved += 1
        
        print(f"Moved {moved} images into class folders")
        
        metadata_folders = ['9', 'CC BY 4.0', 'recyclestuff', 'updated-recycling-dataset']
        for folder in metadata_folders:
            folder_path = split_dir / folder
            if folder_path.exists() and folder_path.is_dir():
                sub_images = []
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    sub_images.extend(folder_path.glob(f'*{ext}'))
                
                if sub_images:
                    print(f"\nFound {len(sub_images)} images in metadata folder '{folder}'")
                    for img_file in sub_images:
                        class_name = extract_class_from_filename(img_file.name)
                        
                        class_dir = split_dir / class_name
                        class_dir.mkdir(exist_ok=True)
                        
                        dest = class_dir / img_file.name
                        if not dest.exists():
                            shutil.move(str(img_file), str(dest))
                    
                    print(f"  Moved {len(sub_images)} images from '{folder}' to class folders")
                    
                    try:
                        if not any(folder_path.iterdir()):
                            folder_path.rmdir()
                            print(f"  Removed empty folder '{folder}'")
                    except:
                        pass

def print_final_statistics(data_dir):
    data_dir = Path(data_dir)
    
    print("\n" + "="*60)
    print("FINAL DATASET STRUCTURE")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        
        print(f"\n{split.upper()}:")
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        
        total = 0
        for class_dir in sorted(class_dirs):
            image_count = len(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')))
            if image_count > 0:
                print(f"  {class_dir.name}: {image_count} images")
                total += image_count
        
        print(f"  TOTAL: {total} images")
    
    print("\n" + "="*60)

def main():
    print("="*60)
    print("REORGANIZING ROBOFLOW DATASET")
    print("="*60)
    
    data_dir = Path("data")
    
    if not data_dir.exists():
        print(f"ERROR: Data directory '{data_dir}' not found.")
        return
    
    reorganize_dataset(data_dir)
    
    print_final_statistics(data_dir)
    
    print("\n✓ Reorganization complete.")

if __name__ == "__main__":
    main()