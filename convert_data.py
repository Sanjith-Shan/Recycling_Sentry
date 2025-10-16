import os
import shutil
import json
from pathlib import Path
from collections import defaultdict

def detect_format(dataset_path):
    dataset_path = Path(dataset_path)
    
    if (dataset_path / 'train' / '_annotations.coco.json').exists():
        return 'coco'
    
    if (dataset_path / 'train' / 'labels').exists():
        return 'yolo'
    
    train_path = dataset_path / 'train'
    if train_path.exists() and any(d.is_dir() for d in train_path.iterdir()):
        for subdir in train_path.iterdir():
            if subdir.is_dir():
                image_files = list(subdir.glob('*.jpg')) + list(subdir.glob('*.png'))
                if image_files:
                    return 'folder_structure'
    
    return 'unknown'

def convert_coco_format(dataset_path, output_path):
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    
    for split in ['train', 'valid', 'test']:
        split_dir = dataset_path / split
        if not split_dir.exists():
            continue
        
        anno_file = split_dir / '_annotations.coco.json'
        if not anno_file.exists():
            print(f"No annotations found for {split}")
            continue
        
        with open(anno_file, 'r') as f:
            coco_data = json.load(f)
        
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        img_to_cat = {}
        for anno in coco_data['annotations']:
            img_id = anno['image_id']
            cat_id = anno['category_id']
            img_to_cat[img_id] = categories[cat_id]
        
        for cat_name in categories.values():
            (output_path / split / cat_name).mkdir(parents=True, exist_ok=True)
        
        for img_data in coco_data['images']:
            img_id = img_data['id']
            img_filename = img_data['file_name']
            
            if img_id in img_to_cat:
                cat_name = img_to_cat[img_id]
                src = split_dir / img_filename
                dst = output_path / split / cat_name / img_filename
                
                if src.exists():
                    shutil.copy2(src, dst)
        
def convert_yolo_format(dataset_path, output_path):
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    
    yaml_file = dataset_path / 'data.yaml'
    if not yaml_file.exists():
        print("data.yaml not found")
        return
    
    with open(yaml_file, 'r') as f:
        lines = f.readlines()
    
    class_names = []
    reading_names = False
    for line in lines:
        if 'names:' in line:
            reading_names = True
            continue
        if reading_names:
            if line.strip().startswith('-'):
                class_name = line.split('-')[1].strip()
                class_names.append(class_name)
            elif ':' in line and not line.strip().startswith('#'):
                parts = line.split(':')
                if len(parts) == 2:
                    class_name = parts[1].strip()
                    class_names.append(class_name)
        
    for split in ['train', 'valid', 'test']:
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'
        
        if not images_dir.exists():
            continue
        
        for class_name in class_names:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
        
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    first_line = f.readline().strip()
                
                if first_line:
                    class_idx = int(first_line.split()[0])
                    if class_idx < len(class_names):
                        class_name = class_names[class_idx]
                        dst = output_path / split / class_name / img_file.name
                        shutil.copy2(img_file, dst)
        
def convert_folder_structure(dataset_path, output_path):
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    for split in ['train', 'valid', 'test']:
        split_dir = dataset_path / split
        if split_dir.exists():
            output_split = output_path / split
            if output_split.exists():
                shutil.rmtree(output_split)
            shutil.copytree(split_dir, output_split)
            
            num_classes = len([d for d in output_split.iterdir() if d.is_dir()])

def rename_valid_to_val(output_path):
    valid_path = Path(output_path) / 'valid'
    val_path = Path(output_path) / 'val'
    
    if valid_path.exists() and not val_path.exists():
        valid_path.rename(val_path)

def print_statistics(output_path):
    output_path = Path(output_path)
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        split_path = output_path / split
        if not split_path.exists():
            continue
        
        total = 0
        for class_dir in sorted(split_path.iterdir()):
            if class_dir.is_dir():
                num_images = len(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')))
                print(f"  {class_dir.name}: {num_images} images")
                total += num_images
        
        print(f"  TOTAL: {total} images")
    

def main():
    import sys
    
    if len(sys.argv) > 1:
        dataset_path = Path(sys.argv[1]).expanduser()
    else:
        dataset_path = input("\nEnter path to Roboflow dataset folder: ").strip()
        dataset_path = Path(dataset_path).expanduser()
    
    if not dataset_path.exists():
        print(f"Path '{dataset_path}' does not exist")
        return
    
    output_path = Path("data")
    
    format_type = detect_format(dataset_path)
    print(f"\nDetected format: {format_type.upper()}")
    
    if format_type == 'unknown':
        print("Could not detect dataset format")
        return
    
    output_path.mkdir(exist_ok=True)
    
    if format_type == 'coco':
        convert_coco_format(dataset_path, output_path)
    elif format_type == 'yolo':
        convert_yolo_format(dataset_path, output_path)
    elif format_type == 'folder_structure':
        convert_folder_structure(dataset_path, output_path)
    
    rename_valid_to_val(output_path)
    
    print_statistics(output_path)

if __name__ == "__main__":
    main()