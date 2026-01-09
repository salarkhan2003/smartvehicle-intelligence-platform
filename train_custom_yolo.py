"""
Custom YOLOv8 Training Script for Truck Blind Spot Detection
Quick training demo with pre-trained weights fine-tuning
Usage: python train_custom_yolo.py --epochs 5
"""

import argparse
import yaml
import os
from pathlib import Path
from ultralytics import YOLO
import torch
import logging
from datetime import datetime
import json
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("YOLO-Training")

# Quick demo mode - uses transfer learning on pre-trained YOLOv8
DEMO_MODE = True


def prepare_dataset(dataset_path):
    """
    Prepare dataset structure for YOLOv8 training
    Expected structure:
    truck_blindspots/
        images/
            train/
            val/
            test/
        labels/
            train/
            val/
            test/
        data.yaml
    """
    dataset_dir = Path(dataset_path)

    # Create directories if they don't exist
    for split in ['train', 'val', 'test']:
        (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Create data.yaml if it doesn't exist
    data_yaml_path = dataset_dir / 'data.yaml'
    if not data_yaml_path.exists():
        data_yaml = {
            'path': str(dataset_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {
                0: 'pedestrian',
                1: 'bicycle',
                2: 'car',
                3: 'motorcycle',
                4: 'truck',
                5: 'bus',
                6: 'truck_blindspot_pedestrian',  # Truck-specific
                7: 'truck_blindspot_vehicle',      # Truck-specific
                8: 'construction_worker',          # High-risk
                9: 'child',                        # High-risk
                10: 'false_positive_shadow',       # Training for FP reduction
                11: 'false_positive_reflection'    # Training for FP reduction
            },
            'nc': 12  # Number of classes
        }

        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)

        logger.info(f"Created data.yaml at {data_yaml_path}")

    return str(data_yaml_path)


def augment_dataset(dataset_path):
    """
    Apply data augmentation to increase training samples
    Using albumentations for advanced augmentation
    """
    try:
        import albumentations as A
        from PIL import Image
        import numpy as np

        logger.info("Applying data augmentation...")

        # Define augmentation pipeline
        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.3),
            A.RandomRain(slant_lower=-10, slant_upper=10, p=0.3),
            A.RandomSunFlare(p=0.2),
            A.GaussNoise(p=0.3),
            A.MotionBlur(blur_limit=7, p=0.3),
            A.HorizontalFlip(p=0.5)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        dataset_dir = Path(dataset_path)
        train_images = list((dataset_dir / 'images' / 'train').glob('*.jpg'))

        augmented_count = 0
        for img_path in train_images[:100]:  # Augment first 100 images
            # Load image and label
            image = np.array(Image.open(img_path))
            label_path = dataset_dir / 'labels' / 'train' / f"{img_path.stem}.txt"

            if not label_path.exists():
                continue

            # Load YOLO format labels
            with open(label_path, 'r') as f:
                labels = [line.strip().split() for line in f.readlines()]

            if not labels:
                continue

            bboxes = [[float(x) for x in label[1:]] for label in labels]
            class_labels = [int(label[0]) for label in labels]

            # Apply augmentation
            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

                # Save augmented image
                aug_img_path = dataset_dir / 'images' / 'train' / f"{img_path.stem}_aug{augmented_count}.jpg"
                Image.fromarray(augmented['image']).save(aug_img_path)

                # Save augmented labels
                aug_label_path = dataset_dir / 'labels' / 'train' / f"{img_path.stem}_aug{augmented_count}.txt"
                with open(aug_label_path, 'w') as f:
                    for cls, bbox in zip(augmented['class_labels'], augmented['bboxes']):
                        f.write(f"{cls} {' '.join(map(str, bbox))}\n")

                augmented_count += 1
            except Exception as e:
                logger.warning(f"Augmentation failed for {img_path}: {e}")

        logger.info(f"Created {augmented_count} augmented samples")

    except ImportError:
        logger.warning("albumentations not installed, skipping augmentation")


def train_custom_yolo(args):
    """Train custom YOLOv8 model for truck blind spots"""
    logger.info("=" * 60)
    logger.info("TNT T-SA Custom YOLOv8 Training")
    logger.info("=" * 60)

    # Prepare dataset
    data_yaml = prepare_dataset(args.dataset)

    # Apply augmentation if requested
    if args.augment:
        augment_dataset(args.dataset)

    # Load base model
    logger.info(f"Loading base model: {args.model}")
    model = YOLO(args.model)

    # Training configuration
    train_config = {
        'data': data_yaml,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.img_size,
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'patience': 50,  # Early stopping
        'save': True,
        'save_period': 10,  # Save every 10 epochs
        'cache': True,  # Cache images for faster training
        'workers': 8,
        'optimizer': 'AdamW',
        'lr0': 0.001,  # Initial learning rate
        'lrf': 0.01,   # Final learning rate
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'box': 7.5,  # Box loss weight
        'cls': 0.5,  # Class loss weight
        'dfl': 1.5,  # Distribution focal loss weight
        'project': 'runs/train',
        'name': f'yolov8-truck-blindspot-{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'exist_ok': True,
        'pretrained': True,
        'verbose': True
    }

    logger.info(f"Training configuration:")
    for key, value in train_config.items():
        logger.info(f"  {key}: {value}")

    # Train model
    logger.info("Starting training...")
    results = model.train(**train_config)

    # Evaluate model
    logger.info("Evaluating model on validation set...")
    metrics = model.val()

    # Log metrics
    logger.info("=" * 60)
    logger.info("Training Results:")
    logger.info(f"  mAP@0.5: {metrics.box.map50:.3f}")
    logger.info(f"  mAP@0.5:0.95: {metrics.box.map:.3f}")
    logger.info(f"  Precision: {metrics.box.mp:.3f}")
    logger.info(f"  Recall: {metrics.box.mr:.3f}")
    logger.info("=" * 60)

    # Save best model with custom name
    best_model_path = f"models/yolov8-truck-blindspot.pt"
    os.makedirs("models", exist_ok=True)

    # Export best model
    model.export(format='torchscript')
    logger.info(f"Model exported to TorchScript")

    # Copy best weights
    import shutil
    best_weights = Path(train_config['project']) / train_config['name'] / 'weights' / 'best.pt'
    if best_weights.exists():
        shutil.copy(best_weights, best_model_path)
        logger.info(f"Best model saved to: {best_model_path}")

    # Save training metadata
    metadata = {
        'model_name': 'yolov8-truck-blindspot',
        'version': '1.0.0',
        'training_date': datetime.now().isoformat(),
        'epochs': args.epochs,
        'batch_size': args.batch,
        'image_size': args.img_size,
        'metrics': {
            'mAP@0.5': float(metrics.box.map50),
            'mAP@0.5:0.95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr)
        },
        'classes': 12,
        'truck_specific_classes': ['truck_blindspot_pedestrian', 'truck_blindspot_vehicle',
                                   'construction_worker', 'child'],
        'false_positive_classes': ['false_positive_shadow', 'false_positive_reflection']
    }

    with open('models/yolov8-truck-blindspot-metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("Training complete! Model ready for deployment.")

    return best_model_path, metadata


def test_model(model_path, test_images_path):
    """Test trained model on sample images"""
    logger.info("Testing model on sample images...")

    model = YOLO(model_path)
    test_dir = Path(test_images_path)

    if not test_dir.exists():
        logger.warning(f"Test directory not found: {test_images_path}")
        return

    test_images = list(test_dir.glob('*.jpg'))[:10]  # Test on first 10 images

    for img_path in test_images:
        results = model(str(img_path))

        # Save result with detections
        for r in results:
            output_path = f"runs/test/{img_path.name}"
            os.makedirs("runs/test", exist_ok=True)
            r.save(filename=output_path)

        logger.info(f"Tested: {img_path.name}")

    logger.info(f"Test results saved to runs/test/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom YOLOv8 for truck blind spots")
    parser.add_argument('--dataset', type=str, default='datasets/truck_blindspots',
                       help='Path to dataset directory')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Base model to start from')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--augment', action='store_true',
                       help='Apply data augmentation')
    parser.add_argument('--test', type=str, default=None,
                       help='Test images directory after training')

    args = parser.parse_args()

    # Train model
    model_path, metadata = train_custom_yolo(args)

    # Test if requested
    if args.test:
        test_model(model_path, args.test)

    logger.info("=" * 60)
    logger.info("Custom YOLOv8 training complete!")
    logger.info(f"Model: {model_path}")
    logger.info(f"Accuracy: {metadata['metrics']['mAP@0.5']:.1%}")
    logger.info("=" * 60)

