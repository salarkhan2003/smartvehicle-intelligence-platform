"""
Seatbelt Detection Training - YOLOv8 + Pose Estimation
Train on 1500 car interior images for seatbelt visible/not visible
Target: 91% accuracy using shoulder/torso keypoints
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import yaml
import json
from datetime import datetime
import shutil
from tqdm import tqdm
import mediapipe as mp


class SeatbeltDetectionTrainer:
    """Train YOLOv8 + MediaPipe pose for seatbelt detection"""

    def __init__(self, dataset_path='seatbelt_data/', epochs=50):
        self.dataset_path = Path(dataset_path)
        self.epochs = epochs
        self.img_size = 640
        self.batch_size = 16
        self.model_name = 'yolov8n.pt'

        # Classes
        self.classes = {
            0: 'driver_seatbelt',
            1: 'driver_no_seatbelt',
            2: 'passenger_seatbelt',
            3: 'passenger_no_seatbelt'
        }

        # MediaPipe pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        )

        print(f"🪑 Seatbelt Detection Trainer Initialized")
        print(f"Dataset: {self.dataset_path}")
        print(f"Epochs: {self.epochs}")

    def create_dataset_structure(self):
        """Create YOLO dataset structure"""
        print("\n📁 Creating dataset structure...")

        dirs = [
            self.dataset_path / 'images' / 'train',
            self.dataset_path / 'images' / 'val',
            self.dataset_path / 'labels' / 'train',
            self.dataset_path / 'labels' / 'val'
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        print("✓ Dataset structure created")

    def generate_synthetic_data(self, num_images=1500):
        """Generate synthetic car interior images"""
        print(f"\n🎨 Generating {num_images} synthetic car interior images...")

        train_split = int(num_images * 0.8)

        for i in tqdm(range(num_images), desc="Generating images"):
            # Create car interior scene
            img = np.random.randint(30, 80, (640, 640, 3), dtype=np.uint8)

            # Dashboard area (darker)
            img[500:, :] = img[500:, :] // 2 + 20

            # Steering wheel (dark circle)
            cv2.circle(img, (320, 520), 80, (40, 40, 40), -1)
            cv2.circle(img, (320, 520), 60, (50, 50, 50), -1)

            labels = []

            # Generate driver (always present)
            # Driver position: left side (Western cars)
            driver_x = 0.35
            driver_y = 0.4
            driver_w = 0.25
            driver_h = 0.5

            # Random seatbelt (70% wearing, 30% not wearing)
            has_seatbelt = np.random.random() > 0.3

            # Draw person torso
            torso_x1 = int((driver_x - driver_w/2) * 640)
            torso_y1 = int((driver_y - driver_h/2) * 640)
            torso_x2 = int((driver_x + driver_w/2) * 640)
            torso_y2 = int((driver_y + driver_h/2) * 640)

            # Clothing color
            clothing_color = tuple(np.random.randint(80, 200, 3).tolist())
            cv2.rectangle(img, (torso_x1, torso_y1), (torso_x2, torso_y2),
                         clothing_color, -1)

            # Draw head
            head_y = torso_y1 - 30
            cv2.circle(img, (int(driver_x * 640), head_y), 25, (180, 150, 120), -1)

            # Seatbelt strap (diagonal from shoulder)
            if has_seatbelt:
                shoulder_x = torso_x1 + 20
                shoulder_y = torso_y1 + 30
                hip_x = torso_x2 - 20
                hip_y = torso_y2 - 20

                # Draw seatbelt (gray diagonal stripe)
                cv2.line(img, (shoulder_x, shoulder_y), (hip_x, hip_y),
                        (100, 100, 100), 15)
                class_id = 0  # driver_seatbelt
            else:
                class_id = 1  # driver_no_seatbelt

            # YOLO format
            labels.append(f"{class_id} {driver_x} {driver_y} {driver_w} {driver_h}")

            # Sometimes add passenger (30% chance)
            if np.random.random() > 0.7:
                pass_x = 0.65
                pass_y = 0.4
                pass_w = 0.25
                pass_h = 0.5

                has_pass_seatbelt = np.random.random() > 0.4

                pass_x1 = int((pass_x - pass_w/2) * 640)
                pass_y1 = int((pass_y - pass_h/2) * 640)
                pass_x2 = int((pass_x + pass_w/2) * 640)
                pass_y2 = int((pass_y + pass_h/2) * 640)

                pass_color = tuple(np.random.randint(80, 200, 3).tolist())
                cv2.rectangle(img, (pass_x1, pass_y1), (pass_x2, pass_y2),
                            pass_color, -1)

                cv2.circle(img, (int(pass_x * 640), pass_y1 - 30), 25,
                          (180, 150, 120), -1)

                if has_pass_seatbelt:
                    cv2.line(img, (pass_x1 + 20, pass_y1 + 30),
                            (pass_x2 - 20, pass_y2 - 20), (100, 100, 100), 15)
                    class_id = 2  # passenger_seatbelt
                else:
                    class_id = 3  # passenger_no_seatbelt

                labels.append(f"{class_id} {pass_x} {pass_y} {pass_w} {pass_h}")

            # Add some noise
            noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Save
            split = 'train' if i < train_split else 'val'
            img_path = self.dataset_path / 'images' / split / f'seatbelt_{i:06d}.jpg'
            label_path = self.dataset_path / 'labels' / split / f'seatbelt_{i:06d}.txt'

            cv2.imwrite(str(img_path), img)
            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))

        print(f"✓ Generated {train_split} training + {num_images-train_split} validation images")

    def create_data_yaml(self):
        """Create data.yaml"""
        print("\n📝 Creating data.yaml...")

        data_yaml = {
            'path': str(self.dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 4,
            'names': ['driver_seatbelt', 'driver_no_seatbelt',
                     'passenger_seatbelt', 'passenger_no_seatbelt']
        }

        yaml_path = self.dataset_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"✓ data.yaml created")
        return yaml_path

    def train_model(self):
        """Train YOLOv8 seatbelt detection"""
        print(f"\n🚀 Training YOLOv8 Seatbelt Detection Model...")

        model = YOLO(self.model_name)

        results = model.train(
            data=str(self.dataset_path / 'data.yaml'),
            epochs=self.epochs,
            imgsz=self.img_size,
            batch=self.batch_size,
            name='seatbelt_detection',
            patience=10,
            save=True,
            device='cpu',
            workers=4,
            pretrained=True,
            optimizer='Adam',
            verbose=True
        )

        print("✓ Training complete!")
        return results

    def test_pose_estimation(self):
        """Test MediaPipe pose for shoulder angle detection"""
        print("\n🧍 Testing pose estimation for seatbelt detection...")

        # Create test image with person
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)

        # Draw simplified person
        cv2.rectangle(test_img, (200, 200), (400, 500), (150, 150, 150), -1)
        cv2.circle(test_img, (300, 150), 50, (180, 150, 120), -1)

        # Seatbelt strap
        cv2.line(test_img, (220, 230), (380, 470), (100, 100, 100), 20)

        # Process with MediaPipe
        results = self.pose.process(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            print("✓ Pose detected!")

            # Key landmarks for seatbelt
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

            print(f"  Left shoulder: ({left_shoulder.x:.2f}, {left_shoulder.y:.2f})")
            print(f"  Right shoulder: ({right_shoulder.x:.2f}, {right_shoulder.y:.2f})")

            # Calculate shoulder angle
            angle = np.degrees(np.arctan2(
                right_shoulder.y - left_shoulder.y,
                right_shoulder.x - left_shoulder.x
            ))
            print(f"  Shoulder angle: {angle:.1f}°")
            print(f"  Seatbelt indicator: {'Present' if abs(angle) > 30 else 'Absent'}")
        else:
            print("⚠️  No pose detected")

    def validate_model(self, model_path='runs/detect/seatbelt_detection/weights/best.pt'):
        """Validate trained model"""
        print(f"\n✅ Validating model...")

        if not Path(model_path).exists():
            print(f"⚠️  Model not found: {model_path}")
            return None

        model = YOLO(model_path)
        metrics = model.val(data=str(self.dataset_path / 'data.yaml'))

        print(f"\n📊 Validation Metrics:")
        print(f"  mAP50: {metrics.box.map50:.3f}")
        print(f"  mAP50-95: {metrics.box.map:.3f}")
        print(f"  Precision: {metrics.box.mp:.3f}")
        print(f"  Recall: {metrics.box.mr:.3f}")

        return metrics

    def export_model(self, model_path='runs/detect/seatbelt_detection/weights/best.pt',
                    output_path='models/seatbelt_detector.pt'):
        """Export model"""
        print(f"\n📦 Exporting model...")

        if not Path(model_path).exists():
            print(f"⚠️  Model not found")
            return

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(model_path, output_path)

        metadata = {
            'model_type': 'yolov8n_seatbelt_detection',
            'classes': self.classes,
            'pose_estimation': 'MediaPipe',
            'trained_epochs': self.epochs,
            'created_at': datetime.now().isoformat(),
            'target_accuracy': '91%'
        }

        with open(Path(output_path).parent / 'seatbelt_model_info.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Model exported to: {output_path}")

    def run_full_pipeline(self):
        """Execute complete training pipeline"""
        print("=" * 70)
        print("🪑 SEATBELT DETECTION TRAINING PIPELINE")
        print("=" * 70)

        self.create_dataset_structure()
        self.generate_synthetic_data(num_images=1500)
        self.create_data_yaml()
        self.train_model()
        self.test_pose_estimation()
        self.validate_model()
        self.export_model()

        print("\n" + "=" * 70)
        print("✅ SEATBELT DETECTION TRAINING COMPLETE")
        print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train Seatbelt Detection Model')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--dataset', type=str, default='seatbelt_data/')

    args = parser.parse_args()

    trainer = SeatbeltDetectionTrainer(dataset_path=args.dataset, epochs=args.epochs)
    trainer.run_full_pipeline()


if __name__ == '__main__':
    main()

