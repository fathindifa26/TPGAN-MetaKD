#!/usr/bin/env python3
"""
Testing script untuk EfficientNet feature extractors
Mengukur AUC, similarity intra/inter class, dan gap dengan paired sample 500
"""

import torch
import numpy as np
import os
import time
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from models import feature_extract_network
import importlib
import json

def load_efficientnet_model(model_size='b0'):
    """Load EfficientNet model dari timm"""
    model_name = f'efficientnet_{model_size}'
    
    # Load config
    pretrain_config = importlib.import_module('feature_extractor_models.efficientnet.pretrain_config')
    kwargs = pretrain_config.stem.copy()
    kwargs.pop('model_name', None)
    
    # Create model
    model = eval('feature_extract_network.' + model_name)(**kwargs)
    
    # EfficientNet dari timm sudah pretrained, tidak perlu load checkpoint
    model.eval()
    return model

def create_dataset(dataset_path, transform=None):
    """Create dataset dari MultiPie (gambar langsung di dalam folder subject)"""
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    images = []
    labels = []
    paths = []
    for subject_id in os.listdir(dataset_path):
        subject_path = os.path.join(dataset_path, subject_id)
        if not os.path.isdir(subject_path):
            continue
        # Cari file gambar langsung di subject_path
        for img_file in os.listdir(subject_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(subject_path, img_file)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img)
                    images.append(img_tensor)
                    labels.append(int(subject_id))
                    paths.append(img_path)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    return images, labels, paths

def extract_features(model, dataloader, device):
    """Extract features dari semua gambar"""
    features = []
    labels = []
    paths = []
    
    model.to(device)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            images, batch_labels, batch_paths = batch
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            if isinstance(outputs, tuple):
                batch_features = outputs[1]  # Ambil features, bukan logits
            else:
                batch_features = outputs
            
            # Convert ke numpy
            batch_features = batch_features.cpu().numpy()
            
            features.extend(batch_features)
            labels.extend(batch_labels)
            paths.extend(batch_paths)
    
    return np.array(features), labels, paths

def calculate_similarity_matrix(features):
    """Hitung similarity matrix"""
    # Normalize features
    features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    # Calculate cosine similarity
    similarity_matrix = np.dot(features_norm, features_norm.T)
    
    return similarity_matrix

def calculate_metrics(features, labels, paths, num_pairs=500):
    """Hitung AUC, similarity intra/inter class, dan gap"""
    print(f"Calculating metrics for {len(features)} samples...")
    
    # Calculate similarity matrix
    similarity_matrix = calculate_similarity_matrix(features)
    
    # Create pairs
    np.random.seed(42)  # Untuk reproducibility
    n_samples = len(features)
    
    # Generate random pairs
    pair_indices = []
    for _ in range(num_pairs):
        i = np.random.randint(0, n_samples)
        j = np.random.randint(0, n_samples)
        while i == j:
            j = np.random.randint(0, n_samples)
        pair_indices.append((i, j))
    
    # Calculate similarities for pairs
    similarities = []
    pair_labels = []  # 1 for same class, 0 for different class
    
    for i, j in pair_indices:
        similarity = similarity_matrix[i, j]
        similarities.append(similarity)
        
        # Check if same class
        if labels[i] == labels[j]:
            pair_labels.append(1)
        else:
            pair_labels.append(0)
    
    similarities = np.array(similarities)
    pair_labels = np.array(pair_labels)
    
    # Calculate AUC
    try:
        auc = roc_auc_score(pair_labels, similarities)
    except ValueError:
        auc = 0.5  # Fallback jika tidak ada positive/negative samples
    
    # Calculate intra-class and inter-class similarities
    intra_similarities = similarities[pair_labels == 1]
    inter_similarities = similarities[pair_labels == 0]
    
    intra_mean = np.mean(intra_similarities) if len(intra_similarities) > 0 else 0
    inter_mean = np.mean(inter_similarities) if len(inter_similarities) > 0 else 0
    
    # Calculate gap
    gap = intra_mean - inter_mean
    
    return {
        'auc': auc,
        'intra_class_similarity': intra_mean,
        'inter_class_similarity': inter_mean,
        'gap': gap,
        'num_intra_pairs': len(intra_similarities),
        'num_inter_pairs': len(inter_similarities)
    }

def main():
    parser = argparse.ArgumentParser(description='Test EfficientNet feature extractor')
    parser.add_argument('--model_size', type=str, default='b0', 
                       choices=['b0', 'b1', 'b2', 'b3', 'b4'],
                       help='EfficientNet model size')
    parser.add_argument('--dataset_path', type=str, default='MultiPie',
                       help='Path to MultiPie dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for feature extraction')
    parser.add_argument('--num_pairs', type=int, default=500,
                       help='Number of pairs for evaluation')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        print(f"Dataset path {args.dataset_path} tidak ditemukan!")
        return
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading EfficientNet-{args.model_size.upper()}...")
    model = load_efficientnet_model(args.model_size)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create dataset
    print("Creating dataset...")
    images, labels, paths = create_dataset(args.dataset_path)
    print(f"Found {len(images)} images from {len(set(labels))} subjects")
    
    # Create dataloader
    dataset = list(zip(images, labels, paths))
    dataloader = []
    for i in range(0, len(dataset), args.batch_size):
        batch = dataset[i:i + args.batch_size]
        batch_images = torch.stack([item[0] for item in batch])
        batch_labels = [item[1] for item in batch]
        batch_paths = [item[2] for item in batch]
        dataloader.append((batch_images, batch_labels, batch_paths))
    
    # Extract features
    print("Extracting features...")
    start_time = time.time()
    features, labels, paths = extract_features(model, dataloader, device)
    extraction_time = time.time() - start_time
    
    print(f"Feature extraction completed in {extraction_time:.2f} seconds")
    print(f"Feature dimension: {features.shape[1]}")
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(features, labels, paths, args.num_pairs)
    
    # Print results
    print("\n" + "="*50)
    print(f"EfficientNet-{args.model_size.upper()} Results")
    print("="*50)
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Intra-class similarity: {metrics['intra_class_similarity']:.4f}")
    print(f"Inter-class similarity: {metrics['inter_class_similarity']:.4f}")
    print(f"Gap: {metrics['gap']:.4f}")
    print(f"Intra-class pairs: {metrics['num_intra_pairs']}")
    print(f"Inter-class pairs: {metrics['num_inter_pairs']}")
    print(f"Feature extraction time: {extraction_time:.2f}s")
    print(f"Average time per sample: {extraction_time/len(features):.4f}s")
    
    # Save results
    if args.output_file:
        results = {
            'model': f'EfficientNet-{args.model_size.upper()}',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'feature_dimension': features.shape[1],
            'num_samples': len(features),
            'num_subjects': len(set(labels)),
            'metrics': metrics,
            'extraction_time': extraction_time,
            'avg_time_per_sample': extraction_time/len(features)
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

if __name__ == '__main__':
    main() 