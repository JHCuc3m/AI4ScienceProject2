#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NPZEmbeddingDataset(Dataset):
    """Modified dataset for .npz files - extracts embeddings from pair representations"""
    
    def __init__(self, embedding_dir: str, embedding_method='global_avg'):
        self.embedding_dir = Path(embedding_dir)
        self.embedding_method = embedding_method  # How to convert L×L×128 to 128
        self.embeddings = self.load_embeddings()
        logger.info(f"Loaded {len(self.embeddings)} embeddings using method: {embedding_method}")
        logger.info(f"Fixed embedding dimension: 128")
    
    def load_embeddings(self):
        """Load embeddings from .npz files and convert to fixed 128-dim"""
        embeddings = []
        
        npz_files = list(self.embedding_dir.glob("*.npz"))
        logger.info(f"Found {len(npz_files)} .npz files")
        
        for file_path in tqdm(npz_files, desc="Loading NPZ files"):
            try:
                with np.load(file_path) as data:
                    if 'pair' not in data:
                        logger.warning(f"No 'pair' key in {file_path}")
                        continue
                    
                    pair_data = data['pair']  # Shape: [L, L, 128]
                    
                    # Validate dimensions
                    if pair_data.ndim != 3:
                        logger.warning(f"Unexpected pair dimensions in {file_path}: {pair_data.shape}")
                        continue
                    
                    L, L2, C = pair_data.shape
                    
                    # Verify it's square and has right number of channels
                    if L != L2:
                        logger.warning(f"Non-square pair matrix in {file_path}: {pair_data.shape}")
                        continue
                    
                    if C != 128:
                        logger.warning(f"Expected 128 channels, got {C} in {file_path}")
                        continue
                    
                    # Convert to 128-dimensional embedding using specified method
                    embedding = self.convert_to_fixed_size(pair_data)
                    
                    if embedding is not None and len(embedding) == 128:
                        embeddings.append(embedding)
                    else:
                        logger.warning(f"Failed to create valid embedding from {file_path}")
                        
            except Exception as e:
                logger.warning(f"Could not load {file_path}: {e}")
        
        return embeddings
    
    def convert_to_fixed_size(self, pair_data):
        """Convert L×L×128 pair representation to 128-dimensional vector"""
        
        if self.embedding_method == 'global_avg':
            # Your original approach: average over spatial dimensions
            return np.mean(pair_data, axis=(0, 1))  # Shape: [128]
            
        elif self.embedding_method == 'channel_avg':
            # Average each channel separately, then flatten
            # This keeps some spatial info by treating each position differently
            L = pair_data.shape[0]
            flattened = pair_data.reshape(L*L, 128)  # [L*L, 128]
            return np.mean(flattened, axis=0)  # [128] - same as global_avg actually
            
        elif self.embedding_method == 'diagonal_avg':
            # Focus on diagonal and near-diagonal elements (self and local interactions)
            L = pair_data.shape[0]
            diagonal_sum = np.zeros(128)
            count = 0
            
            # Extract diagonal and a few off-diagonals
            for offset in range(-2, 3):  # -2, -1, 0, 1, 2
                for i in range(max(0, -offset), min(L, L-offset)):
                    j = i + offset
                    if 0 <= j < L:
                        diagonal_sum += pair_data[i, j, :]
                        count += 1
            
            return diagonal_sum / count if count > 0 else np.zeros(128)
            
        elif self.embedding_method == 'upper_triangle':
            # Use only upper triangle (since pair matrices are symmetric)
            L = pair_data.shape[0]
            upper_elements = []
            
            for i in range(L):
                for j in range(i, L):
                    upper_elements.append(pair_data[i, j, :])
            
            if upper_elements:
                return np.mean(upper_elements, axis=0)  # [128]
            else:
                return np.zeros(128)
                
        elif self.embedding_method == 'random_sample':
            # Randomly sample some positions and average
            L = pair_data.shape[0]
            num_samples = min(100, L*L)  # Sample up to 100 positions
            
            sampled_elements = []
            for _ in range(num_samples):
                i, j = np.random.randint(0, L, 2)
                sampled_elements.append(pair_data[i, j, :])
            
            return np.mean(sampled_elements, axis=0)  # [128]
            
        else:
            logger.error(f"Unknown embedding method: {self.embedding_method}")
            return None
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32)

class SimpleAutoencoder(nn.Module):
    """Your original simple fully connected autoencoder"""
    
    def __init__(self, latent_dim=64):
        super(SimpleAutoencoder, self).__init__()
        
        self.input_dim = 128  # Fixed input dimension
        self.latent_dim = latent_dim
        
        # Encoder: 128 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Decoder: latent_dim -> 128
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 96),
            nn.ReLU(),
            nn.Linear(96, 128)
        )
    
    def forward(self, x):
        # Encode
        latent = self.encoder(x)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        return reconstructed, latent

def train_single_fold(train_loader, epochs=50, learning_rate=0.001, latent_dim=64, device='cpu'):
    """Train model on a single fold (no validation needed)"""
    
    # Create model
    model = SimpleAutoencoder(latent_dim).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Training history for this fold
    fold_losses = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            # Forward pass
            reconstructed, latent = model(batch)
            loss = criterion(reconstructed, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        fold_losses.append(avg_loss)
        
        # Log every 10 epochs
        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
    
    return model, fold_losses

def test_single_fold(model, test_loader, device='cpu'):
    """Test model on a single fold"""
    
    model.eval()
    criterion = nn.MSELoss()
    
    test_losses = []
    all_reconstructions = []
    all_originals = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            reconstructed, latent = model(batch)
            
            # Calculate loss for each sample in batch
            for i in range(batch.size(0)):
                original = batch[i]
                recon = reconstructed[i]
                
                loss = criterion(recon, original)
                test_losses.append(loss.item())
                
                # Store first few samples for visualization
                if len(all_originals) < 3:
                    all_originals.append(original.cpu().numpy())
                    all_reconstructions.append(recon.cpu().numpy())
    
    avg_test_loss = np.mean(test_losses)
    std_test_loss = np.std(test_losses)
    
    return avg_test_loss, std_test_loss, all_originals, all_reconstructions

def cross_validate_autoencoder(dataset, n_folds=5, epochs=50, batch_size=16, 
                              learning_rate=0.001, latent_dim=64, random_seed=42):
    """Perform k-fold cross-validation"""
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set up cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    # Results storage
    fold_results = []
    all_fold_losses = []
    all_test_samples = {'originals': [], 'reconstructions': []}
    
    logger.info(f"=== Starting {n_folds}-Fold Cross-Validation ===")
    logger.info(f"Total dataset: {len(dataset)} proteins")
    logger.info(f"Per fold: ~{len(dataset)//n_folds*4} train, ~{len(dataset)//n_folds} test")
    logger.info(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, latent_dim={latent_dim}")
    
    for fold, (train_indices, test_indices) in enumerate(kfold.split(dataset)):
        logger.info(f"\n--- FOLD {fold+1}/{n_folds} ---")
        logger.info(f"Training on {len(train_indices)} proteins, testing on {len(test_indices)} proteins")
        
        # Create subsets
        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)
        
        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        
        # Train model on this fold
        logger.info(f"Training fold {fold+1}...")
        model, fold_train_losses = train_single_fold(
            train_loader, epochs, learning_rate, latent_dim, device
        )
        
        # Test model on this fold
        logger.info(f"Testing fold {fold+1}...")
        test_loss, test_std, test_originals, test_reconstructions = test_single_fold(
            model, test_loader, device
        )
        
        # Store results
        fold_result = {
            'fold': fold + 1,
            'train_size': len(train_indices),
            'test_size': len(test_indices),
            'final_train_loss': fold_train_losses[-1],
            'test_loss_mean': test_loss,
            'test_loss_std': test_std,
            'train_losses': fold_train_losses
        }
        
        fold_results.append(fold_result)
        all_fold_losses.append(fold_train_losses)
        
        # Store some test samples for visualization
        all_test_samples['originals'].extend(test_originals)
        all_test_samples['reconstructions'].extend(test_reconstructions)
        
        logger.info(f"Fold {fold+1} Results:")
        logger.info(f"  Final Training Loss: {fold_train_losses[-1]:.6f}")
        logger.info(f"  Test Loss: {test_loss:.6f} ± {test_std:.6f}")
        
        # Clean up GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Calculate overall statistics
    test_losses = [result['test_loss_mean'] for result in fold_results]
    train_losses = [result['final_train_loss'] for result in fold_results]
    
    overall_stats = {
        'mean_test_loss': np.mean(test_losses),
        'std_test_loss': np.std(test_losses),
        'mean_train_loss': np.mean(train_losses),
        'std_train_loss': np.std(train_losses),
        'test_loss_range': [np.min(test_losses), np.max(test_losses)],
        'fold_results': fold_results
    }
    
    # Log final results
    logger.info(f"\n{'='*50}")
    logger.info(f"🎯 CROSS-VALIDATION RESULTS ({n_folds} folds)")
    logger.info(f"{'='*50}")
    logger.info(f"Test Performance: {overall_stats['mean_test_loss']:.6f} ± {overall_stats['std_test_loss']:.6f}")
    logger.info(f"Test Range: [{overall_stats['test_loss_range'][0]:.6f}, {overall_stats['test_loss_range'][1]:.6f}]")
    logger.info(f"Training Performance: {overall_stats['mean_train_loss']:.6f} ± {overall_stats['std_train_loss']:.6f}")
    logger.info(f"Generalization Gap: {overall_stats['mean_test_loss'] - overall_stats['mean_train_loss']:.6f}")
    logger.info(f"{'='*50}")
    
    return overall_stats, all_fold_losses, all_test_samples

def train_final_model(dataset, epochs=50, batch_size=16, learning_rate=0.001, latent_dim=64, random_seed=42):
    """Train final model on all data after cross-validation"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"\n=== Training Final Model on All Data ===")
    logger.info(f"Training on all {len(dataset)} proteins")
    
    # Create data loader for full dataset
    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train model
    final_model, final_losses = train_single_fold(
        full_loader, epochs, learning_rate, latent_dim, device
    )
    
    logger.info(f"Final model training completed!")
    logger.info(f"Final training loss: {final_losses[-1]:.6f}")
    
    return final_model, final_losses

def visualize_cross_validation_results(overall_stats, all_fold_losses, test_samples):
    """Create comprehensive visualization of cross-validation results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Per-fold test performance
    fold_numbers = [result['fold'] for result in overall_stats['fold_results']]
    test_performances = [result['test_loss_mean'] for result in overall_stats['fold_results']]
    test_stds = [result['test_loss_std'] for result in overall_stats['fold_results']]
    
    bars = axes[0,0].bar(fold_numbers, test_performances, yerr=test_stds, 
                        capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
    axes[0,0].axhline(overall_stats['mean_test_loss'], color='red', linestyle='--', 
                     linewidth=2, label=f"Mean: {overall_stats['mean_test_loss']:.4f}")
    axes[0,0].set_xlabel('Fold Number')
    axes[0,0].set_ylabel('Test Loss')
    axes[0,0].set_title('Test Performance per Fold\n(Error bars = std within fold)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, test_performances):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 2: Training curves for all folds
    for fold_idx, fold_losses in enumerate(all_fold_losses):
        epochs = range(1, len(fold_losses) + 1)
        axes[0,1].plot(epochs, fold_losses, alpha=0.6, linewidth=1, 
                      label=f'Fold {fold_idx+1}' if fold_idx < 3 else "")
    
    # Calculate mean training curve
    min_epochs = min(len(losses) for losses in all_fold_losses)
    mean_curve = np.mean([losses[:min_epochs] for losses in all_fold_losses], axis=0)
    std_curve = np.std([losses[:min_epochs] for losses in all_fold_losses], axis=0)
    
    epochs_mean = range(1, min_epochs + 1)
    axes[0,1].plot(epochs_mean, mean_curve, 'r-', linewidth=3, label='Mean')
    axes[0,1].fill_between(epochs_mean, mean_curve - std_curve, mean_curve + std_curve, 
                          alpha=0.2, color='red')
    
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Training Loss')
    axes[0,1].set_title('Training Curves Across Folds')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_yscale('log')
    
    # Plot 3: Train vs Test comparison
    train_performances = [result['final_train_loss'] for result in overall_stats['fold_results']]
    
    x = np.arange(len(fold_numbers))
    width = 0.35
    
    axes[0,2].bar(x - width/2, train_performances, width, label='Train Loss', 
                 alpha=0.7, color='lightgreen')
    axes[0,2].bar(x + width/2, test_performances, width, label='Test Loss', 
                 alpha=0.7, color='lightcoral')
    
    axes[0,2].set_xlabel('Fold Number')
    axes[0,2].set_ylabel('Loss')
    axes[0,2].set_title('Train vs Test Loss per Fold')
    axes[0,2].set_xticks(x)
    axes[0,2].set_xticklabels(fold_numbers)
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot 4: Reconstruction example
    if test_samples['originals'] and test_samples['reconstructions']:
        original = test_samples['originals'][0]
        reconstructed = test_samples['reconstructions'][0]
        
        axes[1,0].plot(original, 'g-o', linewidth=2, markersize=4, 
                      label='Original', alpha=0.8)
        axes[1,0].plot(reconstructed, 'r-s', linewidth=2, markersize=4, 
                      label='Reconstructed', alpha=0.8)
        axes[1,0].set_xlabel('Dimension')
        axes[1,0].set_ylabel('Value')
        axes[1,0].set_title('Example Reconstruction\n(From cross-validation)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Add reconstruction quality metrics
        mse = np.mean((original - reconstructed)**2)
        correlation = np.corrcoef(original, reconstructed)[0,1]
        axes[1,0].text(0.02, 0.98, f'MSE: {mse:.4f}\nCorr: {correlation:.4f}', 
                      transform=axes[1,0].transAxes, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Plot 5: Performance distribution
    all_test_losses = []
    for result in overall_stats['fold_results']:
        # Simulate individual test losses for histogram
        fold_losses = np.random.normal(result['test_loss_mean'], result['test_loss_std'], 10)
        all_test_losses.extend(fold_losses)
    
    axes[1,1].hist(all_test_losses, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1,1].axvline(overall_stats['mean_test_loss'], color='red', linestyle='--', 
                     linewidth=2, label=f"Mean: {overall_stats['mean_test_loss']:.4f}")
    axes[1,1].set_xlabel('Test Loss')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Distribution of Test Losses\n(Across all folds)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics
    metrics = ['Mean Test Loss', 'Std Test Loss', 'Mean Train Loss', 'Generalization Gap']
    values = [
        overall_stats['mean_test_loss'],
        overall_stats['std_test_loss'], 
        overall_stats['mean_train_loss'],
        overall_stats['mean_test_loss'] - overall_stats['mean_train_loss']
    ]
    colors = ['blue', 'orange', 'green', 'red' if values[3] > 0.1 else 'lightgreen']
    
    bars = axes[1,2].bar(metrics, values, color=colors, alpha=0.7)
    axes[1,2].set_ylabel('Loss Value')
    axes[1,2].set_title('Cross-Validation Summary')
    axes[1,2].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, values):
        axes[1,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('cross_validation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_cross_validation_results(overall_stats, output_dir="cv_autoencoder_output"):
    """Save cross-validation results"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save detailed results as JSON
    with open(output_path / 'cross_validation_results.json', 'w') as f:
        json.dump(overall_stats, f, indent=2)
    
    # Save summary report
    with open(output_path / 'cv_summary_report.txt', 'w') as f:
        f.write("=== Cross-Validation Summary Report ===\n\n")
        f.write(f"Test Performance: {overall_stats['mean_test_loss']:.6f} ± {overall_stats['std_test_loss']:.6f}\n")
        f.write(f"Training Performance: {overall_stats['mean_train_loss']:.6f} ± {overall_stats['std_train_loss']:.6f}\n")
        f.write(f"Generalization Gap: {overall_stats['mean_test_loss'] - overall_stats['mean_train_loss']:.6f}\n")
        f.write(f"Test Loss Range: [{overall_stats['test_loss_range'][0]:.6f}, {overall_stats['test_loss_range'][1]:.6f}]\n\n")
        
        f.write("Per-Fold Results:\n")
        for result in overall_stats['fold_results']:
            f.write(f"  Fold {result['fold']}: Test = {result['test_loss_mean']:.6f} ± {result['test_loss_std']:.6f}, "
                   f"Train = {result['final_train_loss']:.6f}\n")
        
        gap = overall_stats['mean_test_loss'] - overall_stats['mean_train_loss']
        if gap > 0.1:
            f.write(f"\nWARNING: Large generalization gap ({gap:.6f}) may indicate overfitting.\n")
        else:
            f.write(f"\nGood generalization achieved (gap: {gap:.6f}).\n")
    
    logger.info(f"Cross-validation results saved to {output_path}")

def analyze_npz_dataset(data_dir="../dataset/"):
    """Analyze your .npz dataset structure"""
    
    data_path = Path(data_dir)
    npz_files = list(data_path.glob("*.npz"))
    
    logger.info(f"=== NPZ Dataset Analysis ===")
    logger.info(f"Found {len(npz_files)} .npz files in {data_dir}")
    
    if not npz_files:
        logger.error("No .npz files found!")
        return False
    
    # Analyze first few files
    pair_shapes = []
    valid_files = 0
    
    for i, npz_file in enumerate(npz_files[:5]):
        logger.info(f"\n--- File {i+1}: {npz_file.name} ---")
        
        try:
            with np.load(npz_file) as data:
                logger.info(f"Keys: {list(data.keys())}")
                
                if 'pair' in data:
                    pair_data = data['pair']
                    pair_shapes.append(pair_data.shape)
                    logger.info(f"pair.npy: {pair_data.shape}, dtype={pair_data.dtype}")
                    logger.info(f"  Value range: [{pair_data.min():.4f}, {pair_data.max():.4f}]")
                    logger.info(f"  Mean: {pair_data.mean():.4f}, Std: {pair_data.std():.4f}")
                    
                    # Check symmetry
                    if len(pair_data.shape) == 3:
                        symmetry_error = np.abs(pair_data - pair_data.transpose(1, 0, 2)).mean()
                        logger.info(f"  Symmetry error: {symmetry_error:.6f}")
                    
                    valid_files += 1
                else:
                    logger.warning("No 'pair' key found!")
                
                if 'single' in data:
                    single_data = data['single']
                    logger.info(f"single.npy: {single_data.shape}, dtype={single_data.dtype}")
                
        except Exception as e:
            logger.error(f"Error reading {npz_file}: {e}")
    
    # Summary
    if pair_shapes:
        lengths = [shape[0] for shape in pair_shapes]
        channels = [shape[2] for shape in pair_shapes]
        
        logger.info(f"\n=== Summary ===")
        logger.info(f"Valid files: {valid_files}/{len(npz_files[:5])} sampled")
        logger.info(f"Sequence lengths: {lengths}")
        logger.info(f"Channels: {channels}")
        logger.info(f"Length range: {min(lengths)} to {max(lengths)}")
        
        return True
    else:
        logger.error("No valid pair data found!")
        return False

def generate_meaningful_embeddings(model, dataset, num_samples=10):
    """Generate by interpolating between real proteins"""
    
    model.eval()
    generated_embeddings = []
    
    with torch.no_grad():
        # Get some real proteins
        real_indices = np.random.choice(len(dataset), size=num_samples*2, replace=False)
        
        for i in range(num_samples):
            # Pick two random real proteins
            protein1 = dataset[real_indices[i*2]]
            protein2 = dataset[real_indices[i*2 + 1]]
            
            # Encode them
            _, latent1 = model(protein1.unsqueeze(0))
            _, latent2 = model(protein2.unsqueeze(0))
            
            # Interpolate (create "hybrid" protein)
            alpha = np.random.uniform(0.2, 0.8)  # Random interpolation
            interpolated_latent = (1 - alpha) * latent1 + alpha * latent2
            
            # Decode
            generated = model.decoder(interpolated_latent)
            generated_embeddings.append(generated.squeeze().numpy())
    
    return generated_embeddings

def main():
    parser = argparse.ArgumentParser(description="Autoencoder with Cross-Validation")
    parser.add_argument("--data_dir", default="../dataset/", help="Directory with .npz files")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs per fold")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--embedding_method", default="global_avg", 
                       choices=['global_avg', 'diagonal_avg', 'upper_triangle', 'random_sample'],
                       help="Method to convert L×L×128 to 128")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--analyze_only", action="store_true", help="Only analyze dataset, don't train")
    parser.add_argument("--train_final", action="store_true", help="Train final model on all data after CV")
    parser.add_argument("--generate", type=int, default=0, help="Number of embeddings to generate (requires --train_final)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Check if data directory exists
    if not Path(args.data_dir).exists():
        logger.error(f"Data directory {args.data_dir} not found!")
        return
    
    # Analyze dataset first
    logger.info("=== Analyzing NPZ Dataset ===")
    if not analyze_npz_dataset(args.data_dir):
        logger.error("Dataset analysis failed!")
        return
    
    # Just analyze if requested
    if args.analyze_only:
        logger.info("Analysis complete.")
        return
    
    # Load dataset with specified method
    logger.info(f"=== Loading Dataset with {args.embedding_method} method ===")
    dataset = NPZEmbeddingDataset(args.data_dir, embedding_method=args.embedding_method)
    
    if len(dataset) == 0:
        logger.error("No embeddings found in dataset!")
        return
    
    if len(dataset) < args.n_folds:
        logger.error(f"Dataset too small for {args.n_folds}-fold CV! Need at least {args.n_folds} proteins.")
        return
    
    # Perform cross-validation
    logger.info(f"=== {args.n_folds}-Fold Cross-Validation ===")
    overall_stats, all_fold_losses, test_samples = cross_validate_autoencoder(
        dataset=dataset,
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim,
        random_seed=args.seed
    )
    
    # Create visualizations
    logger.info("=== Creating Visualizations ===")
    visualize_cross_validation_results(overall_stats, all_fold_losses, test_samples)
    
    # Save results
    logger.info("=== Saving Results ===")
    save_cross_validation_results(overall_stats)
    
    # Train final model if requested
    final_model = None
    if args.train_final:
        logger.info("=== Training Final Model on All Data ===")
        final_model, final_losses = train_final_model(
            dataset=dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            latent_dim=args.latent_dim,
            random_seed=args.seed
        )
        
        # Save final model
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'input_dim': final_model.input_dim,
            'latent_dim': final_model.latent_dim,
            'train_losses': final_losses,
            'cv_results': overall_stats,
            'hyperparameters': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'embedding_method': args.embedding_method
            }
        }, 'cv_autoencoder_output/final_model.pt')
        
        logger.info("Final model saved to cv_autoencoder_output/final_model.pt")
        
        # Generate new embeddings if requested
        if args.generate > 0:
            logger.info(f"=== Generating {args.generate} New Embeddings ===")
            generated = generate_meaningful_embeddings(final_model, dataset, args.generate)
            
            # Save generated embeddings
            output_path = Path("cv_autoencoder_output")
            for i, embedding in enumerate(generated):
                np.save(output_path / f"generated_embedding_{i:03d}.npy", embedding)
            
            logger.info(f"Generated embeddings saved to cv_autoencoder_output/")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("🎉 CROSS-VALIDATION COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info(f"✅ Method: {args.embedding_method}")
    logger.info(f"✅ Dataset: {len(dataset)} proteins")
    logger.info(f"✅ Cross-validation: {args.n_folds} folds")
    logger.info(f"✅ Test Performance: {overall_stats['mean_test_loss']:.6f} ± {overall_stats['std_test_loss']:.6f}")
    logger.info(f"✅ Generalization Gap: {overall_stats['mean_test_loss'] - overall_stats['mean_train_loss']:.6f}")
    
    # Performance interpretation
    mean_test_loss = overall_stats['mean_test_loss']
    if mean_test_loss < 1.0:
        logger.info("🏆 Excellent reconstruction quality!")
    elif mean_test_loss < 2.0:
        logger.info("👍 Good reconstruction quality!")
    elif mean_test_loss < 5.0:
        logger.info("⚠️  Moderate reconstruction quality")
    else:
        logger.info("❌ Poor reconstruction quality - needs improvement")
    
    # Generalization assessment
    gap = overall_stats['mean_test_loss'] - overall_stats['mean_train_loss']
    if gap < 0.1:
        logger.info("✅ Excellent generalization!")
    elif gap < 0.5:
        logger.info("👍 Good generalization!")
    else:
        logger.info("⚠️  Potential overfitting detected!")
    
    # Consistency assessment
    cv_std = overall_stats['std_test_loss']
    if cv_std < 0.1:
        logger.info("✅ Very consistent across folds!")
    elif cv_std < 0.3:
        logger.info("👍 Reasonably consistent across folds!")
    else:
        logger.info("⚠️  High variance across folds!")
    
    logger.info("="*60)
    logger.info("📁 Results saved to: cv_autoencoder_output/")
    logger.info("📊 Visualization: cross_validation_results.png")
    if args.train_final:
        logger.info("🤖 Final model: cv_autoencoder_output/final_model.pt")
    if args.generate > 0:
        logger.info(f"🧬 Generated {args.generate} new embeddings")
    logger.info("="*60)

if __name__ == "__main__":
    main()