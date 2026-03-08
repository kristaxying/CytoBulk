import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
import json
import warnings
import sys
from tqdm import tqdm
from os.path import exists
from typing import Callable, Generator
import torch.distributed as dist
from scipy.stats import pearsonr
from ... import utils
from ... import get
from ... import plots
from ... import preprocessing
import pickle
import random
import os
from sklearn.preprocessing import normalize

# Add UMAP related imports
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# Add PCA related imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Const:
    """
    Some constants used in the class.
    """
    MODE_TRAINING = "training"
    MODE_PREDICTION = "prediction"
    SAMPLE_COL = "sample_name"
    GENE_SYMBOL_COL = "GeneSymbol"
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001  # Lower learning rate for better stability
    MAX_SPLIT = 10
    EPOCH_NUM_BULK = 15
    EPOCH_NUM_ST = 20
    SEED = 21
    CHEB_MODE = 1
    MAX_RETRY = 1
    L2_WEIGHT = 1e-4  # L2 regularization weight


def configure_device(use_gpu):
    if use_gpu:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"





def improved_kl_divergence_loss_bk(pred, target, eps=1e-8, temperature=1.0):
    """
    Improved KL divergence loss function with better numerical stability
    """
    pred = torch.clamp(pred, min=eps, max=1 - eps)
    target = torch.clamp(target, min=eps, max=1 - eps)

    if temperature != 1.0:
        pred = pred ** (1.0 / temperature)
        pred = pred / pred.sum(dim=1, keepdim=True)

    log_pred = torch.log(pred + eps)
    kl_loss = F.kl_div(log_pred, target, reduction='batchmean')
    return kl_loss

def improved_kl_divergence_loss(pred, target, eps=1e-12, temperature=1.0):
    """
    Computes KL(target || pred) using F.kl_div(log_pred, target).
    pred is probability (softmax output).
    """
    if temperature != 1.0:
        pred = torch.clamp(pred, min=eps)
        pred = pred.pow(1.0 / temperature)
        pred = pred / pred.sum(dim=1, keepdim=True)


    pred = torch.clamp(pred, min=eps)       # only lower bound
    log_pred = torch.log(pred)              # log-probabilities

    return F.kl_div(log_pred, target, reduction="batchmean", log_target=False)

def smoothed_kl_loss(pred, target, eps=1e-8, alpha=0.1):
    """
    KL divergence loss with label smoothing
    """
    smooth_target = (1 - alpha) * target + alpha / target.size(1)
    pred = torch.clamp(pred, min=eps, max=1 - eps)
    log_pred = torch.log(pred + eps)
    kl_loss = F.kl_div(log_pred, smooth_target, reduction='batchmean')
    return kl_loss


def hybrid_loss(pred, target, model, kl_weight=0.7, mse_weight=0.3, eps=1e-8):
    """
    Hybrid loss function: KL divergence + MSE, balancing distribution similarity and numerical accuracy
    """
    kl_loss = improved_kl_divergence_loss(pred, target, eps=eps)
    mse_loss = F.mse_loss(pred, target)
    total_loss = kl_weight * kl_loss + mse_weight * mse_loss
    return total_loss


def combined_loss(pred, target, model, loss_type='hybrid', kl_weight=0.7, mse_weight=0.3):
    """
    Improved combined loss function supporting multiple loss types
    """
    if loss_type == 'kl_only':
        return improved_kl_divergence_loss(pred, target)
    elif loss_type == 'mse_only':
        return F.mse_loss(pred, target)
    elif loss_type == 'hybrid':
        return hybrid_loss(pred, target, model, kl_weight, mse_weight)
    elif loss_type == 'smooth_kl':
        return smoothed_kl_loss(pred, target)
    elif loss_type == 'focal_mse':
        mse = (pred - target) ** 2
        focal_weight = (1 + mse) ** 0.5
        return (focal_weight * mse).mean()
    else:
        return improved_kl_divergence_loss(pred, target)


def calculate_batch_info(dataset_size, batch_size, min_last_batch=10):
    """
    Calculate batch info and determine if last batch should be dropped
    """
    full_batches = dataset_size // batch_size
    last_batch_size = dataset_size % batch_size
    drop_last = last_batch_size > 0 and last_batch_size < min_last_batch

    return {
        'drop_last': drop_last,
        'last_batch_size': last_batch_size,
        'dropped_samples': last_batch_size if drop_last else 0
    }


def create_safe_dataloader(dataset, batch_size, shuffle=True, min_last_batch=10):
    """
    Create DataLoader with smart drop_last based on minimum batch size
    """
    batch_info = calculate_batch_info(len(dataset), batch_size, min_last_batch)
    '''
    if batch_info['dropped_samples'] > 0:
        print(f"Dropping last batch: {batch_info['last_batch_size']} samples < {min_last_batch} threshold")
    '''

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=batch_info['drop_last']
    )


def plot_umap_embeddings(model, graph_list, X_sim, X_real, epoch, out_dir, n_samples=10000):
    """
    Plot UMAP visualization of simulation and real data embeddings
    """
    model.eval()

    with torch.no_grad():
        if len(X_sim) > n_samples:
            sim_indices = torch.randperm(len(X_sim))[:n_samples]
            X_sim_sample = X_sim[sim_indices]
        else:
            X_sim_sample = X_sim

        if len(X_real) > n_samples:
            real_indices = torch.randperm(len(X_real))[:n_samples]
            X_real_sample = X_real[real_indices]
        else:
            X_real_sample = X_real

        sim_embeddings = []
        real_embeddings = []

        batch_size = 64

        for i in range(0, len(X_sim_sample), batch_size):
            batch = X_sim_sample[i:i + batch_size]
            x = model.graph_nets[0].reduce1(batch)
            x = model.graph_nets[0].act(x)
            x = model.graph_nets[0].norm1(x)
            x = model.graph_nets[0].act(x)
            x = model.graph_nets[0].reduce2(x)
            if hasattr(model.graph_nets[0], 'norm2'):
                x = model.graph_nets[0].norm2(x)

            B, N = x.size(0), x.size(1)
            x = x.view(B, 1, N)
            emb = model.graph_nets[0].act(model.graph_nets[0].conv1(x, graph_list[0]))
            sim_embeddings.append(emb.squeeze(1).cpu())

        for i in range(0, len(X_real_sample), batch_size):
            batch = X_real_sample[i:i + batch_size]
            x = model.graph_nets[0].reduce1(batch)
            x = model.graph_nets[0].act(x)
            x = model.graph_nets[0].norm1(x)
            x = model.graph_nets[0].act(x)
            x = model.graph_nets[0].reduce2(x)
            if hasattr(model.graph_nets[0], 'norm2'):
                x = model.graph_nets[0].norm2(x)

            B, N = x.size(0), x.size(1)
            x = x.view(B, 1, N)
            emb = model.graph_nets[0].act(model.graph_nets[0].conv1(x, graph_list[0]))
            real_embeddings.append(emb.squeeze(1).cpu())

        sim_embeddings = torch.cat(sim_embeddings, dim=0).numpy()
        real_embeddings = torch.cat(real_embeddings, dim=0).numpy()

        all_embeddings = np.vstack([sim_embeddings, real_embeddings])

        if len(all_embeddings) < 15:
            print(f"Not enough samples for UMAP at epoch {epoch}: {len(all_embeddings)}")
            model.train()
            return

        try:
            reducer = umap.UMAP(
                n_neighbors=min(15, len(all_embeddings) - 1),
                min_dist=0.1,
                random_state=42,
                n_components=2
            )
            embedding_2d = reducer.fit_transform(all_embeddings)

            plt.figure(figsize=(10, 8))

            sim_points = embedding_2d[:len(sim_embeddings)]
            real_points = embedding_2d[len(sim_embeddings):]

            plt.scatter(sim_points[:, 0], sim_points[:, 1],
                        c='#fb8500', alpha=0.6, s=20, label='Simulation Data')
            plt.scatter(real_points[:, 0], real_points[:, 1],
                        c='#023047', alpha=0.6, s=20, label='Real Data')

            plt.title('UMAP Visualization of Graph Embeddings')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.legend()
            plt.grid(True, alpha=0.3)

            umap_dir = os.path.join(out_dir, "umap_plots")
            os.makedirs(umap_dir, exist_ok=True)
            plt.savefig(os.path.join(umap_dir, f'umap_epoch_{epoch + 1:03d}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

            from scipy.spatial.distance import cdist
            sim_center = np.mean(sim_points, axis=0)
            real_center = np.mean(real_points, axis=0)
            center_distance = np.linalg.norm(sim_center - real_center)

            cross_distances = cdist(sim_points, real_points)
            avg_min_distance = np.mean(np.min(cross_distances, axis=1))

            stats_file = os.path.join(umap_dir, 'distance_stats.txt')
            with open(stats_file, 'a') as f:
                f.write(
                    f"Epoch {epoch + 1}: Center Distance={center_distance:.4f}, "
                    f"Avg Min Distance={avg_min_distance:.4f}\n"
                )

            print(
                f"UMAP plot saved for epoch {epoch + 1}. "
                f"Center distance: {center_distance:.4f}, "
                f"Avg min distance: {avg_min_distance:.4f}"
            )

        except Exception as e:
            print(f"Error creating UMAP plot for epoch {epoch + 1}: {str(e)}")

    model.train()


class DomainDiscriminator(nn.Module):
    """
    Domain discriminator: used to distinguish between simulation and real data
    """
    def __init__(self, input_dim, hidden_dim=None):
        super(DomainDiscriminator, self).__init__()
        if hidden_dim is None:
            hidden_dim = min(128, max(64, input_dim // 2))

        # NOTE: output logits (no Sigmoid); use BCEWithLogitsLoss for stability
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_p):
        ctx.lambda_p = lambda_p
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_p
        return output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_p=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_p = lambda_p

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_p)


def calculate_dynamic_ratio(data_quantity, real_data_quantity):
    min_ratio = 0.1
    ratio = min_ratio

    while ratio * data_quantity < real_data_quantity:
        ratio += 0.05
        if ratio >= 1.0:
            ratio = 0.9
            break
    return ratio


class CyclicRealDataSampler:
    def __init__(self, X_real, noise_std=0.001, max_repeat=10):
        self.X_real = X_real
        self.noise_std = noise_std
        self.max_repeat = max_repeat
        self.current_idx = 0
        self.n_real = len(X_real)

    def sample(self, n_samples):
        if n_samples <= self.n_real:
            indices = torch.randperm(self.n_real)[:n_samples]
            return self.X_real[indices]
        else:
            samples = []
            remaining = n_samples
            repeat_count = 0

            while remaining > 0 and repeat_count < self.max_repeat:
                current_batch_size = min(remaining, self.n_real)

                indices = torch.arange(self.n_real)
                if repeat_count > 0:
                    indices = indices[torch.randperm(self.n_real)]

                batch_data = self.X_real[indices[:current_batch_size]]

                if repeat_count > 0:
                    noise_level = self.noise_std * repeat_count
                    noise = torch.randn_like(batch_data) * noise_level
                    batch_data = torch.clamp(batch_data + noise, min=0)

                samples.append(batch_data)
                remaining -= current_batch_size
                repeat_count += 1

            return torch.cat(samples, dim=0)




class ImbalancedDomainDatasetSeededNoise(torch.utils.data.Dataset):
    """
    Variant of ImbalancedDomainDataset:
    - Mixes simulation and real data for domain-adversarial training.
    - If real data is insufficient, it repeats real samples to meet the desired real_ratio.
    - Noise is added with a fixed standard deviation (noise_std).
    - Instead of scaling noise by repeat_count, each repetition uses a different random seed
      to generate fresh noise.
    """
    def __init__(self, X_sim, y_sim, X_real, real_ratio=0.3, noise_std=0.01, base_seed=0):
        self.n_sim = len(X_sim)
        self.n_real = len(X_real)

        n_real_samples = int(self.n_sim * real_ratio / (1 - real_ratio))
        '''

        print("Creating dataset (seeded-noise variant):")
        print(f"  Simulation samples: {self.n_sim}")
        print(f"  Real data available: {self.n_real}")
        print(f"  Real data needed: {n_real_samples}")
        print(f"  Real data ratio: {real_ratio:.2%}")
        print(f"  base_seed: {base_seed}")
        '''

        self.sim_data = X_sim
        self.sim_targets = y_sim

        if n_real_samples <= self.n_real:
            indices = torch.randperm(self.n_real)[:n_real_samples]
            self.real_data = X_real[indices]
        else:
            '''
            print("  Real data insufficient, using cyclic sampling with seeded noise (fixed std)")
            '''
            samples = []
            remaining = n_real_samples
            repeat_count = 0

            # Use a dedicated CPU generator for reproducibility and to avoid affecting global RNG state (optional).
            # Note: randperm typically runs on CPU for index generation, which is the most compatible setup.
            g = torch.Generator(device="cpu")

            while remaining > 0:
                current_batch = min(remaining, self.n_real)

                # Use a different seed for each repetition.
                g.manual_seed(int(base_seed) + repeat_count)

                indices = torch.randperm(self.n_real, generator=g)[:current_batch]
                batch_data = X_real[indices]

                if repeat_count > 0:
                    # Add fixed-std noise; do NOT multiply by repeat_count.
                    # randn_like ensures noise is generated on the same device/dtype as batch_data.
                    noise = torch.randn_like(batch_data) * noise_std
                    batch_data = torch.clamp(batch_data + noise, min=0)

                samples.append(batch_data)
                remaining -= current_batch
                repeat_count += 1

            self.real_data = torch.cat(samples, dim=0)
            '''
            print(f"  Repeated real data {repeat_count} times")
            '''

        self.all_data = torch.cat([self.sim_data, self.real_data], dim=0)

        dummy_real_targets = torch.zeros(
            len(self.real_data), y_sim.size(1),
            device=y_sim.device, dtype=y_sim.dtype
        )
        self.all_targets = torch.cat([self.sim_targets, dummy_real_targets], dim=0)

        self.all_domain_labels = torch.cat([
            torch.zeros(len(self.sim_data), device=X_sim.device, dtype=torch.float32),
            torch.ones(len(self.real_data), device=X_sim.device, dtype=torch.float32)
        ], dim=0)
        '''
        print(f"  Final dataset size: {len(self.all_data)}")
        print(f"  Sim:Real ratio = {len(self.sim_data)}:{len(self.real_data)}")
        '''

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return (
            self.all_data[idx],
            self.all_targets[idx],
            self.all_domain_labels[idx]
        )
    
class ImbalancedDomainDataset(torch.utils.data.Dataset):
    """
    A dataset that mixes simulation and real data for domain-adversarial training.
    """
    def __init__(self, X_sim, y_sim, X_real, real_ratio=0.3, noise_std=0.01):
        self.n_sim = len(X_sim)
        self.n_real = len(X_real)

        n_real_samples = int(self.n_sim * real_ratio / (1 - real_ratio))
        '''
        print("Creating dataset:")
        print(f"  Simulation samples: {self.n_sim}")
        print(f"  Real data available: {self.n_real}")
        print(f"  Real data needed: {n_real_samples}")
        print(f"  Real data ratio: {real_ratio:.2%}")
        '''
        self.sim_data = X_sim
        self.sim_targets = y_sim

        if n_real_samples <= self.n_real:
            indices = torch.randperm(self.n_real)[:n_real_samples]
            self.real_data = X_real[indices]
        else:
            '''
            print("  Real data insufficient, using cyclic sampling with noise")
            '''
            samples = []
            remaining = n_real_samples
            repeat_count = 0

            while remaining > 0:
                current_batch = min(remaining, self.n_real)
                indices = torch.randperm(self.n_real)[:current_batch]
                batch_data = X_real[indices]

                if repeat_count > 0:
                    noise = torch.randn_like(batch_data) * noise_std * repeat_count
                    batch_data = torch.clamp(batch_data + noise, min=0)

                samples.append(batch_data)
                remaining -= current_batch
                repeat_count += 1

            self.real_data = torch.cat(samples, dim=0)
            '''
            print(f"  Repeated real data {repeat_count} times")
            '''

        self.all_data = torch.cat([self.sim_data, self.real_data], dim=0)

        dummy_real_targets = torch.zeros(len(self.real_data), y_sim.size(1), device=y_sim.device, dtype=y_sim.dtype)
        self.all_targets = torch.cat([self.sim_targets, dummy_real_targets], dim=0)

        self.all_domain_labels = torch.cat([
            torch.zeros(len(self.sim_data), device=X_sim.device, dtype=torch.float32),
            torch.ones(len(self.real_data), device=X_sim.device, dtype=torch.float32)
        ], dim=0)


    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return (
            self.all_data[idx],
            self.all_targets[idx],
            self.all_domain_labels[idx]
        )


class MultiGraphDeconv(nn.Module):
    def __init__(
        self,
        cell_types,
        in_c,
        hid_c,
        out_c,
        K,
        device,
        top_k,
        is_st=False,
        kernel_type='chebyshev',
        wavelet_type='mexican_hat',
        use_adversarial=False,
        grl_lambda=0.1
    ):
        super().__init__()
        self.cell_types = cell_types
        self.num_cell_types = len(cell_types)
        self.device = device
        self.use_adversarial = use_adversarial
        self.kernel_type = kernel_type

        if kernel_type == 'pca':
            self.pca = PCA(n_components=K)
            self.scaler = StandardScaler()
            self.pca_fitted = False

            self.head = nn.Sequential(
                nn.Linear(K, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, self.num_cell_types),
                nn.Softmax(dim=1)
            )

        elif kernel_type == 'linear':
            self.linear_net = nn.Sequential(
                nn.Linear(in_c, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, self.num_cell_types),
                nn.Softmax(dim=1)
            )
            '''
            self.linear_net = nn.Sequential(
                nn.Linear(in_c, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, self.num_cell_types),
                nn.Softmax(dim=1)
            )
            '''

        else:
            graphnet_cls = GraphNet_st if is_st else GraphNet_bulk
            self.graph_nets = nn.ModuleList([
                graphnet_cls(
                    in_c, hid_c, out_c, K, device, top_k=top_k,
                    kernel_type=kernel_type, wavelet_type=wavelet_type
                )
                for _ in cell_types
            ])

            self.embed_norm = nn.LayerNorm(64)
            self.concat_norm = nn.BatchNorm1d(self.num_cell_types * 64)

            self.head = nn.Sequential(
                nn.Linear(self.num_cell_types * 64, 128),
                nn.LayerNorm(128),
                nn.ELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ELU(),
                nn.Dropout(0.1),
                nn.Linear(64, self.num_cell_types),
                nn.Softmax(dim=1)
            )

            if self.use_adversarial:
                self.grl = GradientReversalLayer(lambda_p=grl_lambda)
                self.domain_discriminator = DomainDiscriminator(self.num_cell_types * 64)

        self._bce_logits = nn.BCEWithLogitsLoss()

    def fit_pca(self, X):
        if self.kernel_type != 'pca':
            return
        '''
        print(f"Fitting PCA with {X.shape[0]} samples and {X.shape[1]} features...")
        '''
        X_np = X.cpu().numpy() if torch.is_tensor(X) else X
        X_scaled = self.scaler.fit_transform(X_np)
        self.pca.fit(X_scaled)
        self.pca_fitted = True

        explained_ratio = np.sum(self.pca.explained_variance_ratio_)
        '''
        print(f"PCA components: {self.pca.n_components}")
        print(f"Explained variance ratio: {explained_ratio:.4f}")
        '''
        return explained_ratio

    def transform_pca(self, X):
        if self.kernel_type != 'pca' or not self.pca_fitted:
            return X

        X_np = X.cpu().numpy() if torch.is_tensor(X) else X
        X_scaled = self.scaler.transform(X_np)
        X_pca = self.pca.transform(X_scaled)
        return torch.from_numpy(X_pca).float().to(self.device)

    def forward(self, graph_list, data, domain_labels=None):
        if self.kernel_type == 'pca':
            if self.pca_fitted:
                data = self.transform_pca(data)
            return self.head(data)

        elif self.kernel_type == 'linear':
            return self.linear_net(data)

        else:
            embeddings = []
            for i, graph_net in enumerate(self.graph_nets):
                emb = graph_net(graph_list[i], data)  # [B, 64]
                emb = self.embed_norm(emb)
                embeddings.append(emb)

            x = torch.cat(embeddings, dim=1)  # [B, num_cell_types*64]
            x = self.concat_norm(x)
            output = self.head(x)
            return output

    def get_embeddings(self, graph_list, data):
        if self.kernel_type == 'pca':
            return self.transform_pca(data) if self.pca_fitted else data

        elif self.kernel_type == 'linear':
            with torch.no_grad():
                x = self.linear_net[:3](data)
            return x

        else:
            embeddings = []
            for i, graph_net in enumerate(self.graph_nets):
                emb = graph_net(graph_list[i], data)
                emb = self.embed_norm(emb)
                embeddings.append(emb)
            x = torch.cat(embeddings, dim=1)
            x = self.concat_norm(x)
            return x

    def forward_adversarial(self, graph_list, data, domain_labels, detach_features=False):
        """
        Returns domain loss.
        - detach_features=True: train domain discriminator only (no gradient to feature extractor)
        - detach_features=False: train feature extractor adversarially (GRL), discriminator params should be frozen externally
        """
        if self.kernel_type in ['pca', 'linear'] or (not self.use_adversarial):
            return torch.tensor(0.0, device=data.device)

        feats = self.get_embeddings(graph_list, data)  # [B, num_cell_types*64]
        if detach_features:
            feats = feats.detach()

        # If not detaching, use GRL so gradient to feats is reversed
        if not detach_features:
            feats = self.grl(feats)

        logits = self.domain_discriminator(feats).squeeze(1)  # [B]
        domain_labels = domain_labels.float().view(-1).to(logits.device)
        loss = self._bce_logits(logits, domain_labels)
        return loss


class LinearModel(torch.nn.Module):
    def __init__(self, t):
        super(LinearModel, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(t, 64),
            nn.BatchNorm1d(64),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, t):
        f = self.encoder(t)
        return f


class GraphConv(nn.Module):
    def __init__(self, in_c, out_c, K, device, bias=True, normalize=True, top_k=None, kernel_type='chebyshev',
                 wavelet_type='mexican_hat'):
        super(GraphConv, self).__init__()
        self.device = device
        self.normalize = normalize
        self.top_k = top_k
        self.K = K + 1
        self.kernel_type = kernel_type
        self.wavelet_type = wavelet_type

        self.weight = nn.Parameter(torch.Tensor(self.K, 1, in_c, out_c))
        nn.init.orthogonal_(self.weight, gain=nn.init.calculate_gain('leaky_relu', 0.4))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            nn.init.orthogonal_(self.bias, gain=nn.init.calculate_gain('leaky_relu', 0.4))
        else:
            self.register_parameter("bias", None)

    def forward(self, inputs, graph, mode=Const.CHEB_MODE):
        L = GraphConv.get_laplacian(graph, self.normalize)
        L_np = L.detach().cpu().numpy()  # keep your dtype/path (do not change math)

        # ---- eig cache (disk) ----
        graph_id = getattr(self, "graph_id", None)
        cache_dir = getattr(self, "cache_dir", None)
        cache_mode = getattr(self, "cache_mode", "off")  # "off" | "write" | "read"

        use_cache = (cache_mode != "off") and (graph_id is not None) and (cache_dir is not None)
        cache_path = None
        cache = None

        # debug counters (optional)
        if not hasattr(self, "_cache_hits"):
            self._cache_hits = 0
            self._cache_misses = 0

        if use_cache:
            cache_path = os.path.join(
                cache_dir,
                f"eigh_cache.pt"
            )
            if os.path.exists(cache_path):
                cache = torch.load(cache_path, map_location="cpu")
            else:
                cache = {}

        key = int(graph_id) if graph_id is not None else None

        if use_cache and (key in cache):
            pack = cache[key]
            lam_topk = pack["lam"].to(self.device)
            u_topk = pack["u"].to(self.device)
            self._cache_hits += 1
        else:
            self._cache_misses += 1

            # READ mode: do not allow compute
            if use_cache and cache_mode == "read":
                raise RuntimeError(
                    f"[EIG-CACHE] MISS in READ mode: graph_id={key}, cache_path={cache_path}"
                )

            # compute
            lam, u = np.linalg.eigh(L_np)
            idx = np.argsort(lam)[::-1][:self.top_k]
            lam = lam[idx]
            u = u[:, idx]

            lam_topk = torch.from_numpy(lam).float().to(self.device)
            u_topk = torch.from_numpy(u).float().to(self.device)

            # write only in WRITE mode
            if use_cache and cache_mode == "write":
                cache[key] = {
                    "idx": torch.from_numpy(idx.astype(np.int64)),
                    "lam": lam_topk.detach().cpu(),
                    "u": u_topk.detach().cpu(),
                }
                try:
                    torch.save(cache, cache_path)
                except Exception as e:
                    print(f"[WARN] failed to save eigh cache to {cache_path}: {e}")

        lam = torch.diag(lam_topk)
        u = u_topk
        if self.kernel_type == 'chebyshev':
            if torch.max(lam) != torch.min(lam):
                lam = 2 * ((lam - torch.min(lam)) / (torch.max(lam) - torch.min(lam))) - torch.eye(lam.size(0)).to(
                    self.device)
            else:
                lam = torch.zeros_like(lam)
            mul_L = self.cheb_polynomial(lam).unsqueeze(1)

        elif self.kernel_type == 'wavelet':
            if torch.max(lam) != torch.min(lam):
                lam_normalized = 2 * (lam - torch.min(lam)) / (torch.max(lam) - torch.min(lam))
            else:
                lam_normalized = torch.zeros_like(lam)
            mul_L = self.wavelet_kernel(lam_normalized).unsqueeze(1)

        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

        if mode == 0:
            result = torch.matmul(inputs, mul_L)
        elif mode == 1:
            result = torch.matmul(inputs, mul_L)
            result = torch.matmul(result, u.t())
        elif mode == 2:
            result = torch.matmul(u, mul_L)
            result = torch.matmul(result, u.t())
            result = torch.matmul(inputs, result)
        else:
            raise ValueError("Unknown mode for GraphConv")

        result = torch.matmul(result, self.weight)
        result = torch.sum(result, dim=0) + self.bias

        temp = []
        for i in range(result.size()[0]):
            if torch.min(result[i]) == torch.max(result[i]):
                temp.append(result[i])
            else:
                temp.append(2 * ((result[i] - torch.min(result[i])) / (torch.max(result[i]) - torch.min(result[i]))) - 1)
        temp = torch.stack(temp)

        return temp

    def cheb_polynomial(self, laplacian):
        N = laplacian.size(0)
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k - 1]) - \
                                               multi_order_laplacian[k - 2]

        return multi_order_laplacian

    def wavelet_kernel(self, laplacian):
        N = laplacian.size(0)
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)

        eigenvalues = torch.diag(laplacian)

        for k in range(self.K):
            scale = 2.0 ** k
            scaled_eigenvalues = scale * eigenvalues

            if self.wavelet_type == 'mexican_hat':
                wavelet_values = (1 - scaled_eigenvalues ** 2) * torch.exp(-scaled_eigenvalues ** 2 / 2)
            elif self.wavelet_type == 'morlet':
                wavelet_values = torch.cos(5 * scaled_eigenvalues) * torch.exp(-scaled_eigenvalues ** 2 / 2)
            elif self.wavelet_type == 'dog':
                wavelet_values = -scaled_eigenvalues * torch.exp(-scaled_eigenvalues ** 2 / 2)
            elif self.wavelet_type == 'paul':
                wavelet_values = torch.pow(1 + scaled_eigenvalues ** 2, -2.5) * torch.cos(scaled_eigenvalues)
            elif self.wavelet_type == 'shannon':
                eps = 1e-8
                sinc_vals = torch.sin(torch.pi * scaled_eigenvalues / 2) / (torch.pi * scaled_eigenvalues / 2 + eps)
                wavelet_values = sinc_vals * torch.cos(3 * torch.pi * scaled_eigenvalues / 2)
            elif self.wavelet_type == 'gaussian':
                wavelet_values = -scaled_eigenvalues * torch.exp(-scaled_eigenvalues ** 2 / 2) / torch.sqrt(
                    2 * torch.tensor(torch.pi, device=laplacian.device))
            else:
                raise ValueError(f"Unsupported wavelet type: {self.wavelet_type}")

            multi_order_laplacian[k] = torch.diag(wavelet_values)

        return multi_order_laplacian

    @staticmethod
    def get_laplacian(graph, normalize):
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L


class GraphNet_bulk(nn.Module):
    def __init__(self, in_c, hid_c, out_c, K, device, top_k=None, kernel_type='chebyshev',
                 wavelet_type='mexican_hat', use_adversarial=False):
        super(GraphNet_bulk, self).__init__()

        self.reduce1 = nn.Linear(in_c, 128)
        self.norm1 = nn.BatchNorm1d(128)
        self.reduce2 = nn.Linear(128, top_k)
        self.norm2 = nn.BatchNorm1d(top_k)
        self.act = nn.ELU()

        self.conv1 = GraphConv(in_c=in_c, out_c=out_c, K=K, device=device, top_k=top_k,
                               kernel_type=kernel_type, wavelet_type=wavelet_type)

        self.post_reduce1 = nn.Linear(in_c, 128)
        self.post_norm1 = nn.BatchNorm1d(128)
        self.post_reduce2 = nn.Linear(128, 64)

    def forward(self, graph, data):
        x = self.reduce1(data)
        x = self.act(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.reduce2(x)

        B, N = x.size(0), x.size(1)
        x = x.view(B, 1, N)
        x = self.act(self.conv1(x, graph))
        x = x.view(B, -1)

        x = self.post_reduce1(x)
        x = self.post_norm1(x)
        x = self.act(x)
        x = self.post_reduce2(x)

        return x.squeeze(1)


class GraphNet_st(nn.Module):
    def __init__(self, in_c, hid_c, out_c, K, device, top_k=None, kernel_type='chebyshev',
                 wavelet_type='mexican_hat', use_adversarial=False):
        super(GraphNet_st, self).__init__()

        self.reduce1 = nn.Linear(in_c, 128)
        self.norm1 = nn.BatchNorm1d(128)
        self.reduce2 = nn.Linear(128, top_k)
        self.norm2 = nn.BatchNorm1d(top_k)
        self.act = nn.ELU()

        self.conv1 = GraphConv(in_c=in_c, out_c=out_c, K=K, device=device, top_k=top_k,
                               kernel_type=kernel_type, wavelet_type=wavelet_type)

        self.post_reduce1 = nn.Linear(in_c, 128)
        self.post_norm1 = nn.BatchNorm1d(128)
        self.post_reduce2 = nn.Linear(128, 64)

    def forward(self, graph, data):
        x = self.reduce1(data)
        x = self.act(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.reduce2(x)
        x = self.norm2(x)

        B, N = x.size(0), x.size(1)
        x = x.view(B, 1, N)
        x = self.act(self.conv1(x, graph))
        x = x.view(B, -1)

        x = self.post_reduce1(x)
        x = self.post_norm1(x)
        x = self.act(x)
        x = self.post_reduce2(x)

        return x.squeeze(1)


def change_lr(optim, new_lr):
    for g in optim.param_groups:
        g['lr'] = new_lr


class InferDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)


def get_G(cell_name, sc_adata, annotation_key, knn_k=None, symmetrize=True, keep_self=True):
    def _get_mat_YW(sc_df):
        mat_Y = torch.from_numpy(sc_df.values).float()
        mat_W = mat_Y @ mat_Y.t()

        # kNN sparsify (optional)
        if knn_k is not None and knn_k > 0 and knn_k < mat_W.size(1):
            W = mat_W.clone()
            if not keep_self:
                W.fill_diagonal_(-float('inf'))

            _, topk_idx = torch.topk(W, k=knn_k, dim=1, largest=True, sorted=False)

            mask = torch.zeros_like(W, dtype=torch.bool)
            row_idx = torch.arange(W.size(0)).unsqueeze(1).expand_as(topk_idx)
            mask[row_idx, topk_idx] = True

            W_sparse = torch.where(mask, mat_W, torch.zeros_like(mat_W))
            if symmetrize:
                W_sparse = torch.maximum(W_sparse, W_sparse.t())
            mat_W = W_sparse

        return mat_W

    sec_num = 1e-6
    sub_adata = sc_adata[sc_adata.obs[annotation_key] == cell_name, :].copy()
    sub_df = get.count_data(sub_adata)
    mat_W = _get_mat_YW(sub_df)
    num = mat_W.size(0)

    mat_G = mat_W + sec_num * torch.eye(num, dtype=mat_W.dtype, device=mat_W.device) \
            + sec_num * torch.ones(num, num, dtype=mat_W.dtype, device=mat_W.device)
    return mat_G, num


def select_gene(expression: pd.DataFrame, sel_gene: list):
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    ret_exp = pd.DataFrame(expression.iloc[:, 0])
    for gene in sel_gene:
        ret_exp[gene] = expression[gene] if gene in expression.columns else 0
    return ret_exp.iloc[:, 1:]


class GraphDeconv:
    def __init__(
            self,
            cell_num=200,
            mode=Const.MODE_PREDICTION,
            use_gpu=True,
            top_k=None,
            kernel_type='chebyshev',
            wavelet_type='gaussian',
            use_adversarial=False,
            loss_type='hybrid'
    ):
        self.cell_num = cell_num
        self.mode = mode
        self.device = configure_device(use_gpu)
        self.top_k = top_k
        self.kernel_type = kernel_type
        self.wavelet_type = wavelet_type
        self.use_adversarial = use_adversarial
        self.loss_type = loss_type

    def fit(
        self,
        expression,
        cell_list=None,
        sc_adata=None,
        annotation_key=None,
        model_folder=None,
        out_dir='./',
        project='',
        file_dir='./',
        save=True,
        is_st=False,
        top_k=50,
        return_embedding=False,
        kernel_type=None,
        wavelet_type=None,
        epoch_suffix=None
    ):
        if kernel_type is None:
            kernel_type = self.kernel_type
        if wavelet_type is None:
            wavelet_type = self.wavelet_type

        utils.check_paths(output_folder=out_dir)
        device = self.device
        tot_cell_list = cell_list

        input_bulk = pd.read_csv(
            f"{file_dir}/batch_effect/{project}_batch_effected.txt",
            sep='\t',
            index_col=0
        )[expression.obs_names.tolist()]

        test_data = input_bulk.T.values
        X = torch.from_numpy(test_data).float().to(device)
        gene_num = X.shape[1]


        if kernel_type in ['pca', 'linear']:
            print(f"Loading {kernel_type} model...")
            model_path = f"{model_folder}/multi_graph_model.pt"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file {model_path} not found")

            model_state = torch.load(model_path, map_location=device)
            has_adversarial = False

            model = MultiGraphDeconv(
                tot_cell_list, gene_num, gene_num, gene_num, K=2, device=device,
                top_k=top_k if top_k is not None else self.top_k,
                is_st=is_st,
                kernel_type=kernel_type,
                wavelet_type=wavelet_type,
                use_adversarial=has_adversarial
            ).to(device)

            if kernel_type == 'pca':
                pca_data_path = f"{model_folder}/pca_components.pkl"
                if os.path.exists(pca_data_path):
                    with open(pca_data_path, 'rb') as f:
                        pca_data = pickle.load(f)
                        model.pca = pca_data['pca']
                        model.scaler = pca_data['scaler']
                        model.pca_fitted = True
                    print("PCA components loaded successfully")
                else:
                    print("Warning: PCA components file not found, model may not work correctly")

            model.load_state_dict(model_state, strict=True)
            graph_list = None

        else:
            print("Loading saved graphs and model...")
            with open(f"{model_folder}/multi_graph_graphs.pkl", "rb") as f:
                graph_list = pickle.load(f)
            graph_list = [graph.to(device) for graph in graph_list]

            model_state = torch.load(f"{model_folder}/multi_graph_model.pt", map_location=device)

            has_adversarial = any('domain_discriminator' in key for key in model_state.keys())
            model = MultiGraphDeconv(
                tot_cell_list, gene_num, gene_num, gene_num, K=2, device=device,
                top_k=top_k if top_k is not None else self.top_k,
                is_st=is_st,
                kernel_type=kernel_type,
                wavelet_type=wavelet_type,
                use_adversarial=has_adversarial
            ).to(device)

            # strict=False: allow loading older checkpoints without adversarial components if needed
            model.load_state_dict(model_state, strict=False)

        print("Model loaded successfully")
        for i, gn in enumerate(model.graph_nets):
            gn.conv1.graph_id = i
            gn.conv1.cache_dir = model_folder  # where to save/load cache
            gn.conv1.cache_mode = "read"

        model.eval()
        all_outputs = []
        all_embeddings = [] if return_embedding else None

        dataset = InferDataset(X)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=Const.BATCH_SIZE,
            shuffle=False,
            drop_last=False
        )

        print(f"Processing {len(X)} samples in {len(dataloader)} batches...")

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Inference")):
                if return_embedding:
                    batch_embedding = model.get_embeddings(graph_list, batch_data)
                    all_embeddings.append(batch_embedding.cpu())
                else:
                    batch_output = model(graph_list, batch_data)
                    all_outputs.append(batch_output.cpu())

                if batch_idx == 0:
                    print(f"First batch shape: {batch_data.shape}")
                    if not return_embedding:
                        print(f"First batch output shape: {batch_output.shape}")

        if return_embedding:
            final_embedding = torch.cat(all_embeddings, dim=0)
            print(f"Final embedding shape: {final_embedding.shape}")
            return final_embedding
        else:
            final_output = torch.cat(all_outputs, dim=0)

            unique_outputs = torch.unique(final_output, dim=0)

            output_np = final_output.numpy()
            pred_df = pd.DataFrame(output_np, index=expression.obs_names, columns=tot_cell_list)

            if save:
                if epoch_suffix is not None:
                    output_filename = f"{out_dir}/{project}_prediction_frac_epoch{epoch_suffix}.csv"
                else:
                    output_filename = f"{out_dir}/{project}_prediction_frac.csv"
                pred_df.to_csv(output_filename)
                print(f"Prediction saved to {output_filename}")
            return pred_df

    def train(
      self,
      presudo_bulk=None,
      bulk_adata=None,
      cell_list=None,
      sc_adata=None,
      annotation_key=None,
      batch_size=Const.BATCH_SIZE,
      out_dir="./",
      project_name="",
      data_num=10000,
      batch_effect=True,
      is_st=False,
      top_k=None,
      kernel_type=None,
      wavelet_type=None,
      use_adversarial=None,
      real_data_ratio=None,
      adversarial_lr=0.001,
      adversarial_update_freq=1,
      loss_type=None,
      kl_weight=1,
      mse_weight=0,
      predict_per_epoch=False,
      prediction_file_dir=None,
      prediction_project=None,
      domain_loss_weight=0.05,
      grl_lambda=0.1,
      val_domain_max_samples=2000,   # NEW: max samples per-domain used in val_domain_loss
  ):
      # ---------------- Basic configs & preparation ----------------
      if kernel_type is None:
          kernel_type = self.kernel_type
      if wavelet_type is None:
          wavelet_type = self.wavelet_type
      if use_adversarial is None:
          use_adversarial = self.use_adversarial
      if loss_type is None:
          loss_type = self.loss_type
      model_path = os.path.join(out_dir, "multi_graph_model.pt")
      # Also check the batch-effect output file
      batch_dir = os.path.join(out_dir, "batch_effect")
      batch_file = os.path.join(batch_dir, f"{project_name}_batch_effected.txt")
      model_exists = os.path.exists(model_path)
      batch_file_exists = os.path.exists(batch_file)
      utils.check_paths(output_folder=out_dir)
      tot_cell_list = cell_list
      presudo_bulk_full = presudo_bulk.copy()
      bulk_adata_full = bulk_adata.copy()
      # If the model exists but the batch-effect file is missing, run batch-effect removal first, then return
      if model_exists and batch_effect and (not batch_file_exists):
            presudo_bulk_full, bulk_adata_full = preprocessing.remove_batch_effect(
                presudo_bulk_full,
                bulk_adata_full,
                out_dir=out_dir,
                project=project_name,
                batch_effect=batch_effect
            )
            print(
                f"Model {model_path} already exists; batch-effect file {batch_file} is missing. "
                "Regenerated batch-effect outputs, then skipping training."
            )
            #loc='batch_effected'
            #sample_list = utils.filter_samples(
                #presudo_bulk_full, bulk_adata_full, data_num=data_num, loc=loc, cell_type_num=len(tot_cell_list)
            #)
            #presudo_bulk_train = presudo_bulk_full[sample_list, :]
            #fraction = get.meta(presudo_bulk_train, position_key="obs")
            #fraction.to_csv(f"{out_dir}/fraction.csv")
            return
      if model_exists and (not batch_effect or batch_file_exists):
            print(f"Model {model_path} already exists, skipping training.")
            return

        


  
      # Default: chebyshev, top_k default is provided by args / self.top_k
      if kernel_type in ['pca', 'linear'] and use_adversarial:
          use_adversarial = False
  
      kernel_info = f"{kernel_type}" + (f"({wavelet_type})" if kernel_type == 'wavelet' else "")
      adversarial_info = " with Adversarial Learning" if use_adversarial else ""
      loss_info = f" using {loss_type} loss (KL:{kl_weight}, MSE:{mse_weight})" if loss_type == 'hybrid' else f" using {loss_type} loss"
  

  
      presudo_bulk_full, bulk_adata_full = preprocessing.remove_batch_effect(
          presudo_bulk_full, bulk_adata_full, out_dir=out_dir, project=project_name, batch_effect=batch_effect
      )
  
      if batch_effect:
          loc = "batch_effected"
          #plots.batch_effect(bulk_adata_full, presudo_bulk_full, out_dir=out_dir + "/plot", title=project_name)
      else:
          loc = None
  
      # Select a subset of pseudo bulk as simulation data for training
      sample_list = utils.filter_samples(
          presudo_bulk_full, bulk_adata_full, data_num=data_num, loc=loc, cell_type_num=len(tot_cell_list)
      )
      presudo_bulk_train = presudo_bulk_full[sample_list, :]
    
  
      # ---------------- Prepare simulation data ----------------
      expression = get.count_data_t(presudo_bulk_train, counts_location=loc)
      fraction = get.meta(presudo_bulk_train, position_key="obs")
      #fraction.to_csv(f"{out_dir}/fraction.csv")
  
      X_sim_full = torch.from_numpy(expression.values).float().to(self.device)
      y_sim_full = torch.from_numpy(fraction[tot_cell_list].values).float().to(self.device)
  
      if predict_per_epoch:
          print("Will perform prediction after each epoch")
          if prediction_file_dir is None:
              prediction_file_dir = out_dir
          if prediction_project is None:
              prediction_project = project_name
  
      # ---------------- Prepare real data (once) ----------------
      X_real_full = None
      if use_adversarial:
          real_st_expression = get.count_data_t(bulk_adata_full, counts_location=loc)
          X_real_full = torch.from_numpy(real_st_expression.values).float().to(self.device)
          if real_data_ratio is None:
              real_data_ratio = calculate_dynamic_ratio(len(X_sim_full), len(X_real_full))
  
      # ---------------- Fixed train/valid split on pure simulation pool ----------------
      base_dataset = torch.utils.data.TensorDataset(X_sim_full, y_sim_full)
  
      split_repeat = Const.MAX_SPLIT if hasattr(Const, "MAX_SPLIT") else 10
      full_size = len(base_dataset)
      train_size = int(0.85 * full_size)
      valid_size = full_size - train_size
  
      best_score = -np.inf
      best_train_indices = None
      best_valid_indices = None
  
      for _ in range(split_repeat):
          perm = torch.randperm(full_size)
          cur_train_idx = perm[:train_size]
          cur_valid_idx = perm[train_size:]
  
          cur_valid_subset = torch.utils.data.Subset(base_dataset, cur_valid_idx)
          # you currently use pearson-based split selection
          cur_score = utils.compute_average_pearson(cur_valid_subset, bulk_adata_full, loc=loc)
  
          if cur_score > best_score:
              best_score = cur_score
              best_train_indices = cur_train_idx
              best_valid_indices = cur_valid_idx
  
  
      sim_train_dataset = torch.utils.data.Subset(base_dataset, best_train_indices)
      sim_valid_dataset = torch.utils.data.Subset(base_dataset, best_valid_indices)

      y_valid = y_sim_full[best_valid_indices].cpu().numpy()
      y_valid_df = pd.DataFrame(
            y_valid,
            columns=fraction.columns  
        )
      y_valid_df.to_csv(f"{out_dir}/validation_y.csv", index=False)
  
      valid_loader = create_safe_dataloader(sim_valid_dataset, batch_size, shuffle=False, min_last_batch=10)
  
      # ---------------- Graph structure & model ----------------
      gene_num = X_sim_full.shape[1]
  
      if kernel_type in ['pca', 'linear']:
          graph_list = None
      else:
          graph_list = []
          for cell in tot_cell_list:
              mat_G, _ = get_G(cell, sc_adata, annotation_key)
              graph_list.append(mat_G.to(self.device))
  
      model = MultiGraphDeconv(
          tot_cell_list, gene_num, gene_num, gene_num, K=2, device=self.device,
          top_k=top_k if top_k is not None else self.top_k,
          is_st=is_st,
          kernel_type=kernel_type,
          wavelet_type=wavelet_type,
          use_adversarial=use_adversarial,
          grl_lambda=grl_lambda
      ).to(self.device)
      if kernel_type not in ['pca', 'linear']:
        for i, gn in enumerate(model.graph_nets):
            gn.conv1.graph_id = i
            gn.conv1.cache_dir = out_dir
  
      if kernel_type == 'pca':
          model.fit_pca(X_sim_full)
  
      # ---------------- Optimizers ----------------
      main_params = [p for n, p in model.named_parameters() if not n.startswith("domain_discriminator.")]
      main_optimizer = torch.optim.Adam(main_params, lr=Const.LEARNING_RATE, weight_decay=1e-6)
  
      main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
          main_optimizer, mode='min', factor=0.7, patience=3, verbose=True, min_lr=1e-5
      )
  
      if use_adversarial:
          d_params = list(model.domain_discriminator.parameters())
          adversarial_optimizer = torch.optim.Adam(d_params, lr=adversarial_lr)
          adversarial_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
              adversarial_optimizer, mode='min', factor=0.7, patience=3, verbose=True, min_lr=1e-5
          )
      else:
          adversarial_optimizer = None
          adversarial_scheduler = None
  
      # ---------------- Early stopping variables ----------------
      best_val_loss = float('inf')
      best_state = None
      patience = 8
      wait = 0
  
      previous_val_loss = None
      consecutive_increases = 0
      overfitting_threshold = 3
      overfitting_locked = False
      locked_epoch = None
  
      low_point_val = float('inf')
      post_lowpoint_worse_count = 0
      lock_after_worse_epochs = 5
  
      if kernel_type not in ['pca', 'linear']:
          freeze_graph = [False] * len(tot_cell_list)
      else:
          freeze_graph = None
  
      train_losses = []
      val_losses = []
      val_mses = []
      val_cosines = []
      val_domain_losses = []   # NEW
  
      # helper: build val domain batch each epoch (small and balanced)
      def _compute_val_domain_loss():
          if (not use_adversarial) or (X_real_full is None) or (kernel_type in ['pca', 'linear']):
              return None
  
          model.eval()
          with torch.no_grad():
              n_sim_val = len(sim_valid_dataset)
              n_real = len(X_real_full)
              if n_sim_val == 0 or n_real == 0:
                  return None
  
              n = min(n_sim_val, n_real, int(val_domain_max_samples))
              if n <= 0:
                  return None
  
              # sample sim valid indices
              sim_idx = torch.randperm(n_sim_val)[:n]
              # gather sim data from subset
              sim_x_list = []
              for j in sim_idx.tolist():
                  xj, _ = sim_valid_dataset[j]
                  sim_x_list.append(xj.unsqueeze(0))
              X_sim_val = torch.cat(sim_x_list, dim=0).to(self.device)
  
              # sample real indices directly
              real_idx = torch.randperm(n_real)[:n]
              X_real_val = X_real_full[real_idx].to(self.device)
  
              X_mix = torch.cat([X_sim_val, X_real_val], dim=0)
              y_domain = torch.cat([
                  torch.zeros(n, device=self.device),
                  torch.ones(n, device=self.device)
              ], dim=0)
  
              # detach_features=True => evaluate D's BCE on current features
              vdl = model.forward_adversarial(graph_list, X_mix, y_domain, detach_features=True)
              return float(vdl.item())
  
      # ---------------- Training loop ----------------
      for epoch in range(Const.EPOCH_NUM_BULK):
          if kernel_type not in ['pca', 'linear']:
             mode = "write" if epoch == 0 else "read"
          for gn in model.graph_nets:
             gn.conv1.cache_mode = mode
          '''
          if epoch == 0:
             print("[EIG-CACHE] epoch 1: WRITE (compute if missing, then save)")
          else:
             print(f"[EIG-CACHE] epoch {epoch+1}: READ-ONLY (cache miss will raise)")
          '''
  
          # construct train_loader
          if use_adversarial:
              # collect sim train tensors
              X_sim_train = []
              y_sim_train = []
              for i in range(len(sim_train_dataset)):
                  xi, yi = sim_train_dataset[i]
                  X_sim_train.append(xi.unsqueeze(0))
                  y_sim_train.append(yi.unsqueeze(0))
              X_sim_train = torch.cat(X_sim_train, dim=0)
              y_sim_train = torch.cat(y_sim_train, dim=0)

              n_sim = len(X_sim_train)
              n_real = len(X_real_full)
              real_sim_ratio = (n_real / max(n_sim, 1))
              if real_sim_ratio < 0.05:
                    epoch_train_dataset = ImbalancedDomainDatasetSeededNoise(
                        X_sim=X_sim_train,
                        y_sim=y_sim_train,
                        X_real=X_real_full,
                        real_ratio=real_data_ratio,
                        noise_std=0.005,
                        base_seed=12345,
                    )
              else:
                    epoch_train_dataset = ImbalancedDomainDataset(
                        X_sim=X_sim_train,
                        y_sim=y_sim_train,
                        X_real=X_real_full,
                        real_ratio=real_data_ratio,
                        noise_std=0.005
                    )
              train_loader = create_safe_dataloader(epoch_train_dataset, batch_size, shuffle=True, min_last_batch=10)
          else:
              train_loader = create_safe_dataloader(sim_train_dataset, batch_size, shuffle=True, min_last_batch=10)
  
          model.train()
          epoch_loss = 0.0
          epoch_domain_loss_D = 0.0
          epoch_domain_loss_F = 0.0
          total_samples = 0
          total_sim_samples = 0
          total_real_samples = 0
  
          for batch_idx, batch_data in enumerate(train_loader):
  
              if use_adversarial:
                  data, target, domain_labels = batch_data
                  sim_mask = (domain_labels == 0)
                  real_mask = (domain_labels == 1)
  
                  sim_count = int(sim_mask.sum().item())
                  real_count = int(real_mask.sum().item())
                  total_sim_samples += sim_count
                  total_real_samples += real_count
  
                  # -------- (1) D-step --------
                  if adversarial_optimizer is not None and batch_idx % adversarial_update_freq == 0:
                      for p in model.domain_discriminator.parameters():
                          p.requires_grad = True
  
                      adversarial_optimizer.zero_grad()
                      d_loss = model.forward_adversarial(graph_list, data, domain_labels, detach_features=True)
                      d_loss.backward()
                      torch.nn.utils.clip_grad_norm_(model.domain_discriminator.parameters(), max_norm=1.0)
                      adversarial_optimizer.step()
                      epoch_domain_loss_D += d_loss.item() * data.size(0)
  
                  # -------- (2) Main + F-step --------
                  for p in model.domain_discriminator.parameters():
                      p.requires_grad = False
  
                  main_optimizer.zero_grad()
                  total_batch_loss = 0.0
  
                  if sim_count > 0:
                      sim_data = data[sim_mask]
                      sim_target = target[sim_mask]
                      output = model(graph_list, sim_data)
                      main_loss = combined_loss(output, sim_target, model, loss_type, kl_weight, mse_weight)
                      total_batch_loss = total_batch_loss + main_loss
                      epoch_loss += main_loss.item() * sim_data.size(0)
  
                  f_loss = model.forward_adversarial(graph_list, data, domain_labels, detach_features=False)
                  total_batch_loss = total_batch_loss + domain_loss_weight * f_loss
                  epoch_domain_loss_F += f_loss.item() * data.size(0)
  
                  total_batch_loss.backward()
                  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                  main_optimizer.step()
  
                  for p in model.domain_discriminator.parameters():
                      p.requires_grad = True
  
              else:
                  data, target = batch_data
                  batch_size_current = data.size(0)
                  total_samples += batch_size_current
  
                  main_optimizer.zero_grad()
                  output = model(graph_list, data)
                  main_loss = combined_loss(output, target, model, loss_type, kl_weight, mse_weight)
                  main_loss.backward()
                  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                  main_optimizer.step()
  
                  epoch_loss += main_loss.item() * batch_size_current
  
          # -------- epoch averages --------
          if use_adversarial:
              if total_sim_samples > 0:
                  epoch_loss /= total_sim_samples
              denom = max(len(train_loader.dataset), 1)
              epoch_domain_loss_D /= denom
              epoch_domain_loss_F /= denom
          else:
              if total_samples > 0:
                  epoch_loss /= total_samples
  
          train_losses.append(epoch_loss)
          '''
          if use_adversarial:
              print(
                  f"Epoch {epoch + 1} data distribution: sim={total_sim_samples}, real={total_real_samples}, "
                  f"ratio={total_sim_samples / max(total_real_samples, 1):.2f}:1"
              )
          '''
          '''
          # optional UMAP
          if plot_umap and use_adversarial and kernel_type not in ['pca', 'linear'] and epoch % 2 == 0 and X_real_full is not None:
              try:
                  plot_umap_embeddings(model, graph_list, X_sim_full, X_real_full, epoch, out_dir, n_samples=1000)
              except Exception as e:
                  print(f"Error generating UMAP plot for epoch {epoch + 1}: {str(e)}")
          '''
  
          # ---------------- Validation (main task) ----------------
          model.eval()
          all_outputs = []
          all_targets = []
  
          with torch.no_grad():
              for batch_data in valid_loader:
                  data, target = batch_data
                  output = model(graph_list, data)
                  all_outputs.append(output.cpu())
                  all_targets.append(target.cpu())
  
          all_outputs = torch.cat(all_outputs, dim=0)
          all_targets = torch.cat(all_targets, dim=0)
  
          val_loss = combined_loss(all_outputs, all_targets, model, loss_type, kl_weight, mse_weight).item()
          val_mse = F.mse_loss(all_outputs, all_targets).item()
          val_mae = F.l1_loss(all_outputs, all_targets).item()
  
          val_losses.append(val_loss)
          val_mses.append(val_mse)
  
          # ---------------- Validation (domain) NEW ----------------
          val_domain_loss = _compute_val_domain_loss()
          if val_domain_loss is not None:
              val_domain_losses.append(val_domain_loss)
  
          #print(f"Epoch {epoch + 1} ({kernel_info} kernel{loss_info}{adversarial_info})")
          print(f"Epoch {epoch + 1}")
          if use_adversarial:
              print(f"  Train Loss: {epoch_loss:.4f}")
          else:
              print(f"  Train Loss: {epoch_loss:.4f}")
  
          print(f"  Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}, Val MAE: {val_mae:.4f}")
          epoch_cosines = []
          for i, cell in enumerate(tot_cell_list):
              mse_i = ((all_outputs[:, i] - all_targets[:, i]) ** 2).mean().item()
              cos_i = torch.nn.functional.cosine_similarity(all_outputs[:, i], all_targets[:, i], dim=0).item()
              epoch_cosines.append(cos_i)
              #print(f"  Cell {cell}: Val MSE={mse_i:.4f}, Val cosine similarity={cos_i:.4f}")
  
          val_cosines.append(float(np.mean(epoch_cosines)))
  
          # ---------------- Optional prediction after each epoch ----------------
          if predict_per_epoch:
              print(f"\n--- Performing prediction for epoch {epoch + 1} ---")
  
              temp_model_dir = os.path.join(out_dir, f"temp_epoch_{epoch + 1}")
              os.makedirs(temp_model_dir, exist_ok=True)
  
              torch.save(model.state_dict(), f"{temp_model_dir}/multi_graph_model.pt")
  
              if kernel_type == 'pca':
                  pca_data = {
                      'pca': model.pca,
                      'scaler': model.scaler,
                      'explained_ratio': float(np.sum(model.pca.explained_variance_ratio_))
                  }
                  with open(f"{temp_model_dir}/pca_components.pkl", "wb") as f:
                      pickle.dump(pca_data, f)
  
              if kernel_type not in ['pca', 'linear']:
                  with open(f"{temp_model_dir}/multi_graph_graphs.pkl", "wb") as f:
                      pickle.dump(graph_list, f)
  
              try:
                  self.fit(
                      expression=bulk_adata_full,
                      cell_list=tot_cell_list,
                      sc_adata=sc_adata,
                      annotation_key=annotation_key,
                      model_folder=temp_model_dir,
                      out_dir=out_dir,
                      project=prediction_project,
                      file_dir=prediction_file_dir,
                      save=True,
                      is_st=is_st,
                      top_k=top_k if top_k is not None else self.top_k,
                      kernel_type=kernel_type,
                      wavelet_type=wavelet_type,
                      epoch_suffix=epoch + 1
                  )
                  print(f"Prediction completed for epoch {epoch + 1}")
              except Exception as e:
                  print(f"Error during prediction for epoch {epoch + 1}: {str(e)}")
                  import traceback
                  traceback.print_exc()
              finally:
                  import shutil
                  try:
                      shutil.rmtree(temp_model_dir)
                  except Exception as cleanup_error:
                      print(f"Warning: Could not remove temp folder {temp_model_dir}: {cleanup_error}")
  
              print(f"--- Prediction for epoch {epoch + 1} finished ---\n")
              model.train()
  
          # ---------------- Scheduler step ----------------
          main_scheduler.step(val_loss)
          if adversarial_scheduler is not None:
              adversarial_scheduler.step(val_loss)
          '''
          if use_adversarial and adversarial_optimizer is not None:
              print("adv lr:", adversarial_optimizer.param_groups[0]["lr"])
          main_lr = main_optimizer.param_groups[0]["lr"]
          print(f"[epoch {epoch}] main lr: {main_lr:.6g}")
          '''
  
          # ---------------- Enhanced early stopping design ----------------
          current_val_loss = val_loss
  
          if previous_val_loss is not None:
              if current_val_loss >= previous_val_loss:
                  consecutive_increases += 1
                  '''
                  print(
                      f"  Val loss(for ES) increased: {previous_val_loss:.4f} -> {current_val_loss:.4f} "
                      f"(consecutive increases: {consecutive_increases}/{overfitting_threshold})"
                  )
                  '''
                  if consecutive_increases >= overfitting_threshold and not overfitting_locked:
                      overfitting_locked = True
                      locked_epoch = epoch
              else:
                  if consecutive_increases > 0:
                     '''
                      print(
                          f"  Val loss(for ES) decreased: {previous_val_loss:.4f} -> {current_val_loss:.4f} "
                          f"(reset consecutive increase counter from {consecutive_increases})"
                      )
                      '''
                  consecutive_increases = 0
  
          if current_val_loss < low_point_val:
              '''
              print(
                  f"  >> New low-point val_loss(for ES): {current_val_loss:.4f} "
                  f"(previous low point: {low_point_val if low_point_val < float('inf') else 'inf'})"
              )
              '''
              low_point_val = current_val_loss
              post_lowpoint_worse_count = 0
          else:
              post_lowpoint_worse_count += 1
              '''
              print(
                  f"  -- Epoch worse than low point: current_val_loss={current_val_loss:.4f}, "
                  f"low_point_val={low_point_val:.4f}, post-lowpoint worse count={post_lowpoint_worse_count}/"
                  f"{lock_after_worse_epochs}"
              )
              '''
              if post_lowpoint_worse_count >= lock_after_worse_epochs and not overfitting_locked:
                  overfitting_locked = True
                  locked_epoch = epoch
  
          can_update_best = (not overfitting_locked) and (current_val_loss < best_val_loss)
  
          if can_update_best:
              old_best = best_val_loss
              best_val_loss = current_val_loss
              best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
              wait = 0
          else:
              wait += 1
              
  
          previous_val_loss = current_val_loss
  
          if wait >= patience:
              print(f"\nEarly stopping at epoch {epoch + 1} (no better val_loss for {patience} epochs)")
              break
  
      # ---------------- End of training: save best model ----------------
      if best_state is not None:
          torch.save(best_state, f"{out_dir}/multi_graph_model.pt")
          #print(f"Best model saved with val_loss: {best_val_loss:.4f}")
      else:
          torch.save(model.state_dict(), f"{out_dir}/multi_graph_model.pt")
          #print("Final model saved (no early-stopping improvement recorded)")
  
      if kernel_type == 'pca':
          pca_data = {
              'pca': model.pca,
              'scaler': model.scaler,
              'explained_ratio': float(np.sum(model.pca.explained_variance_ratio_))
          }
          with open(f"{out_dir}/pca_components.pkl", "wb") as f:
              pickle.dump(pca_data, f)
          #print("PCA components saved")
  
      if kernel_type not in ['pca', 'linear']:
          with open(f"{out_dir}/multi_graph_graphs.pkl", "wb") as f:
              pickle.dump(graph_list, f)
  
      training_history = {
          'train_losses': train_losses,
          'val_losses': val_losses,
          'val_mses': val_mses,
          'val_cosines': val_cosines,
          'val_domain_losses': val_domain_losses,  # NEW
          'loss_type': loss_type,
          'kernel_type': kernel_type,
          'kl_weight': kl_weight if loss_type == 'hybrid' else None,
          'mse_weight': mse_weight if loss_type == 'hybrid' else None,
          'use_adversarial': use_adversarial,
          'domain_loss_weight': domain_loss_weight if use_adversarial else None,
          'grl_lambda': grl_lambda if use_adversarial else None,
          'val_domain_max_samples': val_domain_max_samples if use_adversarial else None,
          'top_k': top_k if top_k is not None else self.top_k,
      }
  
      with open(f"{out_dir}/training_history.pkl", "wb") as f:
          pickle.dump(training_history, f)
  
