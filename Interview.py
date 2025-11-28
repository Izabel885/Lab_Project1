"""
PBMC3k Single-Cell RNA-seq Analysis Pipeline
Technical Interview Task - Izabel Ela Emekli
Date: November 28, 2025

This script performs a complete single-cell RNA-seq analysis including:
- Data loading and QC
- Normalization
- Feature selection
- PCA and UMAP
- Clustering
- Differential expression analysis
"""

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set plotting parameters
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100, facecolor='white', frameon=True)
plt.rcParams['figure.figsize'] = (8, 6)

print("="*60)
print("PBMC3k Single-Cell RNA-seq Analysis")
print("="*60)

# ============================================================================
# TASK 1: GitHub account (already exists)
# TASK 2: Download pbm3k dataset from scanpy
# ============================================================================

print("\n[TASK 2] Loading PBMC3k dataset...")
adata = sc.datasets.pbmc3k()
print(f"Dataset loaded: {adata.n_obs} cells × {adata.n_vars} genes")
print(f"Data shape: {adata.shape}")

# ============================================================================
# TASK 3: Plot scanpy QC figures and assess if data needs trimming
# ============================================================================

print("\n[TASK 3] Performing Quality Control Analysis...")

# Calculate QC metrics
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

# Add mitochondrial gene percentage
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# Create QC plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Quality Control Metrics - PBMC3k Dataset', fontsize=16, fontweight='bold')

# Plot 1: Total counts per cell
axes[0, 0].hist(adata.obs['total_counts'], bins=100, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Total counts per cell', fontsize=11)
axes[0, 0].set_ylabel('Number of cells', fontsize=11)
axes[0, 0].set_title('Total UMI Counts Distribution', fontweight='bold')
axes[0, 0].axvline(x=np.median(adata.obs['total_counts']), color='red', 
                   linestyle='--', label=f"Median: {np.median(adata.obs['total_counts']):.0f}")
axes[0, 0].legend()

# Plot 2: Number of genes detected per cell
axes[0, 1].hist(adata.obs['n_genes_by_counts'], bins=100, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_xlabel('Number of genes detected', fontsize=11)
axes[0, 1].set_ylabel('Number of cells', fontsize=11)
axes[0, 1].set_title('Genes Detected per Cell', fontweight='bold')
axes[0, 1].axvline(x=np.median(adata.obs['n_genes_by_counts']), color='red', 
                   linestyle='--', label=f"Median: {np.median(adata.obs['n_genes_by_counts']):.0f}")
axes[0, 1].legend()

# Plot 3: Mitochondrial gene percentage
axes[1, 0].hist(adata.obs['pct_counts_mt'], bins=100, edgecolor='black', alpha=0.7, color='orange')
axes[1, 0].set_xlabel('% Mitochondrial genes', fontsize=11)
axes[1, 0].set_ylabel('Number of cells', fontsize=11)
axes[1, 0].set_title('Mitochondrial Gene Percentage', fontweight='bold')
axes[1, 0].axvline(x=5, color='red', linestyle='--', label='Threshold: 5%')
axes[1, 0].legend()

# Plot 4: Scatter plot - genes vs counts colored by MT%
scatter = axes[1, 1].scatter(adata.obs['total_counts'], 
                            adata.obs['n_genes_by_counts'],
                            c=adata.obs['pct_counts_mt'], 
                            cmap='viridis', 
                            alpha=0.5, 
                            s=10)
axes[1, 1].set_xlabel('Total counts', fontsize=11)
axes[1, 1].set_ylabel('Number of genes', fontsize=11)
axes[1, 1].set_title('Genes vs Counts (colored by MT%)', fontweight='bold')
plt.colorbar(scatter, ax=axes[1, 1], label='% MT genes')

plt.tight_layout()
plt.savefig('QC_metrics_before_filtering.png', dpi=300, bbox_inches='tight')
plt.show()

# Print QC statistics
print("\n--- QC Statistics Before Filtering ---")
print(f"Total cells: {adata.n_obs}")
print(f"Total genes: {adata.n_vars}")
print(f"Mean counts per cell: {adata.obs['total_counts'].mean():.2f}")
print(f"Median counts per cell: {adata.obs['total_counts'].median():.2f}")
print(f"Mean genes per cell: {adata.obs['n_genes_by_counts'].mean():.2f}")
print(f"Median genes per cell: {adata.obs['n_genes_by_counts'].median():.2f}")
print(f"Mean MT%: {adata.obs['pct_counts_mt'].mean():.2f}%")
print(f"Median MT%: {adata.obs['pct_counts_mt'].median():.2f}%")

# Filter cells based on QC metrics
print("\n--- Filtering cells based on QC thresholds ---")
print("Thresholds applied:")
print("  - Min genes: 200")
print("  - Max genes: 2500")
print("  - Max MT%: 5%")

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata = adata[adata.obs['n_genes_by_counts'] < 2500, :]
adata = adata[adata.obs['pct_counts_mt'] < 5, :]

print(f"\nCells after filtering: {adata.n_obs}")
print(f"Genes after filtering: {adata.n_vars}")

# ============================================================================
# TASK 4: Normalize the data
# ============================================================================

print("\n[TASK 4] Normalizing data...")

# Store raw counts for later use
adata.layers['counts'] = adata.X.copy()

# Normalize to 10,000 counts per cell
sc.pp.normalize_total(adata, target_sum=1e4)

# Log-transform the data
sc.pp.log1p(adata)

# Store normalized data
adata.raw = adata

print("Normalization complete: normalized to 10,000 counts per cell and log-transformed")

# ============================================================================
# TASK 5: Select highly variable genes
# ============================================================================

print("\n[TASK 5] Identifying highly variable genes...")

# Identify highly variable genes
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

print(f"Number of highly variable genes: {sum(adata.var['highly_variable'])}")

# Plot highly variable genes
sc.pl.highly_variable_genes(adata, show=False)
plt.gcf().set_size_inches(10, 6)
plt.suptitle('Highly Variable Genes Selection', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('highly_variable_genes.png', dpi=300, bbox_inches='tight')
plt.show()

# Filter to keep only highly variable genes
adata = adata[:, adata.var['highly_variable']]
print(f"Dataset reduced to {adata.n_vars} highly variable genes")

# ============================================================================
# TASK 6: Assess the top 5 principal components
# ============================================================================

print("\n[TASK 6] Performing PCA and assessing top 5 components...")

# Regress out technical effects and scale
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(adata, max_value=10)

# Perform PCA
sc.tl.pca(adata, svd_solver='arpack', n_comps=50)

# Print variance explained by top 5 PCs
print("\n--- Variance Explained by Top 5 PCs ---")
for i in range(5):
    var_explained = adata.uns['pca']['variance_ratio'][i] * 100
    print(f"PC{i+1}: {var_explained:.2f}%")

cumulative_var = np.sum(adata.uns['pca']['variance_ratio'][:5]) * 100
print(f"\nCumulative variance (PC1-5): {cumulative_var:.2f}%")

# Plot variance ratio
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
axes[0].plot(range(1, 51), adata.uns['pca']['variance_ratio'], 'bo-', linewidth=2, markersize=5)
axes[0].axvline(x=5, color='red', linestyle='--', label='Top 5 PCs', linewidth=2)
axes[0].set_xlabel('Principal Component', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Variance Ratio', fontsize=12, fontweight='bold')
axes[0].set_title('Scree Plot - Variance Explained', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Cumulative variance
cumvar = np.cumsum(adata.uns['pca']['variance_ratio'])
axes[1].plot(range(1, 51), cumvar, 'go-', linewidth=2, markersize=5)
axes[1].axhline(y=0.9, color='red', linestyle='--', label='90% variance', linewidth=2)
axes[1].axvline(x=5, color='orange', linestyle='--', label='Top 5 PCs', linewidth=2)
axes[1].set_xlabel('Number of Components', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Cumulative Variance Ratio', fontsize=12, fontweight='bold')
axes[1].set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('PCA_variance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot PCA scatter plots for top 5 components
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Top Principal Components', fontsize=16, fontweight='bold')

# Plot PC pairs
pca_data = adata.obsm['X_pca']
pca_pairs = [(0, 1), (2, 3), (0, 2), (1, 3)]
pca_labels = [('PC1', 'PC2'), ('PC3', 'PC4'), ('PC1', 'PC3'), ('PC2', 'PC4')]

for idx, ((pc1, pc2), (label1, label2)) in enumerate(zip(pca_pairs, pca_labels)):
    ax = axes[idx // 2, idx % 2]
    scatter = ax.scatter(pca_data[:, pc1], pca_data[:, pc2], 
                        c=adata.obs['total_counts'], 
                        cmap='viridis', alpha=0.6, s=10)
    ax.set_xlabel(f'{label1} ({adata.uns["pca"]["variance_ratio"][pc1]*100:.1f}%)', fontweight='bold')
    ax.set_ylabel(f'{label2} ({adata.uns["pca"]["variance_ratio"][pc2]*100:.1f}%)', fontweight='bold')
    ax.set_title(f'{label1} vs {label2}', fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Total counts')
    
plt.tight_layout()
plt.savefig('PCA_top5_components.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# TASK 7: Plot a UMAP of the data
# ============================================================================

print("\n[TASK 7] Computing neighborhood graph and UMAP...")

# Compute neighborhood graph
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

# Compute UMAP
sc.tl.umap(adata)

# Plot UMAP
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sc.pl.umap(adata, color='total_counts', ax=axes[0], show=False, title='UMAP colored by Total Counts')
sc.pl.umap(adata, color='n_genes_by_counts', ax=axes[1], show=False, title='UMAP colored by Gene Counts')

plt.tight_layout()
plt.savefig('UMAP_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("UMAP computed successfully")

# ============================================================================
# TASK 8: Do a clustering on the data
# ============================================================================

print("\n[TASK 8] Performing clustering...")

# Perform clustering - use sklearn KMeans as fallback if igraph not installed
try:
    sc.tl.leiden(adata, resolution=0.5)
    cluster_key = 'leiden'
    print("Using Leiden clustering")
except (ImportError, ModuleNotFoundError):
    print("igraph not found, using KMeans clustering instead...")
    from sklearn.cluster import KMeans
    # Use UMAP coordinates for clustering
    n_clusters = 8  # Typical number of clusters for PBMC data
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    adata.obs['kmeans'] = kmeans.fit_predict(adata.obsm['X_umap'])
    adata.obs['kmeans'] = adata.obs['kmeans'].astype('category')
    adata.obs['kmeans'] = adata.obs['kmeans'].astype(str)
    cluster_key = 'kmeans'
    print(f"KMeans clustering complete with {n_clusters} clusters")

print(f"Number of clusters identified: {len(adata.obs[cluster_key].unique())}")
print("\nCluster sizes:")
print(adata.obs[cluster_key].value_counts().sort_index())

# Plot UMAP with clusters
fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.umap(adata, color=cluster_key, legend_loc='on data', 
           title=f'UMAP with {cluster_key.capitalize()} Clustering', 
           show=False, ax=ax, palette='tab20')
plt.tight_layout()
plt.savefig('UMAP_with_clusters.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# TASK 9: Highlight statistical differences between clusters
# ============================================================================

print("\n[TASK 9] Analyzing statistical differences between clusters...")

# Find marker genes for each cluster
sc.tl.rank_genes_groups(adata, cluster_key, method='wilcoxon')

# Plot top marker genes
fig = sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, show=False)
plt.tight_layout()
plt.savefig('marker_genes_by_cluster.png', dpi=300, bbox_inches='tight')
plt.show()

# Get marker genes dataframe
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
marker_genes_df = pd.DataFrame({
    group + '_genes': result['names'][group][:10] for group in groups
})

print("\n--- Top 10 Marker Genes per Cluster ---")
print(marker_genes_df.to_string())

# Plot heatmap of top marker genes
sc.pl.rank_genes_groups_heatmap(adata, n_genes=5, show_gene_labels=True, 
                                 cmap='viridis', show=False)
plt.tight_layout()
plt.savefig('marker_genes_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Dotplot of marker genes
sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, show=False)
plt.tight_layout()
plt.savefig('marker_genes_dotplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistical comparison between clusters
print("\n--- Cluster Statistics ---")
cluster_stats = pd.DataFrame({
    'Cluster': adata.obs[cluster_key].unique(),
    'n_cells': [sum(adata.obs[cluster_key] == c) for c in adata.obs[cluster_key].unique()],
    'mean_genes': [adata.obs[adata.obs[cluster_key] == c]['n_genes_by_counts'].mean() 
                   for c in adata.obs[cluster_key].unique()],
    'mean_counts': [adata.obs[adata.obs[cluster_key] == c]['total_counts'].mean() 
                    for c in adata.obs[cluster_key].unique()]
})
cluster_stats = cluster_stats.sort_values('Cluster')
print(cluster_stats.to_string(index=False))

# Violin plot comparing clusters
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc.pl.violin(adata, 'n_genes_by_counts', groupby=cluster_key, show=False, ax=axes[0])
axes[0].set_title('Genes per Cell by Cluster', fontweight='bold')
sc.pl.violin(adata, 'total_counts', groupby=cluster_key, show=False, ax=axes[1])
axes[1].set_title('Total Counts by Cluster', fontweight='bold')
plt.tight_layout()
plt.savefig('cluster_comparison_violin.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistical test between clusters (ANOVA)
print("\n--- ANOVA Test for Gene Counts Between Clusters ---")
cluster_groups = [adata.obs[adata.obs[cluster_key] == c]['n_genes_by_counts'].values 
                  for c in adata.obs[cluster_key].unique()]
f_stat, p_value = stats.f_oneway(*cluster_groups)
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4e}")
if p_value < 0.05:
    print("Result: Significant differences exist between clusters (p < 0.05)")
else:
    print("Result: No significant differences between clusters (p >= 0.05)")

print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)
print("\nGenerated files:")
print("  - QC_metrics_before_filtering.png")
print("  - highly_variable_genes.png")
print("  - PCA_variance_analysis.png")
print("  - PCA_top5_components.png")
print("  - UMAP_visualization.png")
print("  - UMAP_with_clusters.png")
print("  - marker_genes_by_cluster.png")
print("  - marker_genes_heatmap.png")
print("  - marker_genes_dotplot.png")
print("  - cluster_comparison_violin.png")
print("="*60)
