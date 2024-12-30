"""
Module for preprocessing and preparing gene expression data for GRN analysis.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
from typing import List, Optional, Tuple, Dict
from scipy import sparse
from kneed import KneeLocator

class ExpressionDataPreparer:
    def __init__(
        self,
        mean_tpm_thresh: float = 1.0,
        min_genes: int = 1,
        min_obs: int = 1,
        variance_quantile: float = 0.0,
        random_seed: int = 42
    ):
        """Initialize the data preparation pipeline.

        Args:
            mean_tpm_thresh: Minimum mean TPM threshold for filtering genes
            min_genes: Minimum number of genes per observation
            min_obs: Minimum number of observations per gene
            variance_quantile: Quantile threshold for variance filtering
            random_seed: Random seed for reproducibility
        """
        self.mean_tpm_thresh = mean_tpm_thresh
        self.min_genes = min_genes
        self.min_obs = min_obs
        self.variance_quantile = variance_quantile
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def load_expression_data(self, data_path: str) -> ad.AnnData:
        """Load expression data from h5ad file.

        Args:
            data_path: Path to h5ad file

        Returns:
            Loaded AnnData object
        """
        adata = ad.read_h5ad(data_path)
        return adata

    def filter_data(self, adata: ad.AnnData) -> ad.AnnData:
        """Apply filtering steps to expression data.

        Args:
            adata: Input AnnData object

        Returns:
            Filtered AnnData object
        """
        # Filter by mean TPM
        keep_genes = adata.var_names[np.where(adata.X.toarray().mean(axis=0) > self.mean_tpm_thresh)[0]]
        adata = adata[:, keep_genes]

        # Filter cells by minimum genes
        sc.pp.filter_cells(adata, min_genes=0)
        sc.pp.filter_cells(adata, min_genes=self.min_genes)

        # Filter genes by minimum cells
        sc.pp.filter_genes(adata, min_cells=0)
        sc.pp.filter_genes(adata, min_cells=self.min_obs)

        # Log transform
        sc.pp.log1p(adata)

        # Variance filtering
        var_ser = pd.Series(adata.X.toarray().std(axis=0)**2)
        var_thresh = var_ser.quantile(q=self.variance_quantile)
        where_var = np.where(var_ser >= var_thresh)[0]
        adata = adata[:, where_var]

        return adata

    def compute_pca(self, adata: ad.AnnData, n_comps: int = 100) -> Tuple[ad.AnnData, int]:
        """Compute PCA and determine optimal number of components.

        Args:
            adata: Input AnnData object
            n_comps: Number of PCs to compute

        Returns:
            Tuple of (AnnData with PCA, optimal number of PCs)
        """
        sc.tl.pca(adata, n_comps=n_comps)
        
        var_ratios = adata.uns['pca']['variance_ratio']
        kneedle = KneeLocator(
            range(len(var_ratios)),
            var_ratios,
            S=1.0,
            curve="convex",
            direction="decreasing"
        )
        
        adata.uns['pca_vr_knee_point'] = kneedle.knee
        return adata, kneedle.knee

    def process_data(self, data_path: str) -> ad.AnnData:
        """Run complete data processing pipeline.

        Args:
            data_path: Path to input h5ad file

        Returns:
            Processed AnnData object
        """
        adata = self.load_expression_data(data_path)
        adata = self.filter_data(adata)
        adata, _ = self.compute_pca(adata)
        return adata

class BatchProcessor:
    """Process multiple expression datasets for GRN analysis."""
    
    def __init__(self, preparer: ExpressionDataPreparer):
        """Initialize with a data preparer instance."""
        self.preparer = preparer
        
    def process_multiple_tissues(
        self, 
        input_dir: str, 
        output_dir: str,
        tissues: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """Process expression data for multiple tissues.

        Args:
            input_dir: Directory containing input h5ad files
            output_dir: Directory to save processed files
            tissues: List of tissue names to process

        Returns:
            Dictionary mapping tissue names to processed dataframes
        """
        os.makedirs(output_dir, exist_ok=True)
        processed_dfs = {}

        for tissue in tissues:
            input_path = os.path.join(input_dir, f"{tissue}_tpm.h5ad")
            if os.path.exists(input_path):
                # Process data
                adata = self.preparer.process_data(input_path)
                
                # Convert to dataframe
                df = adata.to_df()
                processed_dfs[tissue] = df
                
                # Save processed data
                output_path = os.path.join(output_dir, f"{tissue}_processed.h5ad")
                adata.write(output_path)

        return processed_dfs