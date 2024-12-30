"""
Module for gene knockout inference using GenKI framework.
"""
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from GenKI.preprocesing import build_adata
from GenKI.dataLoader import DataLoader
from GenKI.train import VGAE_trainer
from GenKI import utils

class GenKIAnalyzer:
    def __init__(
        self,
        epochs: int = 600,
        learning_rate: float = 7e-4,
        beta: float = 1e-4,
        seed: int = 8096
    ):
        """Initialize GenKI analysis pipeline.

        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            beta: Beta parameter for VAE
            seed: Random seed
        """
        self.hyperparams = {
            "epochs": epochs,
            "lr": learning_rate,
            "beta": beta,
            "seed": seed
        }
        torch.manual_seed(seed)
        np.random.seed(seed)

    def prepare_adjacency_matrix(
        self,
        grn_df: pd.DataFrame,
        gene_ids: List[str]
    ) -> sp.csr_matrix:
        """Create adjacency matrix from GRN edges.

        Args:
            grn_df: GRN edge dataframe
            gene_ids: List of gene IDs

        Returns:
            Sparse adjacency matrix
        """
        gene_index_map = {gene_id: idx for idx, gene_id in enumerate(gene_ids)}
        gene_count = len(gene_ids)
        adj_matrix = np.zeros((gene_count, gene_count), dtype=int)

        for _, row in grn_df.iterrows():
            input_gene = row['InputGeneID'].upper()
            target_gene = row['TargetGeneID'].upper()
            
            if input_gene in gene_index_map and target_gene in gene_index_map:
                input_index = gene_index_map[input_gene]
                target_index = gene_index_map[target_gene]
                adj_matrix[input_index, target_index] = 1
                adj_matrix[target_index, input_index] = 1

        return sp.csr_matrix(adj_matrix)

    def train_model(
        self,
        data_wrapper: DataLoader,
        log_dir: str,
        model_path: Optional[str] = None
    ) -> VGAE_trainer:
        """Train or load GenKI model.

        Args:
            data_wrapper: GenKI data loader
            log_dir: Directory for logging
            model_path: Path to existing model (optional)

        Returns:
            Trained VGAE trainer
        """
        data_wt = data_wrapper.load_data()
        
        trainer = VGAE_trainer(
            data_wt,
            epochs=self.hyperparams["epochs"],
            lr=self.hyperparams["lr"],
            log_dir=log_dir,
            beta=self.hyperparams["beta"],
            seed=self.hyperparams["seed"],
            verbose=True
        )

        if model_path and os.path.exists(model_path):
            trainer.load_model(os.path.splitext(model_path)[0])
        else:
            trainer.train()
            if model_path:
                torch.save(trainer.model.state_dict(), model_path)

        return trainer

    def analyze_target(
        self,
        adata_path: str,
        grn_path: str,
        target_gene: str,
        output_dir: str,
        model_dir: str
    ) -> pd.DataFrame:
        """Run GenKI analysis for a target gene.

        Args:
            adata_path: Path to expression data
            grn_path: Path to GRN file
            target_gene: Target gene ID
            output_dir: Output directory
            model_dir: Model directory

        Returns:
            DataFrame with analysis results
        """
        # Prepare output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # Initialize data loader
        data_wrapper = DataLoader(
            build_adata(adata_path),
            target_gene=[target_gene],
            target_cell=None,
            obs_label="ident",
            GRN_file_dir=os.path.dirname(grn_path),
            rebuild_GRN=False,
            pcNet_name=os.path.basename(grn_path).split('.')[0],
            verbose=True,
            n_cpus=1
        )

        # Train/load model
        model_path = os.path.join(model_dir, f'model_{target_gene}.th')
        trainer = self.train_model(data_wrapper, model_dir, model_path)

        # Get wild type and knockout data
        data_wt = data_wrapper.load_data()
        data_ko = data_wrapper.load_kodata()

        # Get latent variables
        z_mu_wt, z_std_wt = trainer.get_latent_vars(data_wt)
        z_mu_ko, z_std_ko = trainer.get_latent_vars(data_ko)
        
        # Calculate distance
        dis = utils.get_distance(z_mu_ko, z_std_ko, z_mu_wt, z_std_wt, by="KL")

        # Get raw results
        res_raw = utils.get_generank(data_wt, dis, rank=True)
        res_raw_path = os.path.join(output_dir, f'{target_gene}_res_raw.csv')
        res_raw.to_csv(res_raw_path)

        # Calculate null distribution
        null_path = os.path.join(output_dir, f'{target_gene}_null.npy')
        if os.path.exists(null_path):
            null = np.load(null_path)
        else:
            null = trainer.pmt(data_ko, n=100, by="KL")
            np.save(null_path, null)

        # Get final results
        results = utils.get_generank(data_wt, dis, null, bagging=0.05, cutoff=0.95)
        results = results[results['dis'] >= 0]
        
        return results

    def process_multiple_targets(
        self,
        adata_path: str,
        grn_path: str,
        target_genes: List[str],
        output_dir: str,
        model_dir: str
    ) -> Dict[str, pd.DataFrame]:
        """Run GenKI analysis for multiple target genes.

        Args:
            adata_path: Path to expression data
            grn_path: Path to GRN file
            target_genes: List of target genes
            output_dir: Output directory
            model_dir: Model directory

        Returns:
            Dictionary mapping target genes to their results
        """
        results = {}
        for target_gene in target_genes:
            results[target_gene] = self.analyze_target(
                adata_path,
                grn_path,
                target_gene,
                output_dir,
                model_dir
            )
        return results

    def validate_grn_edges(
        self,
        grn_df: pd.DataFrame,
        genki_results_dir: str
    ) -> pd.DataFrame:
        """Validate GRN edges using GenKI results.

        Args:
            grn_df: GRN edge dataframe
            genki_results_dir: Directory containing GenKI results

        Returns:
            GRN dataframe with validation results
        """
        grn_df['Regulator'] = grn_df['InputImportance'].apply(
            lambda x: 'positive' if x > 0 else 'negative'
        )

        def check_genki_valid(row: pd.Series) -> int:
            target_gene = row['TargetGeneID']
            input_gene = row['InputGeneID']
            
            filename = f"{target_gene}_res.csv"
            filepath = os.path.join(genki_results_dir, filename)
            
            if os.path.exists(filepath):
                oe_df = pd.read_csv(filepath)
                if input_gene in oe_df.iloc[:, 0].values:
                    return 1
            return 0

        grn_df['GenKIValid'] = grn_df.apply(check_genki_valid, axis=1)
        return grn_df