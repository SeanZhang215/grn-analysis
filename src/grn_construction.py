"""
Module for constructing Gene Regulatory Networks using GRNBoost2.
"""
import os
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from arboreto.algo import grnboost2
from scipy import stats
import networkx as nx

class GRNBuilder:
    def __init__(self, n_estimators: int = 500):
        """Initialize GRN construction pipeline.

        Args:
            n_estimators: Number of estimators for GRNBoost2
        """
        self.n_estimators = n_estimators

    def calculate_signs(
        self, 
        grn: pd.DataFrame, 
        expression_df: pd.DataFrame, 
        tf_names: List[str]
    ) -> pd.DataFrame:
        """Calculate regulatory signs (activation/repression) for each TF-target pair.

        Args:
            grn: GRN edge dataframe
            expression_df: Gene expression data
            tf_names: List of transcription factor names

        Returns:
            GRN dataframe with signed importance scores
        """
        signs = []

        for _, row in grn.iterrows():
            tf = row['TF']
            target = row['target']

            if tf in tf_names and target in expression_df.columns:
                tf_expression = expression_df[tf]
                target_expression = expression_df[target]
                correlation = np.corrcoef(tf_expression, target_expression)[0, 1]
                signs.append(np.sign(correlation))
            else:
                signs.append(0)

        grn['signed_importance'] = grn['importance'] * signs / self.n_estimators
        return grn

    def run_grnboost2(
        self,
        expression_df: pd.DataFrame,
        tf_names: List[str],
        seed: int
    ) -> pd.DataFrame:
        """Run GRNBoost2 algorithm to construct GRN.

        Args:
            expression_df: Gene expression data
            tf_names: List of transcription factor names
            seed: Random seed

        Returns:
            GRN edge dataframe
        """
        network = grnboost2(
            expression_data=expression_df,
            tf_names=tf_names,
            seed=seed,
            verbose=False
        )
        return network

    def build_grn(
        self,
        expression_df: pd.DataFrame,
        tf_names: List[str],
        seeds: List[int],
        top_edges_fraction: float = 0.1
    ) -> pd.DataFrame:
        """Build GRN by running GRNBoost2 with multiple seeds and aggregating results.

        Args:
            expression_df: Gene expression data
            tf_names: List of transcription factor names
            seeds: List of random seeds
            top_edges_fraction: Fraction of top edges to keep

        Returns:
            Aggregated GRN edge dataframe
        """
        networks = []
        
        for seed in seeds:
            # Run GRNBoost2
            network = self.run_grnboost2(expression_df, tf_names, seed)
            
            # Keep top edges
            top_edges = int(len(network) * top_edges_fraction)
            network = network.nlargest(top_edges, 'importance')
            
            # Calculate signs
            network = self.calculate_signs(network, expression_df, tf_names)
            networks.append(network)

        # Aggregate networks
        combined_network = pd.concat(networks, axis=0)
        aggregated_network = self._aggregate_networks(combined_network)
        
        return aggregated_network

    def _aggregate_networks(self, combined_network: pd.DataFrame) -> pd.DataFrame:
        """Aggregate multiple GRN runs into a consensus network.

        Args:
            combined_network: Combined network from multiple runs

        Returns:
            Aggregated network
        """
        agg_network = combined_network.groupby(['TF', 'target']).agg({
            'importance': 'mean',
            'signed_importance': 'mean'
        }).reset_index()

        # Calculate edge appearance frequency
        edge_counts = combined_network.groupby(['TF', 'target']).size()
        total_runs = len(combined_network) / len(edge_counts)
        agg_network['appearance_freq'] = agg_network.apply(
            lambda x: edge_counts[(x['TF'], x['target'])] / total_runs, 
            axis=1
        )

        return agg_network

class NetworkAnalyzer:
    """Analyze constructed GRNs."""

    @staticmethod
    def calculate_network_stats(grn: pd.DataFrame) -> Dict:
        """Calculate basic network statistics.

        Args:
            grn: GRN edge dataframe

        Returns:
            Dictionary of network statistics
        """
        stats = {
            'num_edges': len(grn),
            'num_tfs': grn['TF'].nunique(),
            'num_targets': grn['target'].nunique(),
            'pos_reg_frac': (grn['signed_importance'] > 0).mean(),
            'neg_reg_frac': (grn['signed_importance'] < 0).mean()
        }
        return stats

    @staticmethod
    def identify_hub_regulators(
        grn: pd.DataFrame,
        min_targets: int = 5
    ) -> pd.DataFrame:
        """Identify hub regulators based on number of targets.

        Args:
            grn: GRN edge dataframe
            min_targets: Minimum number of targets to be considered a hub

        Returns:
            DataFrame of hub regulators and their properties
        """
        hub_stats = grn.groupby('TF').agg({
            'target': 'count',
            'signed_importance': ['mean', 'std']
        }).reset_index()
        
        hub_stats.columns = ['TF', 'num_targets', 'mean_importance', 'std_importance']
        hub_stats = hub_stats[hub_stats['num_targets'] >= min_targets]
        
        return hub_stats.sort_values('num_targets', ascending=False)

    @staticmethod
    def create_network_graph(
        grn: pd.DataFrame,
        min_importance: float = 0.1
    ) -> nx.DiGraph:
        """Create NetworkX graph from GRN.

        Args:
            grn: GRN edge dataframe
            min_importance: Minimum absolute importance for including edges

        Returns:
            NetworkX directed graph
        """
        # Filter edges by importance
        filtered_grn = grn[abs(grn['signed_importance']) >= min_importance]
        
        # Create graph
        G = nx.from_pandas_edgelist(
            filtered_grn,
            'TF',
            'target',
            ['signed_importance'],
            create_using=nx.DiGraph()
        )
        
        return G