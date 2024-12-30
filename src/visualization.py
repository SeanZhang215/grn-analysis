"""
Module for visualizing GRN analysis results.
"""
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

class NetworkVisualizer:
    """Visualize GRN networks and analysis results."""
    
    @staticmethod
    def plot_network(
        grn_df: pd.DataFrame,
        target_gene: Optional[str] = None,
        min_appearance: float = 0.8,
        min_edges: int = 1,
        figsize: Tuple[int, int] = (16, 16)
    ) -> None:
        """Plot GRN network visualization.

        Args:
            grn_df: GRN edge dataframe
            target_gene: Optional target gene to focus on
            min_appearance: Minimum edge appearance frequency
            min_edges: Minimum number of edges for a node
            figsize: Figure size
        """
        # Filter edges
        filtered_df = grn_df[grn_df['TargetInputAppearPerc'] >= min_appearance]
        
        if target_gene:
            filtered_df = filtered_df[filtered_df['TargetGeneID'] == target_gene]
        
        # Count edges per input gene
        input_gene_counts = filtered_df['InputGeneID'].value_counts()
        input_genes_to_keep = input_gene_counts[input_gene_counts > min_edges].index
        filtered_df = filtered_df[filtered_df['InputGeneID'].isin(input_genes_to_keep)]

        # Create graph
        G = nx.DiGraph()
        for _, row in filtered_df.iterrows():
            G.add_edge(
                row['InputGeneID'],
                row['TargetGeneID'],
                signed_importance=row['InputImportance']
            )

        # Set up node positions
        targets = filtered_df['TargetGeneID'].unique()
        tfs = filtered_df['InputGeneID'].unique()

        if target_gene:
            target_pos = {target_gene: (0, 0)}
            angle = 2 * np.pi / len(tfs)
            tf_pos = {
                tf: (1.5 * np.cos(i * angle), 1.5 * np.sin(i * angle))
                for i, tf in enumerate(tfs)
            }
        else:
            target_pos = nx.circular_layout(G.subgraph(targets))
            angle = 2 * np.pi / len(tfs)
            tf_pos = {
                tf: (1.5 * np.cos(i * angle), 1.5 * np.sin(i * angle))
                for i, tf in enumerate(tfs)
            }

        pos = {**target_pos, **tf_pos}

        # Create plot
        plt.figure(figsize=figsize)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=targets,
            node_color='orange',
            node_size=3000,
            alpha=0.8
        )
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=tfs,
            node_color='lightgreen',
            node_size=2000,
            alpha=0.8
        )

        # Draw edges
        edges = [(u, v) for (u, v) in G.edges() if u in pos and v in pos]
        edge_colors = [
            'red' if G[u][v]['signed_importance'] < 0 else 'blue'
            for u, v in edges
        ]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edges,
            edge_color=edge_colors,
            width=2,
            alpha=0.6,
            arrows=True,
            arrowsize=20
        )

        # Add labels
        labels = {node: node for node in G.nodes() if node in pos}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

        plt.axis('off')
        plt.tight_layout()

    @staticmethod
    def plot_correlation_importance(
        grn_df: pd.DataFrame,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """Plot correlation between gene expression correlation and importance scores.

        Args:
            grn_df: GRN edge dataframe
            figsize: Figure size
        """
        corr = grn_df['InputImportance'].corr(grn_df['PearsonR'])

        g = sns.jointplot(
            x=grn_df['InputImportance'].to_numpy(),
            y=grn_df['PearsonR'].to_numpy(),
            height=figsize[0]/2
        )

        g.ax_joint.set_xlabel('InputImportance')
        g.ax_joint.set_ylabel('PearsonR')

        g.fig.suptitle(f"Correlation vs. Importance (r = {corr:.2f})")
        g.fig.tight_layout()
        g.fig.subplots_adjust(top=0.95)

    @staticmethod
    def plot_genki_validation_summary(
        grn_df: pd.DataFrame,
        figsize: Tuple[int, int] = (10, 10)
    ) -> None:
        """Plot summary of GenKI validation results.

        Args:
            grn_df: GRN edge dataframe with GenKI validation
            figsize: Figure size
        """
        # Count validation statistics
        total_targets = grn_df['TargetGeneID'].nunique()
        negative_regulators_df = grn_df[grn_df['Regulator'] == 'negative']
        unique_negative_targets = negative_regulators_df['TargetGeneID'].nunique()
        validated_negative_targets = negative_regulators_df[
            negative_regulators_df['GenKIValid'] == 1
        ]['TargetGeneID'].nunique()

        # Create pie chart data
        sizes = [
            total_targets - unique_negative_targets,
            unique_negative_targets - validated_negative_targets,
            validated_negative_targets
        ]
        labels = [
            f'No Negative Regulator ({sizes[0]})',
            f'Negative Regulator Not Validated ({sizes[1]})',
            f'Negative Regulator Validated ({sizes[2]})'
        ]
        colors = ['lightgrey', 'orange', 'green']

        # Plot pie chart
        plt.figure(figsize=figsize)
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'fontsize': 12}
        )
        plt.axis('equal')
        
    @staticmethod
    def plot_genki_distribution(
        grn_df: pd.DataFrame,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """Plot distribution of PearsonR values for GenKI validated vs non-validated edges.

        Args:
            grn_df: GRN edge dataframe with GenKI validation
            figsize: Figure size
        """
        plt.figure(figsize=figsize)

        # Separate data by GenKI validation
        validated = grn_df[grn_df['GenKIValid'] == 1]['PearsonR']
        not_validated = grn_df[grn_df['GenKIValid'] == 0]['PearsonR']

        # Plot distributions
        sns.kdeplot(
            validated,
            fill=True,
            color="blue",
            label="GenKI Validated",
            alpha=0.5
        )
        sns.kdeplot(
            not_validated,
            fill=True,
            color="red",
            label="Not Validated",
            alpha=0.5
        )

        plt.title('Distribution of PearsonR by GenKI Validation')
        plt.xlabel('PearsonR')
        plt.ylabel('Density')
        plt.legend()

    def plot_tissue_comparison(
        self,
        tissue_grns: Dict[str, pd.DataFrame],
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """Plot comparison of GRN edges across tissues.

        Args:
            tissue_grns: Dictionary mapping tissue names to their GRN dataframes
            figsize: Figure size
        """
        # Extract negative regulator edges validated by GenKI for each tissue
        negative_regulator_edges = {}
        for tissue, df in tissue_grns.items():
            negative_edges = df[
                (df['InputImportance'] < 0) & 
                (df['GenKIValid'] == 1)
            ]
            negative_regulator_edges[tissue] = set(
                zip(negative_edges['InputGeneID'], negative_edges['TargetGeneID'])
            )

        # Find shared edges across all tissues
        all_tissues = list(tissue_grns.keys())
        shared_edges = set.intersection(*negative_regulator_edges.values())
        
        # Create upset plot data
        intersection_sizes = []
        for tissue in all_tissues:
            intersection_sizes.append(len(negative_regulator_edges[tissue] - shared_edges))
        intersection_sizes.append(len(shared_edges))

        # Plot bar chart
        plt.figure(figsize=figsize)
        x_labels = all_tissues + ['Shared']
        plt.bar(x_labels, intersection_sizes)
        plt.title('Negative Regulator Edges Across Tissues')
        plt.xlabel('Tissue')
        plt.ylabel('Number of Edges')
        plt.xticks(rotation=45)
        plt.tight_layout()

    def create_summary_dashboard(
        self,
        grn_df: pd.DataFrame,
        target_gene: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> None:
        """Create a comprehensive visualization dashboard.

        Args:
            grn_df: GRN edge dataframe with GenKI validation
            target_gene: Optional target gene to focus on
            output_dir: Optional directory to save plots
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Network visualization
        plt.subplot(2, 2, 1)
        self.plot_network(grn_df, target_gene)
        
        # Correlation vs Importance
        plt.subplot(2, 2, 2)
        self.plot_correlation_importance(grn_df)
        
        # GenKI validation summary
        plt.subplot(2, 2, 3)
        self.plot_genki_validation_summary(grn_df)
        
        # GenKI distribution
        plt.subplot(2, 2, 4)
        self.plot_genki_distribution(grn_df)
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"summary_{'all' if target_gene is None else target_gene}.png"
                )
            )