from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Abstract Base Class for Multivariate Analysis
# ----------------------------------------------
# This class defines a template for performing multivariate analysis.
# Subclasses can override specific steps like correlation heatmap and pair plot generation.
class MultivariateAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Perform a comprehensive multivariate analysis by generating a correlation heatmap and pair plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: This method orchestrates the multivariate analysis process.
        """
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generate and display a heatmap of the correlations between features.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: This method should generate and display a correlation heatmap.
        """
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generate and display a pair plot of the selected features.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: This method should generate and display a pair plot.
        """
        pass


# KDD99-Specific Multivariate Analysis
# -------------------------------------
# This class implements advanced multivariate analysis tailored for the KDD99 dataset.
class KDD99MultivariateAnalysis(MultivariateAnalysisTemplate):
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generates and displays a correlation heatmap for the numerical features in the KDD99 dataset.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: Displays a heatmap showing correlations between numerical features.
        """
        # Select only numerical features for correlation analysis
        numerical_features = df.select_dtypes(include=['int64', 'float64'])
        
        # Remove label column if it exists and is numerical
        if 'label' in numerical_features.columns:
            numerical_features = numerical_features.drop('label', axis=1)
        
        if len(numerical_features.columns) == 0:
            print("No numerical features found for correlation analysis.")
            return
        
        # Create correlation matrix
        corr_matrix = numerical_features.corr()
        
        # Create subplots for different correlation views
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('KDD99 Dataset - Correlation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Full correlation heatmap
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                   square=True, ax=axes[0, 0])
        axes[0, 0].set_title('Full Correlation Matrix')
        
        # 2. High correlation heatmap (|correlation| > 0.5)
        high_corr_mask = (abs(corr_matrix) > 0.5) & (abs(corr_matrix) < 1.0)
        high_corr_matrix = corr_matrix.where(high_corr_mask)
        
        sns.heatmap(high_corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=axes[0, 1])
        axes[0, 1].set_title('High Correlations (|r| > 0.5)')
        
        # 3. Top correlated feature pairs
        # Get upper triangle of correlation matrix
        upper_triangle = corr_matrix.where(
            pd.DataFrame(True, index=corr_matrix.index, columns=corr_matrix.columns).values != 
            pd.DataFrame(True, index=corr_matrix.index, columns=corr_matrix.columns).values.T
        )
        
        # Find top correlations
        corr_pairs = upper_triangle.unstack().dropna().abs().sort_values(ascending=False)
        top_10_pairs = corr_pairs.head(10)
        
        axes[1, 0].barh(range(len(top_10_pairs)), top_10_pairs.values)
        axes[1, 0].set_yticks(range(len(top_10_pairs)))
        axes[1, 0].set_yticklabels([f"{pair[0]} - {pair[1]}" for pair in top_10_pairs.index], fontsize=8)
        axes[1, 0].set_xlabel('Absolute Correlation')
        axes[1, 0].set_title('Top 10 Feature Correlations')
        
        # 4. Correlation statistics
        axes[1, 1].axis('off')
        
        # Calculate correlation statistics
        corr_values = corr_matrix.values
        corr_values = corr_values[pd.DataFrame(True, index=corr_matrix.index, columns=corr_matrix.columns).values != 
                                 pd.DataFrame(True, index=corr_matrix.index, columns=corr_matrix.columns).values.T]
        
        high_corr_count = (abs(corr_values) > 0.7).sum()
        moderate_corr_count = ((abs(corr_values) > 0.3) & (abs(corr_values) <= 0.7)).sum()
        low_corr_count = (abs(corr_values) <= 0.3).sum()
        
        stats_text = f"""
        Correlation Statistics for KDD99 Dataset:
        
        Total Feature Pairs: {len(corr_values):,}
        
        Correlation Strength Distribution:
        • High (|r| > 0.7): {high_corr_count:,} pairs
        • Moderate (0.3 < |r| ≤ 0.7): {moderate_corr_count:,} pairs  
        • Low (|r| ≤ 0.3): {low_corr_count:,} pairs
        
        Strongest Positive Correlations:
        """
        
        positive_corrs = corr_pairs[corr_pairs > 0].head(5)
        for i, (pair, corr) in enumerate(positive_corrs.items(), 1):
            stats_text += f"\n        {i}. {pair[0]} ↔ {pair[1]}: {corr:.3f}"
        
        stats_text += "\n\n        Strongest Negative Correlations:"
        negative_corrs = corr_pairs[corr_pairs < 0].head(5)
        for i, (pair, corr) in enumerate(negative_corrs.items(), 1):
            stats_text += f"\n        {i}. {pair[0]} ↔ {pair[1]}: {corr:.3f}"
        
        stats_text += f"""
        
        Multicollinearity Warning:
        Features with |r| > 0.8: {(abs(corr_values) > 0.8).sum()} pairs
        
        These high correlations may indicate:
        • Redundant features
        • Need for feature selection
        • Potential multicollinearity issues
        """
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generates and displays a pair plot for selected KDD99 features.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: Displays a pair plot for the selected features.
        """
        # Select key features for pair plot (to avoid overcrowding)
        key_features = [
            'duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count',
            'serror_rate', 'srv_serror_rate', 'dst_host_count', 'dst_host_srv_count'
        ]
        
        # Filter features that exist in the dataset
        available_features = [feat for feat in key_features if feat in df.columns]
        
        if len(available_features) < 2:
            print("Not enough numerical features available for pair plot.")
            return
        
        # Limit to first 6 features for readability
        selected_features = available_features[:6]
        
        # Add attack category if label exists
        if 'label' in df.columns:
            # Create attack categories
            attack_types = {
                'normal': ['normal.'],
                'dos': ['back.', 'land.', 'neptune.', 'pod.', 'smurf.', 'teardrop.'],
                'probe': ['ipsweep.', 'nmap.', 'portsweep.', 'satan.'],
                'r2l': ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'phf.', 'spy.', 'warezclient.', 'warezmaster.'],
                'u2r': ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.']
            }
            
            attack_category = {}
            for category, attacks in attack_types.items():
                for attack in attacks:
                    attack_category[attack] = category
            
            df_temp = df.copy()
            df_temp['attack_category'] = df_temp['label'].map(attack_category)
            df_temp = df_temp.dropna(subset=['attack_category'])
            
            # Sample data for better visualization (pair plots can be slow with large datasets)
            if len(df_temp) > 2000:
                df_sample = df_temp.sample(2000, random_state=42)
            else:
                df_sample = df_temp
            
            # Create pair plot with attack category coloring
            selected_data = df_sample[selected_features + ['attack_category']]
            
            print(f"Generating pair plot for features: {', '.join(selected_features)}")
            print(f"Using {len(df_sample):,} samples for visualization")
            
            pairplot = sns.pairplot(selected_data, hue='attack_category', 
                                   diag_kind='hist', plot_kws={'alpha': 0.6})
            pairplot.fig.suptitle('KDD99 Dataset - Pair Plot by Attack Category', 
                                 y=1.02, fontsize=16, fontweight='bold')
            
        else:
            # Create simple pair plot without attack category
            selected_data = df[selected_features]
            
            if len(selected_data) > 2000:
                selected_data_sample = selected_data.sample(2000, random_state=42)
            else:
                selected_data_sample = selected_data
            
            print(f"Generating pair plot for features: {', '.join(selected_features)}")
            print(f"Using {len(selected_data_sample):,} samples for visualization")
            
            pairplot = sns.pairplot(selected_data_sample, diag_kind='hist')
            pairplot.fig.suptitle('KDD99 Dataset - Feature Pair Plot', 
                                 y=1.02, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.show()


# Simple Multivariate Analysis (for general use)
# -----------------------------------------------
class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generates and displays a correlation heatmap for the numerical features in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: Displays a heatmap showing correlations between numerical features.
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generates and displays a pair plot for the selected features in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: Displays a pair plot for the selected features.
        """
        sns.pairplot(df)
        plt.suptitle("Pair Plot of Selected Features", y=1.02)
        plt.show()


# Example usage
if __name__ == "__main__":
    # Example usage of the KDD99MultivariateAnalysis class.
    pass