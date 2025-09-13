from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Abstract Base Class for Bivariate Analysis Strategy
# ----------------------------------------------------
# This class defines a common interface for bivariate analysis strategies.
# Subclasses must implement the analyze method.
class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Perform bivariate analysis on two features of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first feature/column to be analyzed.
        feature2 (str): The name of the second feature/column to be analyzed.

        Returns:
        None: This method visualizes the relationship between the two features.
        """
        pass


# Concrete Strategy for Numerical vs Numerical Analysis in KDD99
# ---------------------------------------------------------------
# This strategy analyzes the relationship between two numerical features with KDD99-specific insights.
class KDD99NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between two numerical features with KDD99-specific analysis.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first numerical feature/column to be analyzed.
        feature2 (str): The name of the second numerical feature/column to be analyzed.

        Returns:
        None: Displays comprehensive bivariate analysis of the two numerical features.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Bivariate Analysis: {feature1} vs {feature2}', fontsize=16, fontweight='bold')
        
        # 1. Scatter plot
        sns.scatterplot(x=feature1, y=feature2, data=df, ax=axes[0, 0], alpha=0.6)
        axes[0, 0].set_title(f'Scatter Plot: {feature1} vs {feature2}')
        axes[0, 0].set_xlabel(feature1)
        axes[0, 0].set_ylabel(feature2)
        
        # 2. Scatter plot colored by attack category (if label exists)
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
            
            # Sample data if too large for better visualization
            if len(df_temp) > 5000:
                df_sample = df_temp.sample(5000)
            else:
                df_sample = df_temp
            
            sns.scatterplot(x=feature1, y=feature2, hue='attack_category', 
                           data=df_sample, ax=axes[0, 1], alpha=0.7)
            axes[0, 1].set_title(f'{feature1} vs {feature2} by Attack Category')
            axes[0, 1].set_xlabel(feature1)
            axes[0, 1].set_ylabel(feature2)
        
        # 3. Hexbin plot for density (useful for large datasets)
        axes[1, 0].hexbin(df[feature1], df[feature2], gridsize=30, cmap='Blues')
        axes[1, 0].set_title(f'Density Plot: {feature1} vs {feature2}')
        axes[1, 0].set_xlabel(feature1)
        axes[1, 0].set_ylabel(feature2)
        
        # 4. Correlation and statistics
        axes[1, 1].axis('off')
        
        correlation = df[feature1].corr(df[feature2])
        
        stats_text = f"""
        Correlation Analysis:
        
        Correlation Coefficient: {correlation:.4f}
        
        Interpretation:
        """
        
        if abs(correlation) >= 0.7:
            stats_text += f"\n        Strong {'positive' if correlation > 0 else 'negative'} correlation"
        elif abs(correlation) >= 0.3:
            stats_text += f"\n        Moderate {'positive' if correlation > 0 else 'negative'} correlation"
        else:
            stats_text += "\n        Weak correlation"
        
        stats_text += f"""
        
        {feature1} Statistics:
        Mean: {df[feature1].mean():.4f}
        Std: {df[feature1].std():.4f}
        Range: {df[feature1].min():.4f} to {df[feature1].max():.4f}
        
        {feature2} Statistics:
        Mean: {df[feature2].mean():.4f}
        Std: {df[feature2].std():.4f}
        Range: {df[feature2].min():.4f} to {df[feature2].max():.4f}
        
        Outliers Detection:
        {feature1} outliers: {((df[feature1] < df[feature1].quantile(0.25) - 1.5 * (df[feature1].quantile(0.75) - df[feature1].quantile(0.25))) | (df[feature1] > df[feature1].quantile(0.75) + 1.5 * (df[feature1].quantile(0.75) - df[feature1].quantile(0.25)))).sum():,}
        {feature2} outliers: {((df[feature2] < df[feature2].quantile(0.25) - 1.5 * (df[feature2].quantile(0.75) - df[feature2].quantile(0.25))) | (df[feature2] > df[feature2].quantile(0.75) + 1.5 * (df[feature2].quantile(0.75) - df[feature2].quantile(0.25)))).sum():,}
        """
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
        
        plt.tight_layout()
        plt.show()


# Concrete Strategy for Categorical vs Numerical Analysis in KDD99
# -----------------------------------------------------------------
# This strategy analyzes the relationship between a categorical feature and a numerical feature with KDD99-specific insights.
class KDD99CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between a categorical feature and a numerical feature with KDD99-specific analysis.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the categorical feature/column to be analyzed.
        feature2 (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays comprehensive bivariate analysis of the categorical and numerical features.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Categorical vs Numerical Analysis: {feature1} vs {feature2}', fontsize=16, fontweight='bold')
        
        # 1. Box plot
        unique_categories = df[feature1].nunique()
        
        if unique_categories <= 10:
            sns.boxplot(x=feature1, y=feature2, data=df, ax=axes[0, 0])
            axes[0, 0].tick_params(axis='x', rotation=45)
        else:
            # Show only top categories
            top_categories = df[feature1].value_counts().head(10).index
            df_filtered = df[df[feature1].isin(top_categories)]
            sns.boxplot(x=feature1, y=feature2, data=df_filtered, ax=axes[0, 0])
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].set_title(f'Box Plot: {feature1} vs {feature2} (Top 10 Categories)')
        
        axes[0, 0].set_title(f'Box Plot: {feature1} vs {feature2}')
        axes[0, 0].set_xlabel(feature1)
        axes[0, 0].set_ylabel(feature2)
        
        # 2. Violin plot
        if unique_categories <= 8:
            sns.violinplot(x=feature1, y=feature2, data=df, ax=axes[0, 1])
            axes[0, 1].tick_params(axis='x', rotation=45)
        else:
            top_categories = df[feature1].value_counts().head(8).index
            df_filtered = df[df[feature1].isin(top_categories)]
            sns.violinplot(x=feature1, y=feature2, data=df_filtered, ax=axes[0, 1])
            axes[0, 1].tick_params(axis='x', rotation=45)
            
        axes[0, 1].set_title(f'Violin Plot: {feature1} vs {feature2}')
        axes[0, 1].set_xlabel(feature1)
        axes[0, 1].set_ylabel(feature2)
        
        # 3. Mean values by category
        mean_by_category = df.groupby(feature1)[feature2].agg(['mean', 'count']).sort_values('mean', ascending=True)
        
        if len(mean_by_category) <= 15:
            mean_by_category['mean'].plot(kind='barh', ax=axes[1, 0])
        else:
            mean_by_category['mean'].head(15).plot(kind='barh', ax=axes[1, 0])
            
        axes[1, 0].set_title(f'Mean {feature2} by {feature1}')
        axes[1, 0].set_xlabel(f'Mean {feature2}')
        axes[1, 0].set_ylabel(feature1)
        
        # 4. Statistical summary
        axes[1, 1].axis('off')
        
        # Calculate statistics by category
        stats_by_category = df.groupby(feature1)[feature2].agg(['count', 'mean', 'std', 'min', 'max'])
        overall_mean = df[feature2].mean()
        
        stats_text = f"""
        Statistical Summary by {feature1}:
        
        Overall {feature2} Mean: {overall_mean:.4f}
        Number of Categories: {unique_categories}
        
        Top 5 Categories by Mean {feature2}:
        """
        
        top_5_by_mean = stats_by_category.sort_values('mean', ascending=False).head(5)
        
        for i, (category, stats) in enumerate(top_5_by_mean.iterrows(), 1):
            stats_text += f"\n        {i}. {category}: {stats['mean']:.4f} (n={stats['count']:,})"
        
        stats_text += f"""
        
        Bottom 5 Categories by Mean {feature2}:
        """
        
        bottom_5_by_mean = stats_by_category.sort_values('mean', ascending=True).head(5)
        
        for i, (category, stats) in enumerate(bottom_5_by_mean.iterrows(), 1):
            stats_text += f"\n        {i}. {category}: {stats['mean']:.4f} (n={stats['count']:,})"
        
        # ANOVA-like analysis
        between_group_var = ((stats_by_category['mean'] - overall_mean) ** 2 * stats_by_category['count']).sum() / (len(stats_by_category) - 1)
        within_group_var = ((stats_by_category['std'] ** 2) * (stats_by_category['count'] - 1)).sum() / (len(df) - len(stats_by_category))
        
        if within_group_var > 0:
            f_ratio = between_group_var / within_group_var
            stats_text += f"\n\n        F-ratio (between/within variance): {f_ratio:.4f}"
            
            if f_ratio > 4:
                stats_text += "\n        Strong evidence of group differences"
            elif f_ratio > 2:
                stats_text += "\n        Moderate evidence of group differences"
            else:
                stats_text += "\n        Weak evidence of group differences"
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=8, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        plt.tight_layout()
        plt.show()


# Standard Numerical vs Numerical Analysis
# -----------------------------------------
class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between two numerical features using a scatter plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first numerical feature/column to be analyzed.
        feature2 (str): The name of the second numerical feature/column to be analyzed.

        Returns:
        None: Displays a scatter plot showing the relationship between the two features.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()


# Standard Categorical vs Numerical Analysis
# -------------------------------------------
class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between a categorical feature and a numerical feature using a box plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the categorical feature/column to be analyzed.
        feature2 (str): The name of the second numerical feature/column to be analyzed.

        Returns:
        None: Displays a box plot showing the relationship between the categorical and numerical features.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()


# Context Class that uses a BivariateAnalysisStrategy
# ---------------------------------------------------
# This class allows you to switch between different bivariate analysis strategies.
class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        """
        Initializes the BivariateAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (BivariateAnalysisStrategy): The strategy to be used for bivariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        """
        Sets a new strategy for the BivariateAnalyzer.

        Parameters:
        strategy (BivariateAnalysisStrategy): The new strategy to be used for bivariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Executes the bivariate analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first feature/column to be analyzed.
        feature2 (str): The name of the second feature/column to be analyzed.

        Returns:
        None: Executes the strategy's analysis method and visualizes the results.
        """
        self._strategy.analyze(df, feature1, feature2)


# Example usage
if __name__ == "__main__":
    # Example usage of the BivariateAnalyzer with different strategies.
    pass