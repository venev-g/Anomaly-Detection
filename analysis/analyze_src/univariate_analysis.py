from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Abstract Base Class for Univariate Analysis Strategy
# -----------------------------------------------------
# This class defines a common interface for univariate analysis strategies.
# Subclasses must implement the analyze method.
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform univariate analysis on a specific feature of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed.

        Returns:
        None: This method visualizes the distribution of the feature.
        """
        pass


# Concrete Strategy for Numerical Features in KDD99
# --------------------------------------------------
# This strategy analyzes numerical features by plotting their distribution with KDD99-specific insights.
class KDD99NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a numerical feature with KDD99-specific analysis.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays comprehensive analysis of the numerical feature.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Univariate Analysis: {feature}', fontsize=16, fontweight='bold')
        
        # 1. Distribution plot (histogram + KDE)
        sns.histplot(df[feature], kde=True, bins=50, ax=axes[0, 0])
        axes[0, 0].set_title(f'Distribution of {feature}')
        axes[0, 0].set_xlabel(feature)
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Box plot to show outliers
        sns.boxplot(y=df[feature], ax=axes[0, 1])
        axes[0, 1].set_title(f'Box Plot: {feature}')
        axes[0, 1].set_ylabel(feature)
        
        # 3. Distribution by attack category (if label exists)
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
            
            # Filter out any unmapped categories
            df_temp = df_temp.dropna(subset=['attack_category'])
            
            sns.boxplot(data=df_temp, x='attack_category', y=feature, ax=axes[1, 0])
            axes[1, 0].set_title(f'{feature} by Attack Category')
            axes[1, 0].set_xlabel('Attack Category')
            axes[1, 0].set_ylabel(feature)
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Statistical summary
        axes[1, 1].axis('off')
        stats_text = f"""
        Statistical Summary for {feature}:
        
        Count: {df[feature].count():,}
        Mean: {df[feature].mean():.4f}
        Median: {df[feature].median():.4f}
        Std Dev: {df[feature].std():.4f}
        Min: {df[feature].min():.4f}
        Max: {df[feature].max():.4f}
        
        Quartiles:
        25%: {df[feature].quantile(0.25):.4f}
        50%: {df[feature].quantile(0.50):.4f}
        75%: {df[feature].quantile(0.75):.4f}
        
        Skewness: {df[feature].skew():.4f}
        Kurtosis: {df[feature].kurtosis():.4f}
        
        Zero values: {(df[feature] == 0).sum():,}
        Unique values: {df[feature].nunique():,}
        """
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.show()


# Concrete Strategy for Categorical Features in KDD99
# ----------------------------------------------------
# This strategy analyzes categorical features with KDD99-specific insights.
class KDD99CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a categorical feature with KDD99-specific analysis.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the categorical feature/column to be analyzed.

        Returns:
        None: Displays comprehensive analysis of the categorical feature.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Categorical Analysis: {feature}', fontsize=16, fontweight='bold')
        
        # 1. Count plot
        value_counts = df[feature].value_counts()
        
        # Limit to top categories if too many
        if len(value_counts) > 10:
            top_categories = value_counts.head(10)
            sns.barplot(x=top_categories.values, y=top_categories.index, ax=axes[0, 0])
            axes[0, 0].set_title(f'Top 10 Categories: {feature}')
        else:
            sns.countplot(data=df, y=feature, order=value_counts.index, ax=axes[0, 0])
            axes[0, 0].set_title(f'Distribution of {feature}')
        
        axes[0, 0].set_xlabel('Count')
        axes[0, 0].set_ylabel(feature)
        
        # 2. Pie chart for proportions
        if len(value_counts) <= 8:
            axes[0, 1].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
            axes[0, 1].set_title(f'Proportion of {feature}')
        else:
            # Show top categories and group others
            top_n = 7
            top_categories = value_counts.head(top_n)
            others_count = value_counts.iloc[top_n:].sum()
            
            if others_count > 0:
                pie_data = list(top_categories.values) + [others_count]
                pie_labels = list(top_categories.index) + ['Others']
            else:
                pie_data = top_categories.values
                pie_labels = top_categories.index
                
            axes[0, 1].pie(pie_data, labels=pie_labels, autopct='%1.1f%%')
            axes[0, 1].set_title(f'Proportion of {feature} (Top {top_n} + Others)')
        
        # 3. Cross-tabulation with label (if exists)
        if 'label' in df.columns and feature != 'label':
            # Create attack categories for better visualization
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
            
            # Create cross-tabulation
            crosstab = pd.crosstab(df_temp[feature], df_temp['attack_category'])
            
            # Limit categories if too many
            if len(crosstab) > 10:
                crosstab = crosstab.head(10)
            
            sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
            axes[1, 0].set_title(f'{feature} vs Attack Category')
            axes[1, 0].set_xlabel('Attack Category')
            axes[1, 0].set_ylabel(feature)
        
        # 4. Statistical summary
        axes[1, 1].axis('off')
        
        unique_values = df[feature].nunique()
        mode_value = df[feature].mode().iloc[0] if len(df[feature].mode()) > 0 else 'N/A'
        mode_count = df[feature].value_counts().iloc[0]
        mode_percent = (mode_count / len(df)) * 100
        
        stats_text = f"""
        Statistical Summary for {feature}:
        
        Total Count: {len(df):,}
        Unique Values: {unique_values:,}
        Most Frequent: {mode_value}
        Frequency: {mode_count:,} ({mode_percent:.2f}%)
        
        Top 5 Categories:
        """
        
        top_5 = df[feature].value_counts().head(5)
        for i, (category, count) in enumerate(top_5.items(), 1):
            percent = (count / len(df)) * 100
            stats_text += f"\n        {i}. {category}: {count:,} ({percent:.2f}%)"
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.tight_layout()
        plt.show()


# Standard Numerical Univariate Analysis
# ---------------------------------------
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a numerical feature using a histogram and KDE.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays a histogram with a KDE plot.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()


# Standard Categorical Univariate Analysis
# -----------------------------------------
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a categorical feature using a bar plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the categorical feature/column to be analyzed.

        Returns:
        None: Displays a bar plot showing the frequency of each category.
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, data=df, palette="muted")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()


# Context Class that uses a UnivariateAnalysisStrategy
# ----------------------------------------------------
# This class allows you to switch between different univariate analysis strategies.
class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        """
        Initializes the UnivariateAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The strategy to be used for univariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        """
        Sets a new strategy for the UnivariateAnalyzer.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The new strategy to be used for univariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        """
        Executes the univariate analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed.

        Returns:
        None: Executes the strategy's analysis method and visualizes the results.
        """
        self._strategy.analyze(df, feature)


# Example usage
if __name__ == "__main__":
    # Example usage of the UnivariateAnalyzer with different strategies.
    pass