from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Abstract Base Class for Missing Values Analysis
# -----------------------------------------------
# This class defines a template for missing values analysis.
# Subclasses must implement the methods to identify and visualize missing values.
class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Performs a complete missing values analysis by identifying and visualizing missing values.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: This method performs the analysis and visualizes missing values.
        """
        self.identify_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Identifies missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: This method should print the count of missing values for each column.
        """
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Visualizes missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be visualized.

        Returns:
        None: This method should create a visualization (e.g., heatmap) of missing values.
        """
        pass


# Concrete Class for Missing Values Identification for KDD99
# -----------------------------------------------------------
# This class implements methods to identify and visualize missing values in the KDD99 dataset.
class KDD99MissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Prints the count of missing values for each column in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: Prints the missing values count to the console.
        """
        print("\nMissing Values Analysis for KDD99 Dataset:")
        print("=" * 50)
        
        missing_values = df.isnull().sum()
        total_cells = len(df)
        
        print(f"Total number of samples: {total_cells}")
        print(f"Total number of features: {len(df.columns)}")
        
        if missing_values.sum() == 0:
            print("\n✅ Great! No missing values found in the dataset.")
        else:
            print("\nMissing Values Count by Column:")
            missing_data = missing_values[missing_values > 0]
            print(missing_data)
            
            print("\nMissing Values Percentage by Column:")
            missing_percentage = (missing_data / total_cells) * 100
            print(missing_percentage.round(2))

    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Creates a heatmap to visualize the missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be visualized.

        Returns:
        None: Displays a heatmap of missing values.
        """
        missing_values = df.isnull().sum()
        
        if missing_values.sum() == 0:
            print("\n📊 No missing values visualization needed - dataset is complete!")
            
            # Create a simple completion status visualization
            plt.figure(figsize=(10, 2))
            plt.text(0.5, 0.5, '✅ KDD99 Dataset: 100% Complete\nNo Missing Values Found', 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=16, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
            plt.title("Data Completeness Status", fontsize=18, fontweight='bold')
            plt.tight_layout()
            plt.show()
        else:
            print("\nVisualizing Missing Values...")
            plt.figure(figsize=(20, 10))
            
            # Create a more detailed heatmap for missing values
            missing_matrix = df.isnull()
            
            if len(df.columns) > 20:
                # For many columns, sample some rows for better visualization
                sample_size = min(1000, len(df))
                sample_indices = df.sample(sample_size).index
                missing_matrix_sample = missing_matrix.loc[sample_indices]
                
                sns.heatmap(missing_matrix_sample.T, cbar=True, cmap="viridis", 
                           yticklabels=True, xticklabels=False)
                plt.title(f"Missing Values Heatmap (Sample of {sample_size} rows)", fontsize=14)
            else:
                sns.heatmap(missing_matrix.T, cbar=True, cmap="viridis", 
                           yticklabels=True, xticklabels=False)
                plt.title("Missing Values Heatmap", fontsize=14)
                
            plt.xlabel("Samples")
            plt.ylabel("Features")
            plt.tight_layout()
            plt.show()


# Simple Missing Values Analysis (for general use)
# ------------------------------------------------
class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Prints the count of missing values for each column in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: Prints the missing values count to the console.
        """
        print("\nMissing Values Count by Column:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])

    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Creates a heatmap to visualize the missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be visualized.

        Returns:
        None: Displays a heatmap of missing values.
        """
        print("\nVisualizing Missing Values...")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()


# Example usage
if __name__ == "__main__":
    # Example usage of the KDD99MissingValuesAnalysis class.
    pass