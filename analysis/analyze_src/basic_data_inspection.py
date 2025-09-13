from abc import ABC, abstractmethod
import pandas as pd


# Abstract Base Class for Data Inspection Strategies
# --------------------------------------------------
# This class defines a common interface for data inspection strategies.
# Subclasses must implement the inspect method.
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Perform a specific type of data inspection.

        Parameters:
        df (pd.DataFrame): The dataframe on which the inspection is to be performed.

        Returns:
        None: This method prints the inspection results directly.
        """
        pass


# Concrete Strategy for Data Types Inspection
# --------------------------------------------
# This strategy inspects the data types of each column and counts non-null values.
class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the data types and non-null counts of the dataframe columns.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the data types and non-null counts to the console.
        """
        print("\nData Types and Non-null Counts:")
        print(df.info())


# Concrete Strategy for Summary Statistics Inspection
# -----------------------------------------------------
# This strategy provides summary statistics for both numerical and categorical features.
class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Prints summary statistics for numerical and categorical features.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints summary statistics to the console.
        """
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())
        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=["O"]))


# KDD99-specific Data Inspection Strategy
# ----------------------------------------
# This strategy provides KDD99-specific data inspection including attack types analysis.
class KDD99InspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Performs KDD99-specific data inspection including attack type analysis.

        Parameters:
        df (pd.DataFrame): The KDD99 dataframe to be inspected.

        Returns:
        None: Prints KDD99-specific analysis to the console.
        """
        print("\nKDD99 Dataset Analysis:")
        print(f"Dataset shape: {df.shape}")
        print(f"Total samples: {len(df)}")
        
        # Label distribution analysis
        if 'label' in df.columns:
            print("\nLabel Distribution:")
            label_counts = df['label'].value_counts()
            print(label_counts)
            
            print("\nLabel Percentages:")
            label_percentages = df['label'].value_counts(normalize=True) * 100
            print(label_percentages.round(2))
            
            # Attack type categorization
            attack_types = {
                'normal': ['normal.'],
                'dos': ['back.', 'land.', 'neptune.', 'pod.', 'smurf.', 'teardrop.'],
                'probe': ['ipsweep.', 'nmap.', 'portsweep.', 'satan.'],
                'r2l': ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'phf.', 'spy.', 'warezclient.', 'warezmaster.'],
                'u2r': ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.']
            }
            
            # Create attack category mapping
            attack_category = {}
            for category, attacks in attack_types.items():
                for attack in attacks:
                    attack_category[attack] = category
            
            # Map labels to categories
            df_temp = df.copy()
            df_temp['attack_category'] = df_temp['label'].map(attack_category)
            
            print("\nAttack Category Distribution:")
            category_counts = df_temp['attack_category'].value_counts()
            print(category_counts)
            
            print("\nAttack Category Percentages:")
            category_percentages = df_temp['attack_category'].value_counts(normalize=True) * 100
            print(category_percentages.round(2))


# Context Class that uses a DataInspectionStrategy
# ------------------------------------------------
# This class allows you to switch between different data inspection strategies.
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """
        Initializes the DataInspector with a specific inspection strategy.

        Parameters:
        strategy (DataInspectionStrategy): The strategy to be used for data inspection.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        """
        Sets a new strategy for the DataInspector.

        Parameters:
        strategy (DataInspectionStrategy): The new strategy to be used for data inspection.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        """
        Executes the inspection using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Executes the strategy's inspection method.
        """
        self._strategy.inspect(df)


# Example usage
if __name__ == "__main__":
    # Example usage of the DataInspector with different strategies.
    pass