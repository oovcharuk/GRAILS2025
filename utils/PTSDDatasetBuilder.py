import pandas as pd
import os

class PTSDDatasetBuilder:
    def __init__(self):
        self.base_path = 'datasets/PTSD/human_stress_prediction_data_set.csv'
        self.additional_path = 'datasets/PTSD/aya_ptsd_data_set.csv'
        self.output_dir = 'datasets/PTSD'
        self.existing_output = os.path.join(self.output_dir, 'new_dataset.csv')
        self.combined_output = os.path.join(self.output_dir, 'combined_dataset_all.csv')
        self.combined_dataset = None

    def process_base_dataset(self):
        data = pd.read_csv(self.base_path)
        print("Base dataset preview:")
        print(data.head())

        data['new_label'] = data['subreddit'].apply(lambda x: 1 if x == 'ptsd' else 0)

        new_dataset = data[['text', 'new_label']]
        new_dataset.to_csv(self.existing_output, index=False)
        print(f"New dataset saved as '{self.existing_output}'.")

    def process_additional_dataset(self):
        new_dataset = pd.read_csv(self.additional_path)

        filtered_new_data = new_dataset[new_dataset['label'] > 2].copy()
        filtered_new_data['new_label'] = 1
        filtered_new_data = filtered_new_data[['text', 'new_label']]
        return filtered_new_data

    def combine_datasets(self):
        existing_dataset = pd.read_csv(self.existing_output)
        print("Loaded existing dataset preview:")
        print(existing_dataset.head())

        additional_filtered = self.process_additional_dataset()

        self.combined_dataset = pd.concat([existing_dataset, additional_filtered], ignore_index=True)
        self.combined_dataset.to_csv(self.combined_output, index=False)
        print(f"Combined dataset saved as '{self.combined_output}'.")

    def describe_combined_dataset(self):
        if self.combined_dataset is None:
            self.combined_dataset = pd.read_csv(self.combined_output)

        print("\nDataset information:")
        print(self.combined_dataset.info())

        print("\nStatistics on numeric columns:")
        print(self.combined_dataset.describe())

        print("\nStatistics by categorical columns:")
        print(self.combined_dataset['new_label'].value_counts())
    
    def build(self):
        self.process_base_dataset()
        self.combine_datasets()
        self.describe_combined_dataset()

