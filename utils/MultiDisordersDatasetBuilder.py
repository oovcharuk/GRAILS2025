import pandas as pd

class MultiDisordersDatasetBuilder:
    def __init__(self, target_label, non_target_labels, output_path):
        self.train_path = 'datasets/multi_disorders/train.csv'
        self.depression_path = 'datasets/multi_disorders/depression_dataset_reddit_cleaned.csv'
        self.output_path = output_path
        self.target_label = target_label
        self.non_target_labels = non_target_labels
        self.sample_ratio = 0.2

    def load_data(self):
        self.train_data = pd.read_csv(self.train_path)
        self.depression_data = pd.read_csv(self.depression_path)

    def filter_and_sample(self):
        target_data = self.train_data[self.train_data['label'] == self.target_label]
        self.target_count = len(target_data)
        sample_size = self.target_count // (len(self.non_target_labels) + 1)

        non_target_samples = []
        for label in self.non_target_labels:
            class_data = self.train_data[self.train_data['label'] == label]
            sample = class_data.sample(n=sample_size, random_state=42).copy()
            sample['label'] = 'non-target'
            non_target_samples.append(sample)

        non_depression_data = self.depression_data[self.depression_data['is_depression'] == 0]
        non_depression_sample = non_depression_data.sample(n=sample_size, random_state=42)
        non_depression_sample = non_depression_sample[['clean_text']].copy()
        non_depression_sample['label'] = 'non-target'
        non_depression_sample = non_depression_sample.rename(columns={'clean_text': 'Text'})

        target_data = target_data[['Text', 'label']]
        self.final_dataset = pd.concat([target_data] + non_target_samples + [non_depression_sample])

    def shuffle_and_save(self):
        self.final_dataset = self.final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        if 'Unnamed: 0' in self.final_dataset.columns:
            self.final_dataset = self.final_dataset.drop(columns=['Unnamed: 0'])
        self.final_dataset.to_csv(self.output_path, index=False)

    def run(self):
        self.load_data()
        self.filter_and_sample()
        self.shuffle_and_save()
        print(f"Final dataset size: {self.final_dataset.shape}")
        print(self.final_dataset['label'].value_counts())