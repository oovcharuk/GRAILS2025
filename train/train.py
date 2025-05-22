import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.PTSDTextClassifier import PTSDTextClassifier
from model.MultiDisordersTextClassifier import MultiDisordersTextClassifier
from utils.PTSDDatasetBuilder import PTSDDatasetBuilder
from utils.MultiDisordersDatasetBuilder import MultiDisordersDatasetBuilder

def main():

    # PTSD Dataset generation
    PTSDbuilder = PTSDDatasetBuilder()
    PTSDbuilder.build()

    # Training PTSD model 
    PTSDclassifier = PTSDTextClassifier()
    PTSDclassifier.load_data("datasets/PTSD/combined_dataset_all.csv")
    PTSDclassifier.train()
    
    # Building 5 datasets with multiple disorders 
    labels = ['Anxiety Disorder', 'Depression', 'Panic Disorder', 'Anger/ Intermittent Explosive Disorder']
    builder = MultiDisordersDatasetBuilder('Narcissistic Disorder', labels,'datasets/multi_disorders/NarcissisticDisorder.csv')
    builder.run()

    labels = ['Narcissistic Disorder', 'Depression', 'Panic Disorder', 'Anger/ Intermittent Explosive Disorder']
    builder = MultiDisordersDatasetBuilder('Anxiety Disorder', labels,'datasets/multi_disorders/AnxietyDisorder.csv')
    builder.run()

    labels = ['Narcissistic Disorder', 'Anxiety Disorder', 'Panic Disorder', 'Anger/ Intermittent Explosive Disorder']
    builder = MultiDisordersDatasetBuilder('Depression', labels,'datasets/multi_disorders/Depression.csv')
    builder.run()

    labels = ['Narcissistic Disorder', 'Anxiety Disorder', 'Depression', 'Anger/ Intermittent Explosive Disorder']
    builder = MultiDisordersDatasetBuilder('Panic Disorder', labels,'datasets/multi_disorders/PanicDisorder.csv')
    builder.run()

    labels = ['Narcissistic Disorder', 'Anxiety Disorder', 'Depression', 'Panic Disorder']
    builder = MultiDisordersDatasetBuilder('Anger/ Intermittent Explosive Disorder', labels,'datasets/multi_disorders/AngerIntermittentExplosiveDisorder.csv')
    builder.run()

    # Training 5 models: “Narcissistic Disorder, “Anxiety Disorder, “Panic Disorder, “Anger / Intermittent Explosive Disorder and “Depression” 
    classifier = MultiDisordersTextClassifier(
        dataset_path='datasets/multi_disorders/NarcissisticDisorder.csv',
        output_dir='./trained_models/NarcissisticDisorder',
        target_label='Narcissistic Disorder'
    )
    classifier.run()

    classifier = MultiDisordersTextClassifier(
        dataset_path='datasets/multi_disorders/AnxietyDisorder.csv',
        output_dir='./trained_models/AnxietyDisorder',
        target_label='Anxiety Disorder'
    )
    classifier.run()

    classifier = MultiDisordersTextClassifier(
        dataset_path='datasets/multi_disorders/Depression.csv',
        output_dir='./trained_models/Depression',
        target_label='Depression'
    )
    classifier.run()

    classifier = MultiDisordersTextClassifier(
        dataset_path='datasets/multi_disorders/PanicDisorder.csv',
        output_dir='./trained_models/PanicDisorder',
        target_label='Panic Disorder'
    )
    classifier.run()

    classifier = MultiDisordersTextClassifier(
        dataset_path='datasets/multi_disorders/AngerIntermittentExplosiveDisorder.csv',
        output_dir='./trained_models/AngerIntermittentExplosiveDisorder',
        target_label='Anger Intermittent Explosive Disorder'
    )
    classifier.run()

if __name__ == "__main__":
    main()