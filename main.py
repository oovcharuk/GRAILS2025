from model.PTSDDebertaClassifier import PTSDDebertaClassifier
from utils.PTSDDatasetBuilder import PTSDDatasetBuilder

def main():
    builder = PTSDDatasetBuilder()
    builder.build()

    classifier = PTSDDebertaClassifier()

    # model training
    # classifier.load_data("content/combined_dataset_all.csv")
    # classifier.train()

    classifier.predict_with_attention("He said he had not felt that way before, suggeted I go rest and so i decide to look up feelings of doom in hopes of maybe getting sucked into some rabbit hole of ludicrous conspiracy, a stupid are you psychic test or new age b.s., something I could even laugh at down the road. No, I ended up reading that this sense of doom can be indicative of various health ailments; one of which I am prone to.. So on top of my doom to my gloom..I am now f'n worried about my heart. I do happen to have a physical in 48 hours")

if __name__ == "__main__":
    main()
