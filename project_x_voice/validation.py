import pandas as pd
from phonemizer import phonemize
from epitran import Epitran
import logging

def validate_emotion_mapping():
    # Load metadata from the data folder
    metadata = pd.read_csv("data/metadata.csv")
    
    # Check emotion distribution
    emotion_counts = metadata["emotion"].value_counts()
    print("\nEmotion Distribution:")
    print(emotion_counts)
    
    # Verify that emotion IDs are within the valid range
    try:
        assert metadata["emotion_id"].between(0, 5).all(), "Emotion IDs are out of range! Must be between 0 and 5."
        print("Emotion IDs are valid.\n")
    except AssertionError as e:
        print(str(e))
    
def validate_dataset():
    # Load metadata from the data folder
    metadata = pd.read_csv("data/metadata.csv")
    
    # Iterate through each row and print debugging info
    for idx, row in metadata.iterrows():
        print(f"Processing row {idx}")
        try:
            text = str(row['text'])
            lang = row['language'].lower()
            
            # Validate phonemes based on language
            if lang == "en":
                phonemes = phonemize(
                    text,
                    language='en-us',
                    backend='espeak',
                    strip=True
                )
            elif lang == "hi":
                phonemes = Epitran('hin-Deva').transliterate(text)
            else:
                phonemes = "Unsupported language"
            
            # Log and print the processed data
            print(f"Original ({lang}): {text}")
            print(f"Phonemes: {phonemes}")
            print(f"Emotion ID: {row['emotion_id']}\n")
            
        except Exception as e:
            print(f"Error in row {idx}: {str(e)}")
            continue

if __name__ == "__main__":
    # Run emotion validation before dataset validation
    validate_emotion_mapping()
    
    # Run phoneme validation
    validate_dataset()
