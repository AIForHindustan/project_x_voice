from src.data_processor import BilingualDataset
import pandas as pd
from phonemizer import phonemize
from epitran import Epitran

def validate_dataset():
    # Load metadata from the data folder
    metadata = pd.read_csv("data/metadata.csv")
    
    # Iterate through each row and print debugging info
    for idx, row in metadata.iterrows():
        print(f"Processing row {idx}")
        try:
            text = str(row['text'])
            lang = row['language'].lower()
            
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
                
            print(f"Original ({lang}): {text}")
            print(f"Phonemes: {phonemes}\n")
            
        except Exception as e:
            print(f"Error in row {idx}: {str(e)}")
            continue

if __name__ == "__main__":
    validate_dataset()
