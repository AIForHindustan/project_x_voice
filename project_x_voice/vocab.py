import json
from collections import Counter
from src.data_processor import BilingualDataset

def main():
    # Load dataset
    dataset = BilingualDataset(
        metadata_path="data/metadata.csv",
        data_dir="data",
        config=json.load(open("config/config.json"))
    )

    counter = Counter()
    for i in range(len(dataset)):
        item = dataset[i]
        if item is not None:
            counter.update([chr(c) for c in item["phonemes"].tolist()])

    # Create vocabulary mapping
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        **{char: i + 2 for i, (char, _) in enumerate(counter.most_common())}
    }

    # Save phoneme mapping
    with open("config/phonemes.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

    print(f"✅ Created vocabulary with {len(vocab)} tokens and saved to config/phonemes.json")

if __name__ == "__main__":
    main()
