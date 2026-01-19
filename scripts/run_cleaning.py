import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from src.data_loader import load_transcript, save_transcript
from src.cleaning.normalizer import NameNormalizer, ImprovedNameNormalizer
from src.cleaning.ner_extractor import NERExtractor

def main():
    # 1. Configuration
    input_file = Path("data/raw/2_asr.json")
    output_file = Path("data/processed/2_asr_cleaned.json")
    
    # Use the new structured gazetteer (with misspellings and aliases)
    entities_file = Path("data/external/entities_2015.json")
    
    # Fallback to old simple list if new one doesn't exist
    if not entities_file.exists():
        entities_file = Path("data/external/players_2015.json")
        print("Warning: Using legacy players file. Consider using entities_2015.json")

    # 2. Check files exist
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return
    if not entities_file.exists():
        print(f"ERROR: Database file not found: {entities_file}")
        return

    # 3. Initialize tools
    print("Initializing cleaning pipeline...")
    normalizer = NameNormalizer(str(entities_file))
    
    # 4. Load data
    print(f"Loading transcript from {input_file}...")
    segments = load_transcript(input_file)
    
    # 5. Run cleaning
    print("\nRunning name normalization...")
    print("="*60)
    cleaned_count = 0
    
    for i, segment in enumerate(segments):
        original_text = segment["text"]
        
        # Use the improved normalizer
        cleaned_text = normalizer.normalize_sentence(original_text, debug=False)
        
        if original_text != cleaned_text:
            cleaned_count += 1
            # Print examples of changes
            print(f"\n[Segment {i}] Change detected:")
            print(f"  Original: '{original_text}'")
            print(f"  Cleaned:  '{cleaned_text}'")
        
        segment["text"] = cleaned_text

    # 6. Save result
    print("\n" + "="*60)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_transcript(segments, output_file)
    print(f"\nDone! {cleaned_count} lines were modified.")
    print(f"Result saved to: {output_file}")


def test_normalizer():
    """Test the normalizer with example sentences to verify it works correctly."""
    # Use the new structured gazetteer
    entities_file = Path("data/external/entities_2015.json")
    
    if not entities_file.exists():
        # Fallback to old file
        entities_file = Path("data/external/players_2015.json")
    
    if not entities_file.exists():
        print(f"ERROR: Database file not found")
        return
    
    print("Testing Improved Normalizer...")
    print("="*60)
    
    normalizer = NameNormalizer(str(entities_file))
    
    # Test cases - including problematic ones
    test_sentences = [
        # Should NOT change (common words)
        ("There is a lot of excitement in the stadium.", False),
        ("They will play well today.", False),
        ("The stuff that happens in football is unpredictable.", False),
        ("Which team will win this match?", False),
        
        # SHOULD change (actual player/entity references with misspellings)
        ("Valdez makes a great save.", True),  # Valdez -> Valdes
        ("Wilser is back from injury.", True),  # Wilser -> Wilshere
        ("Great save by De Gea at Old Trafford!", False),  # Already correct
        ("Felaini scores a header!", True),  # Felaini -> Fellaini
        ("Chris Mullin clears the ball.", True),  # Mullin -> Smalling
        
        # Edge cases with punctuation
        ("Hasley Young runs down the wing.", True),  # Hasley -> Ashley
        ("Daley Blin makes a tackle.", True),  # Blin -> Blind
        ("Valdez, with a great save!", True),  # Punctuation handling
        ("It was Ozil's pass.", False),  # Possessive
        ("(Giroud) scores again.", False),  # Parentheses - already correct
        ("Girud scores!", True),  # Misspelling with punctuation
    ]
    
    print("\nTest Results:")
    print("-"*60)
    
    correct = 0
    total = len(test_sentences)
    
    for sentence, should_change in test_sentences:
        cleaned = normalizer.normalize_sentence(sentence, debug=True)
        did_change = sentence != cleaned
        
        status = "✓" if did_change == should_change else "✗"
        if did_change == should_change:
            correct += 1
        
        expected = "SHOULD CHANGE" if should_change else "NO CHANGE EXPECTED"
        actual = "CHANGED" if did_change else "NO CHANGE"
        
        print(f"\n{status} [{expected} -> {actual}]")
        print(f"  Input:  {sentence}")
        print(f"  Output: {cleaned}")
    
    print(f"\n{'='*60}")
    print(f"Results: {correct}/{total} tests passed")


def analyze_entities():
    """Analyze what entities exist in the transcript."""
    input_file = Path("data/raw/2_asr.json")
    
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return
    
    print("Analyzing entities in transcript...")
    print("="*60)
    
    extractor = NERExtractor()
    segments = load_transcript(input_file)
    
    # Analyze
    analysis = extractor.analyze_transcript(segments)
    
    print(f"\nTotal entities found: {analysis['total_entities']}")
    print(f"Unique entities: {analysis['unique_entities']}")
    
    print(f"\nEntity types distribution:")
    for label, count in analysis['label_counts'].items():
        print(f"  {label}: {count}")
    
    print(f"\nTop 20 most frequent entities:")
    for entity, count in analysis['entity_counts'].most_common(20):
        print(f"  {entity}: {count}")
    
    print(f"\nSample segments with entities:")
    for seg in analysis['segments_with_entities'][:10]:
        print(f"\n  [{seg['segment_index']}] {seg['text']}")
        print(f"      Entities: {[(e['text'], e['label']) for e in seg['entities']]}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean ASR transcripts")
    parser.add_argument("--test", action="store_true", help="Run normalizer tests")
    parser.add_argument("--analyze", action="store_true", help="Analyze entities in transcript")
    
    args = parser.parse_args()
    
    if args.test:
        test_normalizer()
    elif args.analyze:
        analyze_entities()
    else:
        main()

if __name__ == "__main__":
    main()