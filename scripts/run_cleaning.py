import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from src.data_loader import load_transcript, save_transcript
from src.cleaning.normalizer import NameNormalizer

def main():
    # 1. Konfigurasjon
    input_file = Path("data/raw/2_asr.json")
    output_file = Path("data/processed/2_asr_cleaned.json")
    entities_file = Path("data/external/players_2015.json")

    # 2. Sjekk at filer finnes
    if not input_file.exists():
        print(f"ERROR: Fant ikke input-filen: {input_file}")
        return
    if not entities_file.exists():
        print(f"ERROR: Fant ikke database-filen: {entities_file}")
        return

    # 3. Initialiser verktøy
    print("Initialiserer cleaning pipeline...")
    normalizer = NameNormalizer(entities_file)
    
    # 4. Last data
    print(f"Laster transkripsjon fra {input_file}...")
    segments = load_transcript(input_file)
    
    # 5. Kjør rensing
    print("Kjører navn-normalisering...")
    cleaned_count = 0
    for segment in segments:
        original_text = segment["text"]
        cleaned_text = normalizer.normalize_sentence(original_text)
        
        if original_text != cleaned_text:
            cleaned_count += 1
            # Print noen eksempler underveis
            if cleaned_count <= 5: 
                print(f"Endring: '{original_text}' \n   ->    '{cleaned_text}'")
        
        segment["text"] = cleaned_text

    # 6. Lagre resultat
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_transcript(segments, output_file)
    print(f"\nFerdig! {cleaned_count} linjer ble endret.")
    print(f"Resultat lagret til: {output_file}")

if __name__ == "__main__":
    main()