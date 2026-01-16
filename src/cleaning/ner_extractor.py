"""
Named Entity Recognition (NER) extractor for football transcripts.

This module provides tools to detect and extract named entities from ASR transcripts,
focusing on football-related entities like player names, team names, and venues.
"""

import spacy
from typing import List, Dict, Set, Optional
from collections import Counter


class NERExtractor:
    """
    Extract named entities from text using spaCy NER.
    
    Specialized for football commentary transcripts where entities include:
    - PERSON: Player names, managers, referees
    - ORG: Club names, football organizations
    - GPE/LOC: Cities, countries, stadiums
    """
    
    # Entity labels we care about for football transcripts
    RELEVANT_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "EVENT"}
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the NER extractor.
        
        Args:
            model_name: spaCy model to use. Options:
                - "en_core_web_sm": Small, fast (default)
                - "en_core_web_md": Medium, better accuracy
                - "en_core_web_lg": Large, best accuracy
                - "en_core_web_trf": Transformer-based, highest accuracy
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model '{model_name}' not found. Downloading...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
        
        print(f"NERExtractor initialized with model: {model_name}")
        self.model_name = model_name
    
    def extract_entities(self, text: str, labels: Optional[Set[str]] = None) -> List[Dict]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text to analyze
            labels: Set of entity labels to extract (default: all relevant labels)
        
        Returns:
            List of entity dictionaries with keys:
                - text: The entity text
                - label: Entity type (PERSON, ORG, etc.)
                - start: Start character position
                - end: End character position
        """
        if labels is None:
            labels = self.RELEVANT_LABELS
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in labels:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        
        return entities
    
    def extract_potential_names(self, text: str) -> List[Dict]:
        """
        Extract tokens that could be names based on heuristics.
        
        This catches names that spaCy NER might miss, especially in ASR output
        where capitalization might be inconsistent.
        
        Heuristics:
        - Capitalized words
        - Words followed by common name patterns
        - Consecutive capitalized words (likely full names)
        """
        doc = self.nlp(text)
        potentials = []
        
        i = 0
        while i < len(doc):
            token = doc[i]
            
            # Check for capitalized word that's not at sentence start
            if (token.text[0].isupper() and 
                len(token.text) > 2 and
                not token.is_punct and
                token.pos_ in {"PROPN", "NOUN", "X"}):  # Proper noun, noun, or unknown
                
                # Check if next token is also capitalized (full name)
                name_tokens = [token]
                j = i + 1
                while j < len(doc):
                    next_token = doc[j]
                    if (next_token.text[0].isupper() and 
                        len(next_token.text) > 1 and
                        not next_token.is_punct):
                        name_tokens.append(next_token)
                        j += 1
                    else:
                        break
                
                # Create entity from consecutive capitalized tokens
                if name_tokens:
                    start = name_tokens[0].idx
                    end = name_tokens[-1].idx + len(name_tokens[-1].text)
                    full_text = text[start:end]
                    
                    potentials.append({
                        "text": full_text,
                        "label": "POTENTIAL_NAME",
                        "start": start,
                        "end": end,
                        "tokens": [t.text for t in name_tokens]
                    })
                    
                    i = j
                    continue
            
            i += 1
        
        return potentials
    
    def extract_all(self, text: str) -> Dict[str, List[Dict]]:
        """
        Extract both NER entities and potential names.
        
        Returns a dictionary with:
            - "ner": Entities from spaCy NER
            - "potential": Potential names from heuristics
            - "combined": Merged and deduplicated list
        """
        ner_entities = self.extract_entities(text)
        potential_names = self.extract_potential_names(text)
        
        # Combine and deduplicate
        combined = list(ner_entities)
        ner_spans = {(e["start"], e["end"]) for e in ner_entities}
        
        for pot in potential_names:
            # Only add if not overlapping with NER entities
            if not any(self._overlaps(pot, ner_spans)):
                combined.append(pot)
        
        # Sort by position
        combined.sort(key=lambda x: x["start"])
        
        return {
            "ner": ner_entities,
            "potential": potential_names,
            "combined": combined
        }
    
    def _overlaps(self, entity: Dict, spans: Set[tuple]) -> bool:
        """Check if entity overlaps with any span in the set."""
        e_start, e_end = entity["start"], entity["end"]
        for s_start, s_end in spans:
            if not (e_end <= s_start or e_start >= s_end):
                return True
        return False
    
    def analyze_transcript(self, segments: List[Dict]) -> Dict:
        """
        Analyze all segments in a transcript and return entity statistics.
        
        Args:
            segments: List of segment dicts with "text" key
        
        Returns:
            Dictionary with:
                - all_entities: All extracted entities
                - entity_counts: Counter of entity texts
                - label_counts: Counter of entity labels
                - segments_with_entities: Segments that contain entities
        """
        all_entities = []
        entity_texts = []
        label_counts = Counter()
        segments_with_entities = []
        
        for i, segment in enumerate(segments):
            text = segment.get("text", "")
            if not text:
                continue
            
            result = self.extract_all(text)
            entities = result["combined"]
            
            if entities:
                segments_with_entities.append({
                    "segment_index": i,
                    "text": text,
                    "entities": entities
                })
            
            for entity in entities:
                all_entities.append(entity)
                entity_texts.append(entity["text"])
                label_counts[entity["label"]] += 1
        
        return {
            "all_entities": all_entities,
            "entity_counts": Counter(entity_texts),
            "label_counts": dict(label_counts),
            "segments_with_entities": segments_with_entities,
            "total_entities": len(all_entities),
            "unique_entities": len(set(entity_texts))
        }


def demo():
    """Demo the NER extractor on sample football commentary."""
    extractor = NERExtractor()
    
    sample_texts = [
        "Mesut Ozil passes to Alexis Sanchez on the left wing.",
        "David De Gea makes a brilliant save at Old Trafford!",
        "The match between Arsenal and Manchester United at Wembley.",
        "Ozil with a great pass to Sanchez!",  # Shortened names
        "wilshere injured again",  # Lowercase name (ASR artifact)
    ]
    
    print("\n" + "="*60)
    print("NER Extractor Demo")
    print("="*60)
    
    for text in sample_texts:
        print(f"\nText: {text}")
        result = extractor.extract_all(text)
        
        if result["ner"]:
            print(f"  NER entities: {[(e['text'], e['label']) for e in result['ner']]}")
        if result["potential"]:
            print(f"  Potential names: {[e['text'] for e in result['potential']]}")


if __name__ == "__main__":
    demo()
