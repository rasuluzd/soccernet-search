"""
Improved Name Normalizer for Football Transcripts.

This module provides a robust approach to normalize misspelled entities in ASR transcripts.

Key improvements:
1. Uses a structured gazetteer with canonical names, aliases, and known misspellings
2. Proper handling of punctuation (strips before matching, restores after)
3. Direct misspelling lookup for fast, accurate corrections
4. NER-based candidate detection to avoid false positives
5. Phonetic matching as fallback for unknown misspellings
"""

from thefuzz import fuzz, process
import json
import spacy
import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EntityMatch:
    """Represents a matched entity with its correction."""
    original: str
    canonical: str
    match_type: str  # 'exact', 'alias', 'misspelling', 'fuzzy'
    confidence: float
    start: int
    end: int


class ImprovedNameNormalizer:
    """
    Improved normalizer using structured gazetteer with aliases and misspellings.
    """
    
    def __init__(self, gazetteer_path: str, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the normalizer.
        
        Args:
            gazetteer_path: Path to the structured entities JSON file
            spacy_model: spaCy model for NER
        """
        # Load and process gazetteer
        with open(gazetteer_path, "r", encoding="utf-8") as f:
            self.gazetteer = json.load(f)
        
        # Build lookup dictionaries
        self._build_lookups()
        
        # Load spaCy
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Downloading spaCy model '{spacy_model}'...")
            spacy.cli.download(spacy_model)
            self.nlp = spacy.load(spacy_model)
        
        # Blocklist of common words to never match
        self.blocklist = self._build_blocklist()
        
        print(f"ImprovedNameNormalizer initialized:")
        print(f"  - {len(self.canonical_names)} canonical entities")
        print(f"  - {len(self.misspelling_map)} known misspellings")
        print(f"  - {len(self.alias_map)} aliases")
    
    def _build_lookups(self):
        """Build fast lookup dictionaries from the gazetteer."""
        # Maps misspelling -> canonical name
        self.misspelling_map: Dict[str, str] = {}
        
        # Maps alias -> canonical name
        self.alias_map: Dict[str, str] = {}
        
        # Set of canonical names
        self.canonical_names: Set[str] = set()
        
        # Maps canonical name parts -> full canonical name
        self.name_parts_map: Dict[str, List[str]] = {}
        
        # All searchable terms for fuzzy matching
        self.all_terms: List[str] = []
        
        def process_entity(entity: Dict):
            """Process a single entity entry."""
            canonical = entity["canonical"]
            self.canonical_names.add(canonical)
            self.all_terms.append(canonical)
            
            # Add canonical name parts
            for part in canonical.split():
                part_lower = part.lower()
                if len(part) > 2:
                    if part_lower not in self.name_parts_map:
                        self.name_parts_map[part_lower] = []
                    self.name_parts_map[part_lower].append(canonical)
            
            # Add aliases
            for alias in entity.get("aliases", []):
                self.alias_map[alias.lower()] = canonical
                self.all_terms.append(alias)
                # Also add individual alias parts
                for part in alias.split():
                    if len(part) > 2:
                        part_lower = part.lower()
                        if part_lower not in self.name_parts_map:
                            self.name_parts_map[part_lower] = []
                        if canonical not in self.name_parts_map[part_lower]:
                            self.name_parts_map[part_lower].append(canonical)
            
            # Add misspellings
            for misspelling in entity.get("misspellings", []):
                self.misspelling_map[misspelling.lower()] = canonical
        
        # Process players
        if "players" in self.gazetteer:
            for team, players in self.gazetteer["players"].items():
                for player in players:
                    process_entity(player)
        
        # Process teams, venues, competitions
        for category in ["teams", "venues", "competitions"]:
            if category in self.gazetteer:
                for entity in self.gazetteer[category]:
                    process_entity(entity)
    
    def _build_blocklist(self) -> Set[str]:
        """Build a comprehensive blocklist of words to never match."""
        return {
            # Articles and very common words - NEVER match these
            "the", "a", "an", "and", "or", "but", "for", "not", "all", "any",
            # Common words that fuzzy match to names
            "there", "their", "they", "where", "were", "here", "which",
            "what", "when", "will", "with", "this", "that", "then",
            "from", "have", "been", "being", "would", "could", "should",
            "about", "after", "before", "other", "these", "those",
            "some", "such", "much", "very", "just", "also", "only",
            "still", "even", "well", "back", "over", "into", "than",
            "them", "more", "most", "both", "each", "made", "make",
            "like", "look", "looked", "looking", "come", "came", "coming",
            "going", "gone", "take", "took", "taken", "give", "gave",
            "given", "find", "found", "think", "thought", "know", "knew",
            "known", "want", "wanted", "need", "needed", "seem", "seemed",
            "good", "great", "best", "better", "first", "last", "next",
            "right", "left", "side", "time", "times", "year", "years",
            "game", "games", "play", "played", "playing", "ball", "goal",
            "goals", "team", "teams", "match", "matches", "season",
            "indeed", "exciting", "summer", "august", "saturday", "companion",
            "promotion", "palpitate", "experience", "begin", "starting",
            "minute", "minutes", "second", "seconds", "half", "halves",
            "shot", "shots", "pass", "passes", "cross", "crosses",
            "header", "headers", "save", "saves", "foul", "fouls",
            "corner", "corners", "throw", "throws", "kick", "kicks",
            "area", "areas", "line", "lines", "post", "posts",
            "red", "yellow", "card", "cards", "penalty", "penalties",
            # Words that incorrectly match to team/competition names
            "english", "spanish", "german", "french", "italian",
            "league", "premier", "division", "cup", "champions",
            "gunner", "gunners", "devil", "devils",
            # Common Spanish/translated words that appear in this transcript
            "también", "pero", "porque", "para", "como", "cuando",
            "entre", "desde", "hasta", "sobre", "bajo", "contra"
        }
    
    def _strip_punctuation(self, text: str) -> Tuple[str, str, str]:
        """
        Strip punctuation from text while preserving it for restoration.
        
        Returns:
            (leading_punct, clean_text, trailing_punct)
        """
        if not text:
            return "", "", ""
            
        # Match leading punctuation
        leading_match = re.match(r'^([^\w]*)', text)
        leading = leading_match.group(1) if leading_match else ""
        
        # Match trailing punctuation  
        trailing_match = re.search(r'([^\w]*)$', text)
        trailing = trailing_match.group(1) if trailing_match else ""
        
        # Get clean text
        clean = text[len(leading):len(text)-len(trailing)] if trailing else text[len(leading):]
        
        return leading, clean, trailing
    
    def _lookup_entity(self, text: str) -> Optional[Tuple[str, str, float]]:
        """
        Look up an entity in the gazetteer.
        
        Returns:
            (canonical_name, match_type, confidence) or None
        """
        if not text:
            return None
            
        text_lower = text.lower().strip()
        
        # Skip blocklisted words
        if text_lower in self.blocklist:
            return None
        
        # Skip very short strings (minimum 4 chars for single words)
        if len(text_lower) < 4:
            return None
        
        # Skip if it looks like a common word (all lowercase, not in our database)
        if text.islower() and text_lower not in self.misspelling_map and text_lower not in self.alias_map:
            # Check if it's not a known name part
            if text_lower not in self.name_parts_map:
                return None
        
        # 1. Exact canonical match
        for canonical in self.canonical_names:
            if canonical.lower() == text_lower:
                return (canonical, "exact", 1.0)
        
        # 2. Direct misspelling lookup (highest priority for corrections!)
        if text_lower in self.misspelling_map:
            return (self.misspelling_map[text_lower], "misspelling", 0.98)
        
        # 3. Alias lookup
        if text_lower in self.alias_map:
            canonical = self.alias_map[text_lower]
            # If the original text is already part of the canonical name, no change needed
            # This prevents "De Gea" -> "David De Gea" -> "Gea" issues
            if text_lower in canonical.lower():
                return None
            return (canonical, "alias", 0.95)
        
        # 4. Name part lookup (e.g., "Ozil" -> "Mesut Ozil")
        if text_lower in self.name_parts_map:
            canonical = self.name_parts_map[text_lower][0]
            # If the original text matches a word in the canonical name, no change needed
            canonical_parts = [p.lower() for p in canonical.split()]
            if text_lower in canonical_parts:
                return None
            return (canonical, "name_part", 0.90)
        
        # 5. Fuzzy matching against misspellings (for unknown variations)
        best_misspelling_match = None
        best_misspelling_score = 0
        
        for misspelling, canonical in self.misspelling_map.items():
            # Only compare if lengths are similar (avoid comparing "a" to "alexander")
            if abs(len(text_lower) - len(misspelling)) <= 3:
                score = fuzz.ratio(text_lower, misspelling)
                if score > best_misspelling_score and score >= 80:
                    best_misspelling_score = score
                    best_misspelling_match = canonical
        
        if best_misspelling_match:
            return (best_misspelling_match, "fuzzy_misspelling", best_misspelling_score / 100)
        
        # 6. Fuzzy matching against name parts
        best_part_match = None
        best_part_score = 0
        
        for part, canonicals in self.name_parts_map.items():
            if len(part) >= 4 and len(text_lower) >= 4:
                # Only compare if lengths are similar
                if abs(len(text_lower) - len(part)) <= 2:
                    score = fuzz.ratio(text_lower, part)
                    if score > best_part_score and score >= 80:
                        best_part_score = score
                        best_part_match = canonicals[0]
        
        if best_part_match:
            return (best_part_match, "fuzzy_name_part", best_part_score / 100)
        
        return None
    
    def _extract_candidates(self, text: str) -> List[Dict]:
        """
        Extract candidate entities from text using NER and heuristics.
        
        Returns list of candidate dicts with: text, start, end, source
        """
        doc = self.nlp(text)
        candidates = []
        covered_spans = set()
        
        # 1. NER entities
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP"}:
                candidates.append({
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "source": f"NER:{ent.label_}"
                })
                for i in range(ent.start_char, ent.end_char):
                    covered_spans.add(i)
        
        # 2. Capitalized words not covered by NER
        for token in doc:
            if token.idx in covered_spans:
                continue
            
            if not token.text or len(token.text) < 3:
                continue
                
            # Check capitalized words
            if token.text[0].isupper() and not token.is_punct:
                candidates.append({
                    "text": token.text,
                    "start": token.idx,
                    "end": token.idx + len(token.text),
                    "source": "CAPITALIZED"
                })
        
        # 3. Multi-word candidates (consecutive capitalized words)
        i = 0
        tokens = list(doc)
        while i < len(tokens):
            if tokens[i].text and len(tokens[i].text) > 0 and tokens[i].text[0].isupper() and not tokens[i].is_punct:
                # Start of potential multi-word name
                start_idx = tokens[i].idx
                end_idx = tokens[i].idx + len(tokens[i].text)
                words = [tokens[i].text]
                
                j = i + 1
                while j < len(tokens):
                    # Check for consecutive capitalized words (allowing for space)
                    next_token = tokens[j]
                    if (next_token.text and 
                        len(next_token.text) > 0 and
                        next_token.text[0].isupper() and 
                        not next_token.is_punct and
                        next_token.idx <= end_idx + 2):  # Allow for space
                        words.append(next_token.text)
                        end_idx = next_token.idx + len(next_token.text)
                        j += 1
                    else:
                        break
                
                if len(words) >= 2:
                    multi_text = " ".join(words)
                    # Check if this doesn't overlap with existing candidates
                    if not any(c["start"] == start_idx and c["end"] == end_idx for c in candidates):
                        candidates.append({
                            "text": multi_text,
                            "start": start_idx,
                            "end": end_idx,
                            "source": "MULTI_WORD"
                        })
                
                i = j
            else:
                i += 1
        
        return candidates
    
    def normalize_sentence(self, text: str, debug: bool = False) -> str:
        """
        Normalize entities in a sentence.
        
        Args:
            text: Input text
            debug: Print debug information
        
        Returns:
            Normalized text
        """
        if not text or not text.strip():
            return text
        
        # Extract candidates
        candidates = self._extract_candidates(text)
        
        if debug and candidates:
            print(f"Candidates: {[(c['text'], c['source']) for c in candidates]}")
        
        # Sort by position (reverse) to replace from end to start
        # This avoids offset tracking issues
        candidates.sort(key=lambda x: -x["start"])
        
        # Remove overlapping candidates (prefer longer matches)
        filtered_candidates = []
        for candidate in candidates:
            overlaps = False
            for existing in filtered_candidates:
                # Check if this candidate overlaps with an existing one
                if not (candidate["end"] <= existing["start"] or candidate["start"] >= existing["end"]):
                    overlaps = True
                    break
            if not overlaps:
                filtered_candidates.append(candidate)
        
        result = text
        
        for candidate in filtered_candidates:
            
            original = candidate["text"]
            
            # Strip punctuation for lookup
            leading_punct, clean_text, trailing_punct = self._strip_punctuation(original)
            
            # Skip if clean text is too short
            if len(clean_text) < 3:
                continue
            
            # Look up the clean text
            lookup_result = self._lookup_entity(clean_text)
            
            if lookup_result:
                canonical, match_type, confidence = lookup_result
                
                # If exact match, no replacement needed
                if match_type == "exact":
                    continue
                
                # Determine the replacement
                replacement = self._smart_replacement(
                    result, candidate["start"], clean_text, canonical
                )
                
                if replacement and replacement != clean_text:
                    # Restore punctuation
                    full_replacement = leading_punct + replacement + trailing_punct
                    
                    if debug:
                        print(f"  {match_type}: '{original}' -> '{full_replacement}' ({confidence:.0%})")
                    
                    # Replace in result (from end to start, so positions stay valid)
                    result = result[:candidate["start"]] + full_replacement + result[candidate["end"]:]
        
        return result
    
    def _smart_replacement(self, text: str, start: int, 
                          clean_text: str, canonical: str) -> Optional[str]:
        """
        Determine the smart replacement, avoiding duplications.
        """
        clean_lower = clean_text.lower()
        canonical_parts = canonical.split()
        
        # If single-word canonical, just return it
        if len(canonical_parts) == 1:
            return canonical
        
        # Check if preceding text already contains part of the canonical name
        preceding_text = text[:start].lower()
        preceding_words = preceding_text.split()
        
        if preceding_words:
            last_word = preceding_words[-1].rstrip('.,!?:;\'\"')
            
            # Check each part of canonical name
            for i, part in enumerate(canonical_parts[:-1]):
                if fuzz.ratio(last_word, part.lower()) >= 80:
                    # Preceding word matches this part, return remaining parts
                    remaining = canonical_parts[i+1:]
                    if remaining:
                        # Find the best matching remaining part
                        best_match = None
                        best_score = 0
                        for rp in remaining:
                            score = fuzz.ratio(clean_lower, rp.lower())
                            if score > best_score:
                                best_score = score
                                best_match = rp
                        return best_match if best_match else remaining[-1]
        
        # For multi-word matches, find the best matching part
        best_part = None
        best_score = 0
        for part in canonical_parts:
            score = fuzz.ratio(clean_lower, part.lower())
            if score > best_score:
                best_score = score
                best_part = part
        
        # If original text closely matches a specific part, return just that part (corrected)
        if best_part and best_score >= 70:
            return best_part
        
        # Default: return last name (most common for football)
        return canonical_parts[-1]
    
    def get_entity_info(self, text: str) -> Optional[Dict]:
        """
        Get detailed information about an entity.
        
        Returns dict with canonical name, type, and confidence.
        """
        _, clean_text, _ = self._strip_punctuation(text)
        result = self._lookup_entity(clean_text)
        
        if result:
            canonical, match_type, confidence = result
            return {
                "original": text,
                "canonical": canonical,
                "match_type": match_type,
                "confidence": confidence
            }
        return None


# Backwards compatibility - keep old interface working
class NameNormalizer(ImprovedNameNormalizer):
    """Backwards-compatible wrapper that works with both old and new formats."""
    
    def __init__(self, path: str, spacy_model: str = "en_core_web_sm"):
        # Check if it's the old simple format or new structured format
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # Old format - use legacy initialization
            print("Using legacy simple list format. Consider upgrading to structured gazetteer.")
            self._init_from_simple_list(data, spacy_model)
        else:
            # New format
            super().__init__(path, spacy_model)
    
    def _init_from_simple_list(self, names: List[str], spacy_model: str):
        """Initialize from simple list of names (legacy format)."""
        self.canonical_names = set(names)
        self.misspelling_map = {}
        self.alias_map = {}
        self.name_parts_map = {}
        self.all_terms = list(names)
        self.gazetteer = {"legacy": names}
        
        for name in names:
            for part in name.split():
                if len(part) > 2:
                    part_lower = part.lower()
                    if part_lower not in self.name_parts_map:
                        self.name_parts_map[part_lower] = []
                    self.name_parts_map[part_lower].append(name)
        
        self.blocklist = self._build_blocklist()
        
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            spacy.cli.download(spacy_model)
            self.nlp = spacy.load(spacy_model)
        
        print(f"NameNormalizer (legacy) loaded with {len(self.canonical_names)} entities.")


def demo():
    """Demo the improved normalizer."""
    import os
    
    # Use the new structured gazetteer
    gazetteer_path = "data/external/entities_2015.json"
    
    if not os.path.exists(gazetteer_path):
        print(f"Gazetteer not found: {gazetteer_path}")
        return
    
    normalizer = ImprovedNameNormalizer(gazetteer_path)
    
    test_cases = [
        # Should NOT change
        "There is a lot of excitement in the stadium.",
        "They will play well today.",
        "Which team will win this match?",
        
        # SHOULD change - misspellings
        "Valdez makes a great save.",  # Valdez -> Valdes
        "Wilser is back from injury.",  # Wilser -> Wilshere
        "Felaini scores a header!",  # Felaini -> Fellaini
        "Chris Mullin clears the ball.",  # Mullin -> Smalling
        "Hasley Young runs down the wing.",  # Hasley -> Ashley
        "Daley Blin makes a tackle.",  # Blin -> Blind
        
        # Punctuation handling
        "Valdez, with a great save!",
        "It was Ozil's pass.",
        "(Giroud) scores again.",
        
        # Already correct
        "Great save by De Gea at Old Trafford!",
        "Mesut Ozil passes to Alexis Sanchez.",
    ]
    
    print("\n" + "="*70)
    print("Improved Normalizer Demo")
    print("="*70)
    
    for text in test_cases:
        result = normalizer.normalize_sentence(text, debug=True)
        changed = "✓ CHANGED" if text != result else "○ NO CHANGE"
        print(f"\n{changed}")
        print(f"  Input:  {text}")
        print(f"  Output: {result}")


if __name__ == "__main__":
    demo()
