from thefuzz import fuzz, process
import json
import spacy
from typing import List, Dict, Set, Tuple

class NameNormalizer:
    """
    A NER-first approach to entity normalization.
    
    Strategy:
    1. Use spaCy NER to detect potential entities (PERSON, ORG, GPE, LOC)
    2. Only fuzzy-match detected entities against the gazetteer
    3. Apply strict thresholds and validation to avoid false positives
    """
    
    def __init__(self, path: str, spacy_model: str = "en_core_web_sm"):
        # Load known entities gazetteer
        with open(path, "r", encoding="utf-8") as f:
            self.known_names: List[str] = json.load(f)
        
        # Build lookup structures for faster matching
        self._build_lookup_structures()
        
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Downloading spaCy model '{spacy_model}'...")
            spacy.cli.download(spacy_model)
            self.nlp = spacy.load(spacy_model)
        
        print(f"NameNormalizer loaded with {len(self.known_names)} entities.")
        print(f"Using spaCy model: {spacy_model}")
    
    def _build_lookup_structures(self):
        """Build efficient lookup structures from the gazetteer."""
        # Set of all known names (full names)
        self.known_names_set: Set[str] = set(name.lower() for name in self.known_names)
        
        # Set of individual name parts (first names, last names)
        self.name_parts: Set[str] = set()
        for name in self.known_names:
            for part in name.split():
                if len(part) > 2:  # Skip initials
                    self.name_parts.add(part.lower())
        
        # Common words that should NEVER be matched (blocklist)
        # These are words that fuzzy matching often confuses with player names
        self.blocklist: Set[str] = {
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
            "league", "premier", "football", "soccer", "sport", "sports",
            "indeed", "exciting", "summer", "august", "saturday", "companion",
            "promotion", "palpitate", "experience", "begin", "starting"
        }
        
        print(f"Built lookup with {len(self.name_parts)} unique name parts")
    
    def _is_potential_entity(self, text: str) -> bool:
        """
        Quick heuristic check if text could be an entity.
        Returns False for obvious non-entities.
        """
        text_lower = text.lower().strip()
        
        # Skip if in blocklist
        if text_lower in self.blocklist:
            return False
        
        # Skip very short strings
        if len(text_lower) < 4:
            return False
        
        # Skip if all lowercase and not a known name part
        # (Real names in transcripts are often capitalized or should match known parts)
        if text.islower() and text_lower not in self.name_parts:
            return False
        
        return True
    
    def _fuzzy_match_entity(self, entity_text: str, threshold: int = 85) -> Tuple[str, int] | None:
        """
        Attempt to fuzzy match an entity against the gazetteer.
        Returns (matched_name, score) or None if no good match.
        """
        if not self._is_potential_entity(entity_text):
            return None
        
        # Try exact match first (case-insensitive)
        entity_lower = entity_text.lower()
        for name in self.known_names:
            if name.lower() == entity_lower:
                return (name, 100)
        
        # Try matching entity as a last name / single name part
        # This handles cases like "Ozil" -> "Mesut Ozil"
        for name in self.known_names:
            name_parts = name.lower().split()
            for part in name_parts:
                if entity_lower == part:
                    return (name, 100)
                # Also try fuzzy match on individual name parts
                # Lower threshold (80) to catch misspellings like valdez->valdes
                if len(entity_lower) >= 4 and len(part) >= 4:
                    ratio = fuzz.ratio(entity_lower, part)
                    if ratio >= 80:
                        return (name, ratio)
        
        # Try fuzzy matching against full names
        result = process.extractOne(
            entity_text, 
            self.known_names,
            scorer=fuzz.token_sort_ratio  # Better for name variations
        )
        
        if result:
            match, score = result[0], result[1]
            
            # Apply stricter validation
            if score >= threshold:
                # Additional check: the matched name should share significant characters
                # This helps avoid "there" -> "Wilshere" type matches
                if self._validate_match(entity_text, match, score):
                    return (match, score)
        
        return None
    
    def _validate_match(self, original: str, matched: str, score: int) -> bool:
        """
        Additional validation to prevent false positives.
        """
        orig_lower = original.lower()
        match_lower = matched.lower()
        
        # Check if original is a substring of match or vice versa
        # e.g., "Ozil" should match "Mesut Ozil"
        match_parts = match_lower.split()
        for part in match_parts:
            if orig_lower in part or part in orig_lower:
                return True
            # Check partial ratio for individual parts
            if fuzz.ratio(orig_lower, part) >= 80:
                return True
        
        # For high scores, do additional length check
        # The original should be at least 60% the length of the shortest name part
        min_part_len = min(len(p) for p in match_parts)
        if len(original) < min_part_len * 0.6:
            return False
        
        # Final check: token set ratio should also be high
        token_score = fuzz.token_set_ratio(original, matched)
        return token_score >= 75
    
    def extract_ner_entities(self, text: str) -> List[Dict]:
        """
        Use spaCy to extract named entities from text.
        Returns list of {text, start, end, label} dicts.
        """
        doc = self.nlp(text)
        entities = []
        
        # Extract NER entities
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP"}:
                entities.append({
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "label": ent.label_
                })
        
        # Also check individual capitalized words that might be missed by NER
        # (ASR often produces names that NER doesn't recognize)
        for token in doc:
            # Skip if already covered by an entity
            if any(token.idx >= e["start"] and token.idx < e["end"] for e in entities):
                continue
                
            # Check if it's a capitalized word
            if (token.text and token.text[0].isupper() and 
                len(token.text) > 3 and
                not token.is_punct):
                
                # Check if it looks like a name part (fuzzy match to known parts)
                token_lower = token.text.lower()
                if token_lower in self.name_parts:
                    entities.append({
                        "text": token.text,
                        "start": token.idx,
                        "end": token.idx + len(token.text),
                        "label": "POTENTIAL_NAME"
                    })
                else:
                    # Try fuzzy matching against name parts for possible misspellings
                    for name_part in self.name_parts:
                        if len(name_part) >= 4 and fuzz.ratio(token_lower, name_part) >= 80:
                            entities.append({
                                "text": token.text,
                                "start": token.idx,
                                "end": token.idx + len(token.text),
                                "label": "POTENTIAL_MISSPELLING"
                            })
                            break
        
        return entities
    
    def normalize_sentence(self, text: str, threshold: int = 85, debug: bool = False, expand_names: bool = False) -> str:
        """
        Normalize entities in a sentence using NER-first approach.
        
        Args:
            text: Input text to normalize
            threshold: Minimum fuzzy match score (default 85, stricter than before)
            debug: If True, print matching details
            expand_names: If True, expand short names to full names (e.g., "Ozil" -> "Mesut Ozil")
                         If False, only correct misspellings without expanding
        
        Returns:
            Text with normalized entity names
        """
        if not text.strip():
            return text
        
        # Step 1: Extract entities using NER
        entities = self.extract_ner_entities(text)
        
        if debug and entities:
            print(f"Detected entities: {[(e['text'], e['label']) for e in entities]}")
        
        # Step 2: Sort entities by position (reverse) to replace from end to start
        # This preserves character positions during replacement
        entities_sorted = sorted(entities, key=lambda x: x["start"], reverse=True)
        
        # Step 3: Try to match and replace each entity
        result = text
        replacements = []
        
        for entity in entities_sorted:
            entity_text = entity["text"]
            match_result = self._fuzzy_match_entity(entity_text, threshold)
            
            if match_result:
                matched_name, score = match_result
                
                # Determine the replacement text
                replacement = self._get_smart_replacement(
                    result, entity, entity_text, matched_name, expand_names
                )
                
                if replacement and replacement != entity_text:
                    if debug:
                        print(f"  Matched: '{entity_text}' -> '{replacement}' (score: {score})")
                    
                    # Replace in the result string
                    result = result[:entity["start"]] + replacement + result[entity["end"]:]
                    replacements.append((entity_text, replacement, score))
        
        return result
    
    def _get_smart_replacement(self, text: str, entity: Dict, entity_text: str, 
                               matched_name: str, expand_names: bool) -> str:
        """
        Determine the smart replacement for an entity, avoiding duplications.
        
        For example:
        - "Old Trafford" in text, entity="Trafford", match="Old Trafford" -> keep "Trafford"
        - "Valdez" in text, match="Victor Valdes" -> "Valdes" (correct spelling only)
        - "Wilser" in text, match="Jack Wilshere" -> "Wilshere" (correct spelling)
        """
        entity_lower = entity_text.lower()
        matched_parts = matched_name.split()
        
        # If the matched name has multiple parts
        if len(matched_parts) > 1:
            # Check if preceding word in text is already part of the match
            start = entity["start"]
            preceding_text = text[:start].rstrip().lower()
            preceding_words = preceding_text.split() if preceding_text else []
            
            if preceding_words:
                last_word = preceding_words[-1].rstrip('.,!?:;')
                # Check if the preceding word is part of the matched name
                for i, part in enumerate(matched_parts[:-1]):
                    if fuzz.ratio(last_word, part.lower()) >= 85:
                        # The preceding word is already part of the name, 
                        # so only use the remaining parts
                        remaining_parts = matched_parts[i+1:]
                        if remaining_parts:
                            # Find the best matching part for the entity
                            best_part = None
                            best_score = 0
                            for rp in remaining_parts:
                                score = fuzz.ratio(entity_lower, rp.lower())
                                if score > best_score:
                                    best_score = score
                                    best_part = rp
                            return best_part if best_part else remaining_parts[-1]
            
            # If not expanding names, only return the part that matches best
            if not expand_names:
                # Find which part of the matched name best corresponds to the entity
                best_part = None
                best_score = 0
                for part in matched_parts:
                    score = fuzz.ratio(entity_lower, part.lower())
                    if score > best_score:
                        best_score = score
                        best_part = part
                
                # If we found a good match, return just that part (corrected spelling)
                if best_part and best_score >= 70:
                    return best_part
                
                # Otherwise return the last name (most common case for football)
                return matched_parts[-1]
        
        # Single-word match or expand_names is True
        return matched_name if expand_names else matched_name
    
    def normalize_sentence_simple(self, text: str, threshold: int = 90) -> str:
        """
        Simpler word-by-word approach but with better filtering.
        Use this as a fallback or for comparison.
        """
        words = text.split()
        cleaned_words = []
        
        for word in words:
            # Skip short words
            if len(word) <= 3:
                cleaned_words.append(word)
                continue
            
            # Skip blocklisted words
            if word.lower() in self.blocklist:
                cleaned_words.append(word)
                continue
            
            # Only try matching if word is capitalized or in name_parts
            if not (word[0].isupper() or word.lower() in self.name_parts):
                cleaned_words.append(word)
                continue
            
            # Try fuzzy match
            match_result = self._fuzzy_match_entity(word, threshold)
            
            if match_result:
                matched_name, score = match_result
                cleaned_words.append(matched_name)
            else:
                cleaned_words.append(word)
        
        return " ".join(cleaned_words)