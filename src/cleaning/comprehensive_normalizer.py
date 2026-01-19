"""
Comprehensive Name Normalizer for ASR Football Transcripts

This module uses multiple techniques to identify ALL possible misspellings:
1. Phonetic matching (Metaphone) - catches sound-alike errors
2. Synthetic misspelling generation - auto-generates common spelling variations
3. Character n-gram similarity - catches partial matches
4. Contextual pattern detection - identifies names by surrounding words
5. Aggressive fuzzy matching with validation
"""

import json
import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict

import spacy
from thefuzz import fuzz

# Try to import phonetic libraries, install if needed
try:
    import jellyfish
except ImportError:
    jellyfish = None


class ComprehensiveNormalizer:
    """
    A comprehensive normalizer that uses multiple techniques to catch
    all possible misspellings in football ASR transcripts.
    """
    
    def __init__(self, gazetteer_path: str = None):
        """Initialize the comprehensive normalizer."""
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Load gazetteer
        if gazetteer_path is None:
            gazetteer_path = Path(__file__).parent.parent.parent / "data" / "external" / "entities_2015.json"
        
        with open(gazetteer_path, 'r', encoding='utf-8') as f:
            self.gazetteer = json.load(f)
        
        # Build comprehensive lookup tables
        self._build_comprehensive_lookups()
        
        # Build blocklist
        self.blocklist = self._build_blocklist()
        
        # Context patterns that indicate a name might follow/precede
        self.name_context_patterns = [
            r'\b(?:by|from|to|for|with|against|between)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:scores?|shoots?|passes?|heads?|clears?|saves?|tackles?)',
            r'(?:goal|save|pass|shot|header|tackle)\s+(?:by|from)\s+([A-Z][a-z]+)',
            r"([A-Z][a-z]+)'s\s+(?:ball|shot|pass|goal|save)",
        ]
        
        print(f"ComprehensiveNormalizer initialized:")
        print(f"  - {len(self.canonical_names)} canonical entities")
        print(f"  - {len(self.all_variations)} total variations (including synthetic)")
        print(f"  - {len(self.phonetic_map)} phonetic codes")
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing accents and converting to lowercase."""
        nfkd = unicodedata.normalize('NFKD', text)
        return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower()
    
    def _get_phonetic_codes(self, text: str) -> List[str]:
        """Get phonetic codes for a text using multiple algorithms."""
        codes = []
        normalized = self._normalize_text(text)
        
        if jellyfish:
            try:
                codes.append(('soundex', jellyfish.soundex(normalized)))
            except:
                pass
            try:
                codes.append(('metaphone', jellyfish.metaphone(normalized)))
            except:
                pass
            try:
                codes.append(('nysiis', jellyfish.nysiis(normalized)))
            except:
                pass
        else:
            # Simple phonetic approximation if jellyfish not available
            codes.append(('simple', self._simple_phonetic(normalized)))
        
        return codes
    
    def _simple_phonetic(self, text: str) -> str:
        """Simple phonetic encoding as fallback."""
        # Remove vowels except at start, collapse double letters
        if not text:
            return ""
        result = text[0]
        prev = text[0]
        for c in text[1:]:
            if c not in 'aeiou' and c != prev:
                result += c
                prev = c
        return result[:6]
    
    def _generate_synthetic_misspellings(self, name: str) -> List[str]:
        """
        Generate common misspelling variations of a name.
        
        Covers:
        - Missing letters
        - Double letters
        - Swapped adjacent letters
        - Common phonetic substitutions
        - Accent variations
        """
        variations = set()
        name_lower = name.lower()
        
        # 1. Missing single letters
        for i in range(len(name_lower)):
            variation = name_lower[:i] + name_lower[i+1:]
            if len(variation) >= 3:
                variations.add(variation)
        
        # 2. Double letters
        for i in range(len(name_lower)):
            variation = name_lower[:i] + name_lower[i] + name_lower[i:]
            variations.add(variation)
        
        # 3. Swapped adjacent letters
        for i in range(len(name_lower) - 1):
            variation = name_lower[:i] + name_lower[i+1] + name_lower[i] + name_lower[i+2:]
            variations.add(variation)
        
        # 4. Common phonetic substitutions
        substitutions = {
            'ph': 'f', 'f': 'ph',
            'c': 'k', 'k': 'c',
            'c': 's', 's': 'c',
            'j': 'g', 'g': 'j',
            'z': 's', 's': 'z',
            'v': 'b', 'b': 'v',
            'i': 'y', 'y': 'i',
            'i': 'e', 'e': 'i',
            'a': 'e', 'e': 'a',
            'o': 'u', 'u': 'o',
            'll': 'l', 'l': 'll',
            'ss': 's', 's': 'ss',
            'rr': 'r', 'r': 'rr',
            'nn': 'n', 'n': 'nn',
            'tt': 't', 't': 'tt',
            'ck': 'k', 'k': 'ck',
            'gu': 'g', 'g': 'gu',
            'qu': 'k', 'k': 'qu',
            'x': 'ks', 'ks': 'x',
            'sh': 's', 's': 'sh',
            'ch': 'sh', 'sh': 'ch',
            'th': 't', 't': 'th',
            'wh': 'w', 'w': 'wh',
            'ough': 'o', 'ow': 'o',
            'tion': 'sion', 'sion': 'tion',
            'ei': 'ie', 'ie': 'ei',
            'ea': 'ee', 'ee': 'ea',
        }
        
        for old, new in substitutions.items():
            if old in name_lower:
                variations.add(name_lower.replace(old, new, 1))
        
        # 5. Accent/diacritic variations
        accent_map = {
            'a': ['á', 'à', 'ä', 'â', 'ã'],
            'e': ['é', 'è', 'ë', 'ê'],
            'i': ['í', 'ì', 'ï', 'î'],
            'o': ['ó', 'ò', 'ö', 'ô', 'õ'],
            'u': ['ú', 'ù', 'ü', 'û'],
            'n': ['ñ'],
            'c': ['ç'],
        }
        
        for char, accented_versions in accent_map.items():
            if char in name_lower:
                for accented in accented_versions:
                    variations.add(name_lower.replace(char, accented, 1))
        
        # 6. Common ASR errors (phonetic confusions)
        asr_confusions = [
            ('er', 'a'), ('er', 'ar'), ('ar', 'er'),
            ('or', 'er'), ('er', 'or'),
            ('an', 'en'), ('en', 'an'),
            ('in', 'en'), ('en', 'in'),
            ('le', 'el'), ('el', 'le'),
            ('ey', 'y'), ('y', 'ey'),
            ('ow', 'o'), ('o', 'ow'),
            ('ay', 'ey'), ('ey', 'ay'),
        ]
        
        for old, new in asr_confusions:
            if old in name_lower:
                variations.add(name_lower.replace(old, new, 1))
        
        return list(variations)
    
    def _build_comprehensive_lookups(self):
        """Build comprehensive lookup tables including synthetic misspellings."""
        # Core lookups
        self.canonical_names: Set[str] = set()
        self.all_variations: Dict[str, Tuple[str, str]] = {}  # variation -> (canonical, corrected_form)
        self.phonetic_map: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # phonetic_code -> [(canonical, corrected)]
        self.name_parts: Dict[str, str] = {}  # name_part -> canonical
        
        def find_best_correction(variation: str, canonical: str) -> str:
            """Find the best correction that matches the variation structure."""
            variation_parts = variation.split()
            canonical_parts = canonical.split()
            
            if len(variation_parts) == len(canonical_parts):
                return canonical
            
            if len(variation_parts) == 1:
                # Single word - find best matching part (skip short parts like "De")
                best_match = None
                best_score = 0
                for part in canonical_parts:
                    if len(part) < 3 and len(canonical_parts) > 1:
                        continue
                    score = fuzz.ratio(variation.lower(), part.lower())
                    if score > best_score or (score == best_score and best_match and len(part) > len(best_match)):
                        best_score = score
                        best_match = part
                return best_match if best_match else max(canonical_parts, key=len)
            
            return canonical
        
        def process_entity(entity: Dict):
            """Process a single entity with all its variations."""
            canonical = entity["canonical"]
            self.canonical_names.add(canonical)
            
            # Add canonical name parts
            for part in canonical.split():
                if len(part) >= 3:
                    self.name_parts[part.lower()] = canonical
            
            # Process explicit misspellings from gazetteer
            for misspelling in entity.get("misspellings", []):
                corrected = find_best_correction(misspelling, canonical)
                self.all_variations[misspelling.lower()] = (canonical, corrected)
                self._add_phonetic_entry(misspelling, canonical, corrected)
            
            # Process aliases
            for alias in entity.get("aliases", []):
                corrected = find_best_correction(alias, canonical)
                self.all_variations[alias.lower()] = (canonical, corrected)
                self._add_phonetic_entry(alias, canonical, corrected)
                # Add alias parts
                for part in alias.split():
                    if len(part) >= 3:
                        self.name_parts[part.lower()] = canonical
            
            # Generate synthetic misspellings for canonical name
            for part in canonical.split():
                if len(part) >= 4:  # Only for longer parts
                    for synthetic in self._generate_synthetic_misspellings(part):
                        if synthetic not in self.all_variations:
                            self.all_variations[synthetic] = (canonical, part)
                            self._add_phonetic_entry(synthetic, canonical, part)
            
            # Generate synthetic misspellings for aliases
            for alias in entity.get("aliases", []):
                for part in alias.split():
                    if len(part) >= 4:
                        for synthetic in self._generate_synthetic_misspellings(part):
                            if synthetic not in self.all_variations:
                                corrected = find_best_correction(synthetic, canonical)
                                self.all_variations[synthetic] = (canonical, corrected)
        
        # Process all entities
        if "players" in self.gazetteer:
            for team, players in self.gazetteer["players"].items():
                for player in players:
                    process_entity(player)
        
        for category in ["teams", "venues", "competitions"]:
            if category in self.gazetteer:
                for entity in self.gazetteer[category]:
                    process_entity(entity)
    
    def _add_phonetic_entry(self, text: str, canonical: str, corrected: str):
        """Add a text to the phonetic lookup map."""
        for code_type, code in self._get_phonetic_codes(text):
            if code:
                key = f"{code_type}:{code}"
                self.phonetic_map[key].append((canonical, corrected))
    
    def _build_blocklist(self) -> Set[str]:
        """Build a comprehensive blocklist of common words to never match."""
        common_words = {
            # Articles and pronouns
            "the", "a", "an", "this", "that", "these", "those",
            "he", "she", "it", "they", "we", "you", "i",
            "his", "her", "its", "their", "our", "your", "my",
            "him", "them", "us",
            
            # Common verbs
            "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might",
            "can", "must", "shall",
            "go", "goes", "went", "going", "gone",
            "get", "gets", "got", "getting",
            "make", "makes", "made", "making",
            "take", "takes", "took", "taking", "taken",
            "come", "comes", "came", "coming",
            "see", "sees", "saw", "seeing", "seen",
            "know", "knows", "knew", "knowing", "known",
            "think", "thinks", "thought", "thinking",
            "want", "wants", "wanted", "wanting",
            "give", "gives", "gave", "giving", "given",
            "find", "finds", "found", "finding",
            "tell", "tells", "told", "telling",
            "say", "says", "said", "saying",
            "put", "puts", "putting",
            "keep", "keeps", "kept", "keeping",
            "let", "lets", "letting",
            "begin", "begins", "began", "beginning", "begun",
            "seem", "seems", "seemed", "seeming",
            "leave", "leaves", "left", "leaving",
            "call", "calls", "called", "calling",
            "try", "tries", "tried", "trying",
            "ask", "asks", "asked", "asking",
            "need", "needs", "needed", "needing",
            "feel", "feels", "felt", "feeling",
            "become", "becomes", "became", "becoming",
            "turn", "turns", "turned", "turning",
            "start", "starts", "started", "starting",
            "show", "shows", "showed", "showing", "shown",
            "hear", "hears", "heard", "hearing",
            "play", "plays", "played", "playing",
            "run", "runs", "ran", "running",
            "move", "moves", "moved", "moving",
            "live", "lives", "lived", "living",
            "believe", "believes", "believed", "believing",
            "hold", "holds", "held", "holding",
            "bring", "brings", "brought", "bringing",
            "happen", "happens", "happened", "happening",
            "write", "writes", "wrote", "writing", "written",
            "provide", "provides", "provided", "providing",
            "sit", "sits", "sat", "sitting",
            "stand", "stands", "stood", "standing",
            "lose", "loses", "lost", "losing",
            "pay", "pays", "paid", "paying",
            "meet", "meets", "met", "meeting",
            "include", "includes", "included", "including",
            "continue", "continues", "continued", "continuing",
            "set", "sets", "setting",
            "learn", "learns", "learned", "learning",
            "change", "changes", "changed", "changing",
            "lead", "leads", "led", "leading",
            "understand", "understands", "understood", "understanding",
            "watch", "watches", "watched", "watching",
            "follow", "follows", "followed", "following",
            "stop", "stops", "stopped", "stopping",
            "create", "creates", "created", "creating",
            "speak", "speaks", "spoke", "speaking", "spoken",
            "read", "reads", "reading",
            "spend", "spends", "spent", "spending",
            "grow", "grows", "grew", "growing", "grown",
            "open", "opens", "opened", "opening",
            "walk", "walks", "walked", "walking",
            "win", "wins", "won", "winning",
            "offer", "offers", "offered", "offering",
            "remember", "remembers", "remembered", "remembering",
            "love", "loves", "loved", "loving",
            "consider", "considers", "considered", "considering",
            "appear", "appears", "appeared", "appearing",
            "buy", "buys", "bought", "buying",
            "wait", "waits", "waited", "waiting",
            "serve", "serves", "served", "serving",
            "die", "dies", "died", "dying",
            "send", "sends", "sent", "sending",
            "expect", "expects", "expected", "expecting",
            "build", "builds", "built", "building",
            "stay", "stays", "stayed", "staying",
            "fall", "falls", "fell", "falling", "fallen",
            "cut", "cuts", "cutting",
            "reach", "reaches", "reached", "reaching",
            "kill", "kills", "killed", "killing",
            "remain", "remains", "remained", "remaining",
            
            # Prepositions and conjunctions
            "in", "on", "at", "to", "for", "with", "by", "from",
            "up", "about", "into", "over", "after", "under",
            "between", "out", "against", "during", "through",
            "before", "above", "below", "behind", "beyond",
            "and", "or", "but", "so", "yet", "nor",
            "if", "then", "than", "when", "where", "while",
            "because", "although", "unless", "until", "since",
            
            # Question words
            "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
            
            # Adjectives and adverbs
            "good", "great", "best", "better", "bad", "worse", "worst",
            "new", "old", "young", "big", "small", "large", "little",
            "long", "short", "high", "low", "right", "left", "wrong",
            "first", "last", "next", "early", "late",
            "well", "very", "just", "also", "only", "even", "still",
            "now", "then", "here", "there", "never", "always", "often",
            "really", "actually", "probably", "maybe", "perhaps",
            
            # Numbers and time
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "hundred", "thousand", "million",
            "today", "tomorrow", "yesterday", "week", "month", "year",
            "time", "day", "night", "morning", "evening", "afternoon",
            "minute", "second", "hour",
            
            # Football-specific common words (not names)
            "ball", "goal", "match", "game", "team", "player", "players",
            "half", "first", "second", "extra", "penalty", "corner",
            "free", "kick", "shot", "pass", "cross", "header",
            "save", "tackle", "foul", "card", "yellow", "red",
            "offside", "referee", "manager", "coach", "captain",
            "substitute", "substitution", "injury", "injured",
            "pitch", "field", "stadium", "ground",
            "league", "cup", "championship", "title", "trophy",
            "season", "final", "semifinal", "quarterfinal",
            "home", "away", "draw", "win", "loss", "defeat", "victory",
            "score", "result", "table", "standings", "points",
            "premier", "division", "conference",
            
            # Other common words
            "thing", "things", "way", "ways", "part", "parts",
            "place", "places", "case", "cases", "point", "points",
            "work", "world", "life", "hand", "hands",
            "people", "man", "men", "woman", "women", "child", "children",
            "number", "fact", "area", "money", "story",
            "lot", "kind", "head", "side", "end",
            "problem", "question", "answer", "reason", "example",
            "group", "company", "government", "country", "state",
            "school", "family", "system", "program", "service",
            
            # Specific problematic words
            "english", "spanish", "french", "german", "italian",
            "european", "american", "british", "national", "international",
            "gunner", "gunners", "devil", "devils", "united",
            
            # Country names that should not be corrected
            "monaco", "ecuador", "spain", "england", "france", "germany",
            "italy", "brazil", "argentina", "portugal", "netherlands",
            "belgium", "colombia", "chile", "mexico", "usa", "japan",
            "korea", "australia", "wales", "scotland", "ireland",
            
            # Other teams/organizations not in our gazetteer
            "real", "madrid", "barcelona", "barca", "bayern", "munich",
            "juventus", "inter", "milan", "psg", "dortmund", "atletico",
            "sevilla", "valencia", "villarreal", "sporting", "benfica", "porto",
            "ajax", "feyenoord", "celtic", "rangers", "olympiakos", "piraeus",
            
            # Common words that match player names phonetically
            "all", "more", "plus", "canal", "angel", "theo", "quite",
            "lack", "jack", "daley", "daily", "maria", "fio",
            "anderson", "ander", "hector", "lionel",
        }
        return common_words
    
    def _lookup_comprehensive(self, text: str) -> Optional[Tuple[str, str, float, str]]:
        """
        Comprehensive lookup using multiple techniques.
        
        Returns: (canonical, corrected, confidence, match_type) or None
        """
        text_lower = text.lower().strip()
        text_normalized = self._normalize_text(text)
        
        # Skip blocklisted words
        if text_normalized in self.blocklist:
            return None
        
        # Skip very short text
        if len(text_normalized) < 3:
            return None
        
        # 1. Exact match in variations (highest priority)
        if text_lower in self.all_variations:
            canonical, corrected = self.all_variations[text_lower]
            return (canonical, corrected, 1.0, "exact_variation")
        
        # 2. Normalized match (handles accents)
        if text_normalized in self.all_variations:
            canonical, corrected = self.all_variations[text_normalized]
            return (canonical, corrected, 0.98, "normalized_variation")
        
        # 3. Exact canonical match
        for canonical in self.canonical_names:
            if self._normalize_text(canonical) == text_normalized:
                return (canonical, canonical, 1.0, "exact_canonical")
        
        # 4. Phonetic matching
        phonetic_match = self._phonetic_lookup(text)
        if phonetic_match:
            return phonetic_match
        
        # 5. Fuzzy matching against all variations (lower threshold)
        fuzzy_match = self._fuzzy_lookup(text, threshold=75)
        if fuzzy_match:
            return fuzzy_match
        
        # 6. Name part matching (for single words that might be surnames)
        if text_normalized in self.name_parts:
            canonical = self.name_parts[text_normalized]
            # Find the matching part
            for part in canonical.split():
                if self._normalize_text(part) == text_normalized:
                    return (canonical, part, 0.85, "name_part")
        
        return None
    
    def _phonetic_lookup(self, text: str) -> Optional[Tuple[str, str, float, str]]:
        """Look up text using phonetic codes."""
        for code_type, code in self._get_phonetic_codes(text):
            if code:
                key = f"{code_type}:{code}"
                if key in self.phonetic_map:
                    # Get matches and find best one by fuzzy similarity
                    matches = self.phonetic_map[key]
                    best_match = None
                    best_score = 0
                    
                    for canonical, corrected in matches:
                        # Compare to the corrected form
                        score = fuzz.ratio(text.lower(), corrected.lower())
                        if score > best_score and score >= 60:
                            best_score = score
                            best_match = (canonical, corrected)
                    
                    if best_match:
                        return (best_match[0], best_match[1], best_score / 100, f"phonetic_{code_type}")
        
        return None
    
    def _fuzzy_lookup(self, text: str, threshold: int = 75) -> Optional[Tuple[str, str, float, str]]:
        """Fuzzy matching against all known variations."""
        text_lower = text.lower()
        best_match = None
        best_score = 0
        
        for variation, (canonical, corrected) in self.all_variations.items():
            # Only compare if lengths are reasonably similar
            if abs(len(text_lower) - len(variation)) <= 4:
                score = fuzz.ratio(text_lower, variation)
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = (canonical, corrected)
        
        if best_match:
            return (best_match[0], best_match[1], best_score / 100, "fuzzy")
        
        return None
    
    def _extract_candidates(self, text: str) -> List[Dict]:
        """Extract candidate entities using NER and heuristics."""
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
            if token.text[0].isupper() and not token.text.isupper():
                if token.idx not in covered_spans:
                    if token.text.lower() not in self.blocklist:
                        candidates.append({
                            "text": token.text,
                            "start": token.idx,
                            "end": token.idx + len(token.text),
                            "source": "CAPITALIZED"
                        })
        
        # 3. Context-based detection
        for pattern in self.name_context_patterns:
            for match in re.finditer(pattern, text):
                name = match.group(1) if match.lastindex else match.group(0)
                start = match.start(1) if match.lastindex else match.start()
                end = match.end(1) if match.lastindex else match.end()
                
                # Check if not already covered
                if start not in covered_spans:
                    candidates.append({
                        "text": name,
                        "start": start,
                        "end": end,
                        "source": "CONTEXT"
                    })
        
        # 4. Multi-word sequences that might be names
        words = text.split()
        for i, word in enumerate(words):
            if word and word[0].isupper():
                # Check for two-word names
                if i + 1 < len(words) and words[i + 1] and words[i + 1][0].isupper():
                    two_word = f"{word} {words[i + 1]}"
                    # Find position in original text
                    pos = text.find(two_word)
                    if pos != -1 and pos not in covered_spans:
                        candidates.append({
                            "text": two_word,
                            "start": pos,
                            "end": pos + len(two_word),
                            "source": "MULTI_WORD"
                        })
        
        # Sort by position (end to start for replacement)
        candidates.sort(key=lambda x: x["start"], reverse=True)
        
        # Remove overlapping candidates (prefer longer matches)
        filtered = []
        for candidate in candidates:
            overlaps = any(
                not (candidate["end"] <= other["start"] or candidate["start"] >= other["end"])
                for other in filtered
            )
            if not overlaps:
                filtered.append(candidate)
        
        return filtered
    
    def _strip_punctuation(self, text: str) -> Tuple[str, str, str]:
        """Strip leading/trailing punctuation from text."""
        leading = ""
        trailing = ""
        
        # Strip leading punctuation
        i = 0
        while i < len(text) and text[i] in '.,!?:;\'\"()-[]{}':
            leading += text[i]
            i += 1
        
        # Strip trailing punctuation
        j = len(text) - 1
        while j >= i and text[j] in '.,!?:;\'\"()-[]{}':
            trailing = text[j] + trailing
            j -= 1
        
        clean = text[i:j+1] if i <= j else ""
        return leading, clean, trailing
    
    def normalize_sentence(self, text: str, debug: bool = False) -> str:
        """
        Normalize a sentence, correcting all detected misspellings.
        
        Args:
            text: The input text to normalize
            debug: If True, print debug information
            
        Returns:
            The normalized text with corrections applied
        """
        candidates = self._extract_candidates(text)
        
        if debug:
            print(f"Candidates: {[(c['text'], c['source']) for c in candidates]}")
        
        result = text
        
        for candidate in candidates:
            original = candidate["text"]
            leading, clean_text, trailing = self._strip_punctuation(original)
            
            if len(clean_text) < 3:
                continue
            
            lookup_result = self._lookup_comprehensive(clean_text)
            
            if lookup_result:
                canonical, corrected, confidence, match_type = lookup_result
                
                # Skip exact matches (no change needed)
                if self._normalize_text(corrected) == self._normalize_text(clean_text):
                    continue
                
                # CRITICAL: Only apply corrections if:
                # 1. It's detected by NER as a named entity, OR
                # 2. It's an exact match in our known misspellings
                is_ner_entity = candidate["source"].startswith("NER:")
                is_exact_misspelling = match_type in ("exact_variation", "normalized_variation")
                
                if not is_ner_entity and not is_exact_misspelling:
                    # For non-NER candidates (CAPITALIZED, CONTEXT, MULTI_WORD),
                    # only allow exact misspelling matches from gazetteer
                    continue
                
                # Skip if the original text is already part of the canonical name
                # This prevents "De Gea" -> "David De Gea" expansion
                clean_normalized = self._normalize_text(clean_text)
                canonical_normalized = self._normalize_text(canonical)
                if clean_normalized in canonical_normalized:
                    # Check if clean_text matches a suffix of canonical (e.g., "De Gea" in "David De Gea")
                    canonical_parts = canonical.split()
                    is_valid_suffix = False
                    for i in range(len(canonical_parts)):
                        suffix = ' '.join(canonical_parts[i:])
                        if self._normalize_text(suffix) == clean_normalized:
                            is_valid_suffix = True
                            break
                    if is_valid_suffix:
                        continue  # Already correct, skip
                
                # For low-confidence matches, require additional validation
                if confidence < 0.8:
                    # Validate with context or skip
                    if not self._validate_with_context(text, candidate["start"], corrected):
                        continue
                
                replacement = leading + corrected + trailing
                
                if debug:
                    print(f"  {match_type}: '{original}' -> '{replacement}' ({confidence:.0%})")
                
                # Apply replacement
                result = result[:candidate["start"]] + replacement + result[candidate["end"]:]
        
        return result
    
    def _validate_with_context(self, text: str, position: int, corrected: str) -> bool:
        """Validate a match using context clues."""
        # Get surrounding text
        start = max(0, position - 30)
        end = min(len(text), position + 30)
        context = text[start:end].lower()
        
        # Football context words that suggest a name follows/precedes
        football_context = [
            "goal", "score", "pass", "shot", "header", "tackle", "save",
            "by", "from", "to", "with", "against",
            "player", "striker", "defender", "midfielder", "goalkeeper",
        ]
        
        for word in football_context:
            if word in context:
                return True
        
        return False
    
    def analyze_transcript(self, segments: List[Dict], debug: bool = False) -> Dict:
        """
        Analyze a full transcript and return detailed results.
        
        Returns dict with:
        - corrections: list of all corrections made
        - potential_misses: candidates that weren't corrected (for review)
        - statistics: summary statistics
        """
        results = {
            "corrections": [],
            "potential_misses": [],
            "statistics": {
                "total_segments": len(segments),
                "segments_modified": 0,
                "total_corrections": 0,
            }
        }
        
        for i, segment in enumerate(segments):
            original = segment.get("text", "")
            cleaned = self.normalize_sentence(original, debug=debug)
            
            if cleaned != original:
                results["statistics"]["segments_modified"] += 1
                results["corrections"].append({
                    "segment_id": i,
                    "original": original,
                    "cleaned": cleaned,
                })
            
            # Track potential misses (capitalized words that weren't corrected)
            candidates = self._extract_candidates(original)
            for candidate in candidates:
                _, clean_text, _ = self._strip_punctuation(candidate["text"])
                if clean_text not in cleaned:
                    results["potential_misses"].append({
                        "segment_id": i,
                        "word": clean_text,
                        "context": original,
                    })
        
        results["statistics"]["total_corrections"] = len(results["corrections"])
        
        return results


def install_dependencies():
    """Install required dependencies for comprehensive matching."""
    import subprocess
    import sys
    
    packages = ["jellyfish"]
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


if __name__ == "__main__":
    # Install dependencies if needed
    install_dependencies()
    
    # Test the comprehensive normalizer
    normalizer = ComprehensiveNormalizer()
    
    test_cases = [
        "Smolin makes a tackle",
        "Hasley Young runs down the wing",
        "Valdez with a great save",
        "Felaini scores a header",
        "Chris Mullin clears the ball",
        "Dejaa saves the shot",
        "Wiltshire passes to Giroud",
        "Mertez controls the ball",
    ]
    
    print("\nTest Results:")
    print("-" * 60)
    for test in test_cases:
        result = normalizer.normalize_sentence(test, debug=True)
        print(f"  Input:  {test}")
        print(f"  Output: {result}")
        print()
