from thefuzz import process
import json

class NameNormalizer:
    def __init__(self,path):
        with open(path,"r",encoding="utf-8") as f:
            self.known_names=json.load(f)
            print(f"NameNormalizer loaded with {len(self.known_names)} entities.")

    def normalize_sentence(self,text,treshold=80):
        words=text.split()
        cleaned_words=[]
        for word in words:
            if len(word)<=3:
                cleaned_words.append(word)
                continue
            match,score=process.extractOne(word,self.known_names)
            if score>=treshold:
                corrected_name = match.split()[-1] 
                cleaned_words.append(corrected_name)
            else:
                cleaned_words.append(word)
        return " ".join(cleaned_words)