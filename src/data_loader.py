import json
from pathlib import Path

def load_transcript(file_path):
    path=Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    segments=data.get("segments", {})
    transcript_data=[]
    for key in segments:
        seg=segments[key]
        if len(seg)>=3:
            transcript_data.append({
                id: key,
                "start": seg[0],
                "end": seg[1],
                "text": seg[2]
            })
            
    return transcript_data

def save_transcript(transcript_data, file_path):
    output={"segments": {}}
    for item in transcript_data:
        output["segments"][item["id"]]=[item["start"], item["end"], item["text"]]
    with open(file_path, 'w', encoding='utf-8') as file:   
        json.dump(output, file, ensure_ascii=False, indent=4)