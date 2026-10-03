import re
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent/"starter"))
from data_prep import AGG_OPS,COND_OPS

AGG_WORDS=[a.lower() for a in AGG_OPS]

def parse_query(text):
    text=text.strip()
    select_pattern = r"^select\s+(?:(\w+)\s+)?<c(\d+)>(.*)$"
    select_match = re.match(select_pattern, text, re.IGNORECASE)
    if not select_match:
        return None
    
    agg_word,sel_col,rest=select_match.groups()
    sel=int(sel_col)
    agg=0
    
    if agg_word:
        agg_word_lower=agg_word.lower()
        if agg_word_lower in AGG_WORDS:
            agg=AGG_WORDS.index(agg_word_lower)
        else:
            return None
        
    rest=rest.strip()
    
    if not rest:
        return {"sel":sel,"agg":agg,"conds":[]}
    
    if not rest.lower().startswith("where"):
        return None
    
    cond_pattern = r"\b(where|and)\s+<c(\d+)>\s+([=><])"
    matches = list(re.finditer(cond_pattern, rest, re.IGNORECASE))
    if not matches or matches[0].start() != 0 or matches[0].group(1).lower() != "where":
        return None        
    conds=[]
    for i,match in enumerate(matches):
        col=int(match.group(2))
        op=match.group(3)
        
        start_val=match.end()
        end_val=matches[i+1].start() if i+1<len(matches) else len(rest)
        
        value=rest[start_val:end_val].strip()
        
        if not value:
            return None
        
        conds.append([col,COND_OPS.index(op),value])
        
    return {"sel":sel,"agg":agg,"conds":conds}