"""ocr_engine/pattern_extractor.py (v3) — tuned to real OCR output."""
import re
from difflib import SequenceMatcher

CYR_TO_LAT={"Е":"E","С":"C","А":"A","О":"O","Р":"P","М":"M","Н":"H","К":"K","В":"B","Т":"T"}
VALID_PASSPORT_PREFIXES={"EC","AC","AN","AS","AT","AL","AP","A","P","PP"}
KNOWN_CITIES=["Душанбе","Ваҳдат","Вахдат","Хуҷанд","Хучанд","Кӯлоб","Кулоб","Хоруғ","Хорог","Бохтар","Қӯрғонтеппа","Курган-Тюбе","Истаравшан","Турсунзода","Норак"]

def _center(bb): xs=[p[0] for p in bb]; ys=[p[1] for p in bb]; return (sum(xs)/4.0,sum(ys)/4.0)
def _rel_y(bb,h): return _center(bb)[1]/max(h,1)*100.0
def _rel_x(bb,w): return _center(bb)[0]/max(w,1)*100.0
def _image_bounds(blocks):
    if not blocks: return 1,1
    xs=[]; ys=[]
    for b in blocks:
        for p in b["bbox"]: xs.append(p[0]); ys.append(p[1])
    return (max(xs)-min(xs) if xs else 1),(max(ys)-min(ys) if ys else 1)
def _fuzzy(a,b,t=0.7): return SequenceMatcher(None,a.lower(),b.lower()).ratio()>=t

# --- Passport / card numbers: layout-aware (see id_extractor.py) ---
def _extract_passport(blocks, w, h):
    from .id_extractor import extract_passport_number
    return extract_passport_number(blocks, w, h)


def _extract_card_number(blocks, w, h):
    from .id_extractor import extract_registration_card_number
    return extract_registration_card_number(blocks, w, h)

# --- Citizenship: prefer a KNOWN country (lexicon); never force one ---
_COUNTRIES = ["Хитой", "Руссия", "Россия", "Афғонистон", "Узбекистон",
              "Ӯзбекистон", "Қирғизистон", "Покистон", "Эрон", "Туркия"]


def _extract_citizenship(blocks, w, h):
    cands = []
    for b in blocks:
        t = b["text"].strip(" .:;,-()_")
        low = t.lower()
        yr = _rel_y(b["bbox"], h)
        if not (20 <= yr <= 45):
            continue
        matched = None
        for c in _COUNTRIES:
            if c.lower() in low or _fuzzy(low, c, 0.78):
                matched = c
                break
        if matched:
            cands.append((b["confidence"] + 0.2, matched))
        elif re.search(r"[\u0400-\u04FF]{4,}", t) and not any(ch.isdigit() for ch in t):
            # Unknown citizenship: keep the OCR'd word so it is not lost.
            cands.append((b["confidence"], t))
    if not cands:
        return None
    cands.sort(reverse=True)
    return (cands[0][1], max(cands[0][0], 0.6))

# --- Name: 2-word capitalized Cyrillic in mid-upper region ---
def _is_personal_name(text):
    t=text.strip(" .:;,-()_")
    if any(ch.isdigit() for ch in t): return False
    words=re.findall(r"[\u0400-\u04FF\u02B9]+",t)
    if len(words)<2: return False
    cap=[w for w in words if w[0].isupper() and len(w)>=3]
    if len(cap)<2: return False
    low=t.lower()
    for lbl in ["пасаб","насаб","ном","инспектор","нозир","вазорат","корхо","чумхур","точикистон","точикия","регистр"]:
        if lbl in low: return False
    return True
def _extract_name(blocks, w, h):
    cands=[]
    for b in blocks:
        yr=_rel_y(b["bbox"],h)
        if not (25<=yr<=50): continue
        if _is_personal_name(b["text"]):
            cands.append((-b["confidence"],b["text"].strip(" .:;,-()_"),b["confidence"]))
    if not cands: return None
    cands.sort()
    return (cands[0][1], max(cands[0][2],0.7))

# --- Dates: collect numeric fragments in date rows, attempt triples ---
_FULL_DATE_RE=re.compile(r"(\d{1,2})[.,/\-\s](\d{1,2})[.,/\-\s](\d{2,4})")
def _norm_dmy(d,mo,y):
    try: di=int(d); mi=int(mo); yi=int(y)
    except: return None
    if not (1<=di<=31 and 1<=mi<=12): return None
    if len(y)==2: y=("20"+y) if yi<80 else ("19"+y)
    elif len(y)==4:
        if yi<1990 or yi>2100: return None
    else: return None
    return f"{str(di).zfill(2)}.{str(mi).zfill(2)}.{y}"
def _extract_dates(blocks, w, h):
    found=[]
    for b in blocks:
        m=_FULL_DATE_RE.search(b["text"])
        if m:
            v=_norm_dmy(m.group(1),m.group(2),m.group(3))
            if v: found.append((_rel_y(b["bbox"],h),v,b["confidence"]))
    # Collect numeric fragments and try same-row triples
    nums=[]
    for b in blocks:
        t=b["text"].strip().strip("'`.,№#")
        if re.fullmatch(r"\d{1,4}",t):
            cy=_center(b["bbox"])[1]; cx=_center(b["bbox"])[0]
            nums.append((cy,cx,t,b["confidence"],b["bbox"]))
    nums.sort()
    used=set(); tol=max(20,h*0.05)
    for i,(cy,cx,t,c,bb) in enumerate(nums):
        if i in used: continue
        row=[(cx,t,c,bb,i)]
        for j in range(i+1,len(nums)):
            if j in used: continue
            cy2,cx2,t2,c2,bb2=nums[j]
            if abs(cy2-cy)<=tol:
                row.append((cx2,t2,c2,bb2,j))
        if len(row)>=3:
            row.sort()
            for k in range(len(row)-2):
                d,mo,y=row[k][1],row[k+1][1],row[k+2][1]
                v=_norm_dmy(d,mo,y)
                if v:
                    conf=(row[k][2]+row[k+1][2]+row[k+2][2])/3.0
                    found.append((_rel_y(row[k][3],h),v,conf))
                    for r in row[k:k+3]: used.add(r[4])
                    break
    if not found: return {}
    found.sort()
    out={}; seen=[]
    for yr,v,c in found:
        if any(v==v2 and abs(yr-y2)<5 for y2,v2,_ in seen): continue
        seen.append((yr,v,c))
    for yr,v,c in seen:
        if yr<55 and 5 not in out: out[5]=(v,max(c,0.55))
        elif yr<80 and 7 not in out: out[7]=(v,max(c,0.55))
        elif 13 not in out: out[13]=(v,max(c,0.55))
    if 5 not in out and seen: out[5]=(seen[0][1],max(seen[0][2],0.5))
    return out

# --- Serial: 3-5 digit number in middle Y band, rightmost ---
def _extract_serial(blocks, w, h):
    cands=[]
    for b in blocks:
        yr=_rel_y(b["bbox"],h)
        if yr<38 or yr>72: continue
        cleaned=re.sub(r"^[№NnoНн.:\s]+","",b["text"]).strip()
        m=re.fullmatch(r"(\d{3,5})",cleaned)
        if m: cands.append((-_rel_x(b["bbox"],w),m.group(1),b["confidence"]))
    if not cands: return None
    cands.sort()
    return (cands[0][1], max(cands[0][2],0.55))

# --- PRS MIA RT: ХШБ ВКД ... pattern anywhere ---
def _extract_prs_mia_rt(blocks, w, h):
    for b in blocks:
        t = b["text"].upper().replace("B", "В").replace("K", "К").replace("T", "Т").replace("H", "Н")
        t = t.replace("M", "М").replace("G", "Г")
        if re.search(r"Х\s*Ш\s*Б", t) and re.search(r"В\s*К\s*Д", t):
            return ("ХШБ ВКД ЧТ", max(b["confidence"], 0.7))
        if re.search(r"ХШБ", t) and re.search(r"ВКД", t):
            return ("ХШБ ВКД ЧТ", max(b["confidence"], 0.65))
    return None

# --- MIA: ВКД, even if only inside a larger string ---
def _extract_mia(blocks, w, h):
    for b in blocks:
        t=b["text"].upper().replace("B","В").replace("K","К").replace("D","Д")
        if re.search(r"\bВКД\b",t):
            return ("ВКД", max(b["confidence"],0.65))
    # Standalone fuzzy
    for b in blocks:
        t=b["text"].strip()
        up=t.upper().replace("B","В").replace("K","К").replace("D","Д").replace("H","Н")
        cl=re.sub(r"[^А-ЯЁ]","",up)
        if cl in {"ВКД","ВК","ВКЛ"}:
            return ("ВКД", max(b["confidence"],0.6))
    return None

# --- Place: known city lookup (fuzzy), no ш. prefix required ---
def _extract_place(blocks, w, h):
    cands=[]
    for b in blocks:
        yr=_rel_y(b["bbox"],h)
        if not (52<=yr<=82): continue
        t=b["text"].strip(" .:;,-()_#")
        low=t.lower()
        for city in KNOWN_CITIES:
            cl=city.lower()
            if cl in low:
                return (f"ш. {city}", max(b["confidence"],0.7))
            # Fuzzy on each word
            for word in re.findall(r"[\u0400-\u04FF]+",t):
                if len(word)>=4 and SequenceMatcher(None,word.lower(),cl).ratio()>=0.78:
                    cands.append((b["confidence"],city,word))
    if cands:
        cands.sort(reverse=True)
        return (f"ш. {cands[0][1]}", max(cands[0][0],0.6))
    return None

# --- Place_cont: street, any block in lower-mid with mixed Cyr+digits ---
def _extract_place_cont(blocks, w, h):
    cands=[]
    for b in blocks:
        yr=_rel_y(b["bbox"],h)
        if not (58<=yr<=82): continue
        t=b["text"].strip()
        low=t.lower()
        # Recognize street prefix variants (к. / куч. / кӯч. / жуч / Хуч)
        has_prefix=bool(re.search(r"^(к|куч|кӯч|жуч|хуч)\s*[\.,]?",low))
        has_digit=any(c.isdigit() for c in t)
        cyr_words=re.findall(r"[\u0400-\u04FF]{3,}",t)
        if has_prefix and has_digit and cyr_words:
            return (t.strip(" .:;,_"), max(b["confidence"],0.6))
        if has_digit and len(cyr_words)>=1 and len(t)>=8:
            cands.append((b["confidence"],t))
    if cands:
        cands.sort(reverse=True)
        return (cands[0][1].strip(" .:;,_"), max(cands[0][0],0.5))
    return None

# --- Inspector: surname in bottom Y, right-half preferred ---
_INSPECTOR_RE=re.compile(r"^[\u0400-\u04FF][\u0400-\u04FF\u02B9]{3,}([\s\.][\u0400-\u04FF][\u0400-\u04FF\u02B9.]{0,})*$")
def _extract_inspector(blocks, w, h):
    cands=[]
    for b in blocks:
        yr=_rel_y(b["bbox"],h); xr=_rel_x(b["bbox"],w)
        if yr<70: continue
        t=b["text"].strip(" .:;,-()_")
        if any(ch.isdigit() for ch in t): continue
        if _fuzzy(t.lower(),"нозир",0.7): continue
        if not _INSPECTOR_RE.match(t): continue
        # Prefer right side
        cands.append((-xr,b["confidence"],t))
    if not cands: return None
    cands.sort()
    return (cands[0][2], max(cands[0][1],0.6))

def extract_global_patterns(ocr_results):
    if not ocr_results: return {}
    w,h=_image_bounds(ocr_results); out={}
    out.update(_extract_dates(ocr_results,w,h))
    for fn_num,fn in [(2,_extract_passport),(1,_extract_card_number),(3,_extract_citizenship),(4,_extract_name),(6,_extract_prs_mia_rt),(8,_extract_serial),(9,_extract_mia),(10,_extract_place),(11,_extract_place_cont),(12,_extract_inspector)]:
        v=fn(ocr_results,w,h)
        if v:
            # Don't duplicate passport value as card number
            if fn_num==1 and 2 in out and v[0]==out[2][0]: continue
            out[fn_num]=v
    return out