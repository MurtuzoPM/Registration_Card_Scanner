"""ocr_engine/pattern_extractor.py (v2)"""
import re
from difflib import SequenceMatcher

FIELD_NUMS = {1:"registration_card_number",2:"passport_number",3:"citizenship",4:"name_and_surname",5:"date_of_registration",6:"prs_mia_rt",7:"valid_until",8:"serial_control_number",9:"mia",10:"place_of_residence",11:"place_of_residence_cont",12:"inspector",13:"date_of_registration_extension"}
VALID_PASSPORT_PREFIXES = {"EC","AC","AN","AS","AT","AL","AP","A","P","PP"}
CYR_TO_LAT = {"Е":"E","С":"C","А":"A","О":"O","Р":"P","М":"M","Н":"H","К":"K","В":"B","Т":"T"}

def _center(bbox):
    xs=[p[0] for p in bbox]; ys=[p[1] for p in bbox]
    return (sum(xs)/4.0, sum(ys)/4.0)
def _rel_y(bbox, h): return _center(bbox)[1]/max(h,1)*100.0
def _rel_x(bbox, w): return _center(bbox)[0]/max(w,1)*100.0
def _image_bounds(blocks):
    if not blocks: return 1,1
    xs=[]; ys=[]
    for b in blocks:
        for p in b["bbox"]: xs.append(p[0]); ys.append(p[1])
    return (max(xs)-min(xs) if xs else 1), (max(ys)-min(ys) if ys else 1)
def _fuzzy_contains(h, n, t=0.72):
    h=h.lower(); n=n.lower()
    if n in h: return True
    for w in h.split():
        if len(w)<3: continue
        if SequenceMatcher(None,w,n).ratio()>=t: return True
    return False

# --- Dates ---
_FULL_DATE_RE = re.compile(r"(\d{1,2})[.,/\-\s](\d{1,2})[.,/\-\s](\d{2,4})")
def _norm_dmy(d,mo,y):
    try: di=int(d); mi=int(mo)
    except: return None
    if not (1<=di<=31 and 1<=mi<=12): return None
    if len(y)==2:
        yi=int(y); y=("20"+y) if yi<80 else ("19"+y)
    elif len(y)==4:
        yi=int(y)
        if yi<1900 or yi>2100: return None
    else: return None
    return f"{str(di).zfill(2)}.{str(mi).zfill(2)}.{y}"

def _extract_dates(blocks, w, h):
    found=[]
    for b in blocks:
        m=_FULL_DATE_RE.search(b["text"])
        if m:
            v=_norm_dmy(m.group(1),m.group(2),m.group(3))
            if v: found.append((_rel_y(b["bbox"],h),v,b["confidence"]))
    numeric=[]
    for b in blocks:
        t=b["text"].strip().strip("'`.,")
        if re.fullmatch(r"\d{1,4}",t):
            cy=_center(b["bbox"])[1]; cx=_center(b["bbox"])[0]
            numeric.append((cy,cx,t,b["confidence"],b["bbox"]))
    numeric.sort()
    used=set(); tol=max(8,h*0.04)
    for i,(cy,cx,t,c,bb) in enumerate(numeric):
        if i in used: continue
        row=[(cx,t,c,bb)]; ul={i}
        for j in range(i+1,len(numeric)):
            if j in used: continue
            cy2,cx2,t2,c2,bb2=numeric[j]
            if abs(cy2-cy)<=tol:
                row.append((cx2,t2,c2,bb2)); ul.add(j)
        if len(row)>=3:
            row.sort()
            for k in range(len(row)-2):
                d,mo,y=row[k][1],row[k+1][1],row[k+2][1]
                v=_norm_dmy(d,mo,y)
                if v:
                    conf=(row[k][2]+row[k+1][2]+row[k+2][2])/3.0
                    found.append((_rel_y(row[k][3],h),v,conf))
                    used|=ul; break
    if not found: return {}
    found.sort(key=lambda t:t[0])
    deduped=[]
    for y,v,c in found:
        if any(v==v2 and abs(y-y2)<5 for y2,v2,_ in deduped): continue
        deduped.append((y,v,c))
    out={}
    for yr,v,c in deduped:
        if yr<55 and 5 not in out: out[5]=(v,max(c,0.6))
        elif yr<80 and 7 not in out: out[7]=(v,max(c,0.6))
        elif 13 not in out: out[13]=(v,max(c,0.6))
    if 5 not in out and deduped: out[5]=(deduped[0][1],max(deduped[0][2],0.5))
    return out

# --- Passport ---
def _cyr_to_lat(s): return "".join(CYR_TO_LAT.get(c,c) for c in s)
def _norm_digits(s):
    return s.upper().replace("O","0").replace("Z","7").replace("S","5").replace("I","1").replace("B","8").replace("L","1")
def _extract_passport(blocks, w, h):
    cands=[]
    for b in blocks:
        text=b["text"].strip().strip("№N.:")
        m=re.match(r"^([A-Za-zА-Яа-я]{1,3})\s*([0-9OZSIBLozsibl]{6,9})$",text)
        if m:
            pref=_cyr_to_lat(m.group(1).upper()); dig=_norm_digits(m.group(2))
            if re.fullmatch(r"\d{6,9}",dig):
                if pref in VALID_PASSPORT_PREFIXES:
                    cands.append((pref+dig, max(b["confidence"],0.75)))
                else:
                    cands.append((dig, max(b["confidence"]*0.8,0.55)))
            continue
        m2=re.match(r"^([0-9OZSIBLozsibl]{6,9})$",text)
        if m2:
            dig=_norm_digits(m2.group(1))
            if re.fullmatch(r"\d{6,9}",dig):
                cands.append((dig, max(b["confidence"],0.6)))
    if not cands: return None
    cands.sort(key=lambda c:(-int(any(ch.isalpha() for ch in c[0])),-c[1]))
    return cands[0]

# --- Card number ---
def _extract_card_number(blocks, w, h):
    cands=[]
    for b in blocks:
        yr=_rel_y(b["bbox"],h); xr=_rel_x(b["bbox"],w)
        if yr>28: continue
        text=b["text"]; cleaned=re.sub(r"^[№NnoНн.:\s]+","",text).strip()
        m=re.fullmatch(r"(\d{6,8})",cleaned)
        if m: cands.append((yr,-xr,m.group(1),b["confidence"])); continue
        m2=re.search(r"№\s*(\d{6,8})",text)
        if m2: cands.append((yr,-xr,m2.group(1),b["confidence"]))
    if not cands: return None
    cands.sort()
    return (cands[0][2], max(cands[0][3],0.6))

# --- Serial ---
def _extract_serial(blocks, w, h):
    cands=[]
    for b in blocks:
        yr=_rel_y(b["bbox"],h)
        if yr<35 or yr>72: continue
        cleaned=re.sub(r"^[№NnoНн.:\s]+","",b["text"]).strip()
        m=re.fullmatch(r"(\d{3,5})",cleaned)
        if m: cands.append((-_rel_x(b["bbox"],w), m.group(1), b["confidence"]))
    if not cands: return None
    cands.sort()
    return (cands[0][1], max(cands[0][2],0.55))

# --- Name / Citizenship ---
NAME_LABEL_WORDS={"пасаб","насаб","ном","пасабу","насабу","инспектор","нозир"}
def _is_personal_name(text):
    t=text.strip(" .:;,-()_")
    if any(ch.isdigit() for ch in t): return False
    words=re.findall(r"[\u0400-\u04FF\u02B9]+",t)
    if len(words)<2: return False
    cap=[w for w in words if w[0].isupper() and len(w)>=3]
    if len(cap)<2: return False
    low=t.lower()
    for lbl in NAME_LABEL_WORDS:
        if lbl in low: return False
    return True
def _find_anchor(blocks, keys, thr=0.7):
    best=None; bs=0.0
    for b in blocks:
        t=b["text"].lower()
        for kw in keys:
            k=kw.lower()
            s=1.0 if k in t else SequenceMatcher(None,t,k).ratio()
            if s>bs and s>=thr: best,bs=b,s
    return best
def _below_of(blocks, a, mdy, xtol, w, h):
    if a is None: return []
    ax,ay=_center(a["bbox"]); out=[]
    for b in blocks:
        if b is a: continue
        bx,by=_center(b["bbox"])
        dy=(by-ay)/max(h,1)*100.0; dx=abs(bx-ax)/max(w,1)*100.0
        if 0<dy<=mdy and dx<=xtol: out.append((dy,b))
    out.sort(); return [b for _,b in out]
def _right_of(blocks, a, mdx, ytol, w, h):
    if a is None: return []
    ax,ay=_center(a["bbox"]); out=[]
    for b in blocks:
        if b is a: continue
        bx,by=_center(b["bbox"])
        dx=(bx-ax)/max(w,1)*100.0; dy=abs(by-ay)/max(h,1)*100.0
        if 0<dx<=mdx and dy<=ytol: out.append((dx,b))
    out.sort(); return [b for _,b in out]

CITIZENSHIP={"хитой":"Хитой","хитои":"Хитой","руссия":"Руссия","россия":"Россия","ӯзбекистон":"Ӯзбекистон","узбекистон":"Ӯзбекистон","афғонистон":"Афғонистон","афгонистон":"Афғонистон"}
def _extract_citizenship(blocks, w, h):
    a=_find_anchor(blocks,["ШАҲРВАНДӢ","ШАХРВАНДӢ","шахрванди","шаҳрванди","гражданство"],0.6)
    cands=_right_of(blocks,a,70,5,w,h)+_below_of(blocks,a,8,70,w,h)
    for b in cands:
        t=b["text"].strip(" .:;,-()_"); low=t.lower()
        if not t: continue
        if _fuzzy_contains(low,"шахрванди",0.8) or _fuzzy_contains(low,"шаҳрванди",0.8): continue
        if _is_personal_name(t): continue
        for kw,canon in CITIZENSHIP.items():
            if kw in low: return (canon, max(b["confidence"],0.75))
        if "гуреза" in low or re.search(r"\b(чиа|ниа|нии)\b",low):
            return (t, max(b["confidence"],0.7))
        wd=re.findall(r"[\u0400-\u04FF]{3,}",t)
        if len(wd)==1: return (t, max(b["confidence"],0.55))
    return None

def _extract_name(blocks, w, h):
    cit=_find_anchor(blocks,["ШАҲРВАНДӢ","ШАХРВАНДӢ","шахрванди","шаҳрванди"],0.6)
    nl=_find_anchor(blocks,["пасаб ё ном","насаб ё ном","(пасаб ё ном)","(насаб ё ном)","пасабу ном","насабу ном"],0.55)
    if nl is not None:
        nx,ny=_center(nl["bbox"]); cs=[]
        for b in blocks:
            if b is nl: continue
            bx,by=_center(b["bbox"])
            dy=(ny-by)/max(h,1)*100.0; dx=abs(bx-nx)/max(w,1)*100.0
            if 0<dy<=8 and dx<=35 and _is_personal_name(b["text"]): cs.append((dy,b))
        if cs:
            cs.sort(); ch=cs[0][1]
            return (ch["text"].strip(" .:;,-()_"), max(ch["confidence"],0.7))
    if cit is not None:
        cx,cy=_center(cit["bbox"]); cs=[]
        for b in blocks:
            if b is cit: continue
            bx,by=_center(b["bbox"])
            dy=(by-cy)/max(h,1)*100.0
            if 2<=dy<=14 and _is_personal_name(b["text"]): cs.append((dy,b))
        if cs:
            cs.sort(); ch=cs[0][1]
            return (ch["text"].strip(" .:;,-()_"), max(ch["confidence"],0.65))
    return None

def _extract_prs_mia_rt(blocks, w, h):
    for b in blocks:
        t=b["text"].upper()
        t2=t.replace("B","В").replace("K","К").replace("T","Т").replace("H","Н")
        has_x=bool(re.search(r"Х\s*Ш\s*Б",t2))
        has_v=bool(re.search(r"В\s*К\s*Д",t2))
        has_ct=bool(re.search(r"\bЧТ\b",t2))
        if (has_x and has_v) or (has_v and has_ct):
            cleaned=re.sub(r"[^А-ЯЁ\s]"," ",t2).strip()
            cleaned=re.sub(r"\s+"," ",cleaned)
            if len(cleaned)>=3: return (cleaned, max(b["confidence"],0.6))
    return None

def _extract_mia(blocks, w, h):
    cands=[]
    for b in blocks:
        text=b["text"].strip(); up=text.upper()
        up=up.replace("B","В").replace("K","К").replace("D","Д").replace("H","Н")
        cl=re.sub(r"[^А-ЯЁ]","",up)
        if len(cl)>4: continue
        if cl in {"ВКД","ВК","ВКЛ","ВНД","ВКЗ"}:
            yr=_rel_y(b["bbox"],h)
            if 50<=yr<=78: cands.append((abs(yr-68),"ВКД",b["confidence"]))
    if not cands: return None
    cands.sort(); return (cands[0][1], max(cands[0][2],0.6))

_CITY_RE=re.compile(r"^ш\s*\.\s*[\u0400-\u04FF][\u0400-\u04FF\u02B9\s\-]{2,}$",re.IGNORECASE)
_STREET_RE=re.compile(r"^(к(?:ӯч|уч|\u04EFч)?)\s*\.\s*[\u0400-\u04FF][\u0400-\u04FF0-9\s.'\"\-«»№/]+",re.IGNORECASE)
def _extract_place(blocks, w, h):
    for b in blocks:
        t=b["text"].strip().strip(" ,;:"); yr=_rel_y(b["bbox"],h)
        if not (55<=yr<=90): continue
        if _CITY_RE.match(t): return (t, max(b["confidence"],0.7))
    a=_find_anchor(blocks,["Ҷои истиқомат","чои истикомат","истиқомат","истикомат"],0.6)
    if a is not None:
        for b in _right_of(blocks,a,70,5,w,h)+_below_of(blocks,a,10,60,w,h):
            t=b["text"].strip(" .:;,-()"); low=t.lower()
            if "истикомат" in low or "истиқомат" in low: continue
            if re.match(r"^ш\b",t,re.IGNORECASE) and re.search(r"[\u0400-\u04FF]{3,}",t):
                return (t, max(b["confidence"],0.6))
    return None
def _extract_place_cont(blocks, w, h):
    for b in blocks:
        t=b["text"].strip().strip(" ,;:"); yr=_rel_y(b["bbox"],h)
        if not (60<=yr<=92): continue
        if _STREET_RE.match(t): return (t, max(b["confidence"],0.65))
    return None

_INSPECTOR_RE=re.compile(r"^[\u0400-\u04FF][\u0400-\u04FF\u02B9]{3,}(\s+[\u0400-\u04FF][\u0400-\u04FF\u02B9.]{0,})*$")
def _extract_inspector(blocks, w, h):
    a=_find_anchor(blocks,["нозир","инспектор"],0.65)
    cs=[]
    for b in blocks:
        t=b["text"].strip(" .:;,-()_"); yr=_rel_y(b["bbox"],h)
        if yr<72: continue
        if any(ch.isdigit() for ch in t): continue
        if _fuzzy_contains(t.lower(),"нозир",0.7): continue
        if not _INSPECTOR_RE.match(t): continue
        cs.append((yr,b,t))
    if a is not None and cs:
        ax,ay=_center(a["bbox"])
        cs.sort(key=lambda c: abs(_center(c[1]["bbox"])[0]-ax)+abs(_center(c[1]["bbox"])[1]-ay)*0.5)
    elif cs:
        cs.sort()
    if not cs: return None
    ch=cs[0]; return (ch[2], max(ch[1]["confidence"],0.55))

def extract_global_patterns(ocr_results):
    if not ocr_results: return {}
    w,h=_image_bounds(ocr_results); out={}
    out.update(_extract_dates(ocr_results,w,h))
    for fnum,fn in [(2,_extract_passport),(1,_extract_card_number),(8,_extract_serial),(6,_extract_prs_mia_rt),(3,_extract_citizenship),(4,_extract_name),(9,_extract_mia),(10,_extract_place),(11,_extract_place_cont),(12,_extract_inspector)]:
        v=fn(ocr_results,w,h)
        if v: out[fnum]=v
    return out