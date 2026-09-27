"""Fair blind test: AFTER = first valid sample for the prompt; BEFORE = base model's sample,
blank lines removed (a favour to the base model so we compare music, not formatting)."""
import json, random, shutil, subprocess as sp
from pathlib import Path
from reward import score_one
SF2='/usr/share/sounds/sf2/FluidR3_GM.sf2'
def wav(abc, stem):
    Path(stem+'.abc').write_text(abc); sp.run(['abc2midi',stem+'.abc','-o',stem+'.mid'],capture_output=True)
    if not Path(stem+'.mid').exists(): return False
    sp.run(['fluidsynth','-ni','-g','0.8','-F',stem+'.wav','-r','44100',SF2,stem+'.mid'],capture_output=True); return Path(stem+'.wav').exists()
B={r['src_id']:r for r in json.load(open('results/before.json'))}; A=json.load(open('results/after.json'))
out=Path('blind'); shutil.rmtree(out,ignore_errors=True); out.mkdir(); tmp=Path('blind_tmp'); tmp.mkdir(exist_ok=True)
rng=random.Random(7); key={}; rows=[]; i=0
for r in A:
    a=next((s for s in r['samples'] if s['total']>0),None)
    if not a: continue
    b=None
    for s in B[r['src_id']]['samples']:
        if s.get('abc'):
            st='\n'.join(l for l in s['abc'].split('\n') if l.strip())
            sc=score_one('```abc\n'+st+'\n```')
            b=(st,sc['total']); break
    if not b: continue
    i+=1; sid=r['src_id']
    ok1=wav(a['abc'],str(tmp/f'{sid}_after')); ok2=wav(b[0],str(tmp/f'{sid}_before'))
    if not (ok1 and ok2): i-=1; continue
    flip=rng.random()<0.5
    shutil.copy(tmp/f'{sid}_{"after" if flip else "before"}.wav',out/f'{i:02d}_A.wav')
    shutil.copy(tmp/f'{sid}_{"before" if flip else "after"}.wav',out/f'{i:02d}_B.wav')
    key[f'{i:02d}']={'trained':'A' if flip else 'B','src_id':sid,'after_score':a['total'],'before_score_blankfixed':b[1]}
    rows.append(f'<div class=q><h3>Pair {i}</h3><p>{r["prompt"]}</p><div class=ab><span>A</span><audio controls src="{i:02d}_A.wav"></audio></div><div class=ab><span>B</span><audio controls src="{i:02d}_B.wav"></audio></div></div>')
json.dump(key,open(out/'ANSWER_KEY.json','w'),indent=1)
(out/'index.html').write_text('<!doctype html><meta charset=utf-8><title>Blind Test</title><style>body{background:#0b0b12;color:#fff;font-family:system-ui;max-width:820px;margin:auto;padding:24px}.q{border:1px solid #333;border-radius:12px;padding:16px;margin:16px 0}p{color:#aaa}.ab{display:flex;gap:12px;align-items:center;margin:8px 0}.ab span{font-size:28px;font-weight:700;color:#ff7a18;width:30px}audio{width:100%}</style><h1>Which one sounds better? A or B?</h1>'+''.join(rows))
print(i,'pairs')
