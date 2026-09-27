"""Run before spending GPU money: is abc2midi installed and does Xiaomi's scorer work?

    python precheck.py

Xiaomi's own config warns: without abc2midi every score silently becomes 0.0
and the run "looks healthy". This catches that.
"""

import shutil
import sys

from reward import score_one

GOOD = """```abc
X:1
T:Precheck Waltz
M:3/4
L:1/8
Q:1/4=96
K:G
V:1 clef=treble
V:2 clef=bass
[V:1] |: G2 B2 d2 | c2 B2 A2 | B2 G2 E2 | D6 | G2 B2 d2 | e2 d2 c2 | B2 A2 F2 | G6 :|
[V:2] |: G,,2 D,2 G,2 | A,,2 E,2 A,2 | E,,2 B,,2 E,2 | D,,2 A,,2 D,2 | G,,2 D,2 G,2 | C,2 G,2 C2 | D,2 A,2 D2 | G,,6 :|
[V:1] |: d2 g2 f2 | e2 c2 A2 | B2 d2 G2 | A6 | B2 c2 d2 | e2 f2 g2 | f2 e2 d2 | g6 :|
[V:2] |: B,,2 G,2 D2 | C,2 G,2 E2 | G,,2 D,2 B,2 | D,,2 A,,2 F,2 | G,,2 D,2 B,2 | C,2 G,2 C2 | D,2 A,2 D2 | G,,6 :|
```"""

if not shutil.which("abc2midi"):
    sys.exit("FAIL: abc2midi not on PATH.  apt-get install -y abcmidi")

s = score_one(GOOD)
print("precheck piece ->", {k: v for k, v in s.items() if k != "abc"})
print("garbage text   ->", score_one("I cannot write music, sorry.")["total"])
print("OK: scorer is live" if s["total"] > 0 else "WARN: known-good piece scored 0 -- inspect 'fail' above")
