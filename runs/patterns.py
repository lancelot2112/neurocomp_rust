"""Patterns across island runs 9-14: recursion against held-out score (from the logs' tables),
and inhibition, gating and internal reward in the elites (from the checkpoints)."""
import re, collections, os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
RUNS = [9, 10, 11, 12, 13, 14]
PROBES = ['respond', 'tell', 'discriminate', 'reverse', 'remember', 'win-stay', 'copy', 'echo', 'babble',
          'surprise', 'maze', 'screen', 'repeat', 'select', 'feel']

# ---------- A: loops in the best against its held-out score on its current goal ----------
goal_re = re.compile(r'(\w+) ([\w-]+) ([\w-]+) (-?\d+)/(\d+) \(met')
loops_re = re.compile(r'loops: (\d+) of (\d+) wired populations on a loop \((\d+) self-loops\)')
rows = []  # (run, gen, island, stage, loopfrac, has_loop, score/goal, explored, marked)
for r in RUNS:
    path = f'runs/islands{r}/log.txt'
    lines = open(path, encoding='utf-8', errors='replace').read().split('\n')
    goals = {}  # island -> (stage, probe, goal) at the latest goals line
    i = 0
    while i < len(lines):
        l = lines[i]
        m = re.match(r'^ *(\d+)  \d\d:\d\d:\d\dZ', l)
        if m and 'goals:' in l:
            goals = {g[0]: (g[1], g[2], int(g[4])) for g in goal_re.findall(l)}
        m = re.match(r'^ *(\d+) +respon', l)
        if m:
            gen = int(m.group(1))
            cols = l.split()[1:]
            j = i + 1
            while j + 2 < len(lines) and re.match(r'^      [a-z]', lines[j]):
                isl = lines[j].split()[0]
                held = lines[j + 1].split()
                ops = lines[j + 2]
                lm = loops_re.search(ops)
                if isl in goals and lm:
                    stage, probe, goal = goals[isl]
                    if probe in PROBES:
                        k = PROBES.index(probe)
                        v = held[k] if k < len(held) else '.'
                        if v not in ('.',) and re.match(r'-?\d+$', v):
                            on, pops = int(lm.group(1)), int(lm.group(2))
                            ex = re.search(r'explored (\d+)', ops)
                            mk = re.search(r'marked (\d+)', ops)
                            rows.append((r, gen, isl, stage, on / max(pops, 1), on > 0, int(v) / goal,
                                         int(ex.group(1)) if ex else 0, int(mk.group(1)) if mk else 0))
                j += 3
            i = j
            continue
        i += 1

print('A. Recursion against held-out score (best of each island, every 10 generations, runs 9-14)')
print(f'   {len(rows)} readings')
by = collections.defaultdict(lambda: [[], []])
for r, gen, isl, stage, lf, hl, s, ex, mk in rows:
    by[(isl, stage)][1 if hl else 0].append(s)
print('   island/stage (n without, n with): mean held-out score as a share of the goal, without loops -> with loops')
wins = losses = 0
for key in sorted(by):
    a, b = by[key]
    if len(a) >= 5 and len(b) >= 5:
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        wins += mb > ma
        losses += mb < ma
        print(f'   {key[0]:>10} {key[1]:<8} ({len(a):3}, {len(b):3}): {ma:+.2f} -> {mb:+.2f}')
print(f'   loops better in {wins}, worse in {losses} of the island-stages with 5+ readings each way')
# heavy recursion: half or more of the populations on a loop
heavy = [s for *_, lf, hl, s, ex, mk in [(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]) for x in rows] if lf >= 0.5]
light = [x[6] for x in rows if 0 < x[4] < 0.5]
none = [x[6] for x in rows if x[4] == 0]
for name, v in (('no loops', none), ('under half on loops', light), ('half or more on loops', heavy)):
    if v:
        print(f'   all islands, {name:<22}: n {len(v):4}, mean {sum(v)/len(v):+.2f}, share at or above the goal {sum(1 for x in v if x >= 1)/len(v):.0%}')

# ---------- B: the elites' wiring ----------
INPUTS = [(0x00, 'S'), (0x07, 'eye'), (0x38, 'ear'), (0xC1, 'touch'), (0xE0, 'affect'), (0x80, 'M')]
OUTPUTS = [(0xFF, 'A'), (0x1F, 'voice'), (0xF8, 'hand')]
CLEARING = {'clear', 'and', 'nor', 'nand'}
def ham(a, b): return bin(a ^ b).count('1')
def nearest(tag, chans):
    best = None
    for i, t in enumerate(chans):
        if best is None or ham(tag, t) < ham(tag, chans[best]):
            best = i
    return best
def port(tag, ports):
    return min(ports, key=lambda p: ham(tag, p[0]))[1]

def parse_ckpt(path):
    islands = collections.OrderedDict()
    cur = None
    g = None
    for l in open(path, encoding='utf-8', errors='replace'):
        l = l.strip()
        if l.startswith('island '):
            cur = l.split()[1]
            islands[cur] = []
        elif l.startswith('genome'):
            g = {'chans': [], 'pops': []}
            islands[cur].append(g)
        elif l.startswith('channel'):
            _, tag, width, hold = l.split()
            g['chans'].append(int(tag))
        elif l.startswith('pop'):
            kv = dict(x.split('=', 1) for x in l.split()[1:] if '=' in x)
            g['pops'].append(kv)
    return islands

def wiring(g):
    """Each population's resolved source and destination (('c', i) a channel, ('p', name) a port)."""
    ch = g['chans']
    out = []
    for p in g['pops']:
        if p['from_port'] == '1':
            src = ('p', port(int(p['from']), INPUTS))
        else:
            k = nearest(int(p['from']), ch)
            src = ('c', k) if k is not None else None
        if p['to_port'] == '1':
            dst = ('p', port(int(p['to']), OUTPUTS))
        else:
            k = nearest(int(p['to']), ch)
            dst = ('c', k) if k is not None else None
        def gate(v):
            if v == '-':
                return None
            t, on = v.split(':')
            k = nearest(int(t), ch)
            return (('c', k), on == '1') if k is not None else None
        out.append(dict(p=p, src=src, dst=dst, op=p['op'].split(':')[0], gate=gate(p['gate']), pause=gate(p['pause'])))
    return out

def feeds_from(w, chan, sense, depth=3):
    """Whether channel `chan` is written, within `depth` hops, by a population reading the port
    `sense` (or all the senses, which include it)."""
    frontier = {chan}
    for _ in range(depth):
        nxt = set()
        for x in w:
            if x['dst'] in frontier and x['src'] is not None:
                if x['src'][0] == 'p' and x['src'][1] in (sense, 'S'):
                    return True
                if x['src'][0] == 'c':
                    nxt.add(x['src'])
        frontier = nxt
        if not frontier:
            break
    return False

print()
print('B. The elites (top 8 of each island) in each run\'s last checkpoint')
hdr = 'run island      pops  clearing-into-a-read-channel  competing-on-a-channel  run-while-gated  learn-gated(affect/internal)  repeat-births  replicating  marked'
print('   ' + hdr)
tot = collections.Counter()
per_island = collections.defaultdict(collections.Counter)
for r in RUNS:
    ck = parse_ckpt(f'runs/islands{r}/checkpoint.txt')
    for isl, pop in ck.items():
        c = collections.Counter()
        for g in pop[:8]:
            w = wiring(g)
            read = {x['src'] for x in w if x['src'] and x['src'][0] == 'c'}
            gated_on = {x['gate'][0] for x in w if x['gate']} | {x['pause'][0] for x in w if x['pause']}
            writers = collections.defaultdict(set)
            for x in w:
                if x['dst']:
                    writers[x['dst']].add('clear' if x['op'] in CLEARING else 'excite')
            for x in w:
                if x['src'] is None or x['dst'] is None:
                    continue
                c['pops'] += 1
                if x['op'] in CLEARING and (x['dst'] in read or x['dst'] in gated_on or (x['dst'][0] == 'p')):
                    c['inhib'] += 1
                if x['op'] in CLEARING and 'excite' in writers[x['dst']]:
                    c['compete'] += 1
                if x['pause']:
                    c['pause'] += 1
                if x['gate']:
                    c['gate'] += 1
                    if feeds_from(w, x['gate'][0], 'affect'):
                        c['gate_affect'] += 1
                    else:
                        c['gate_internal'] += 1
                if x['p'].get('repeat') == '1' and x['p'].get('grow') == '1':
                    c['repeat'] += 1
                if int(x['p'].get('replicate', '0')) > 0:
                    c['replicate'] += 1
                if int(x['p'].get('mark', '0')) > 0:
                    c['marked'] += 1
        n = max(c['pops'], 1)
        pct = lambda k: f"{100 * c[k] // n:3}%"
        print(f"   {r:>3} {isl:<11} {c['pops']:4}  {pct('inhib'):>28}  {pct('compete'):>22}  {pct('pause'):>15}  {pct('gate'):>6} ({c['gate_affect']}/{c['gate_internal']})                {pct('repeat'):>8}  {pct('replicate'):>11}  {pct('marked'):>6}")
        tot.update(c)
        per_island[isl].update(c)
print()
n = tot['pops']
print(f"   all: {n} populations; clearing into a read channel {tot['inhib']*100//n}%, clearing where another excites {tot['compete']*100//n}%, run-while gated {tot['pause']*100//n}%, learn-gated {tot['gate']*100//n}% (from affect {tot['gate_affect']}, internal {tot['gate_internal']}), repeat births {tot['repeat']*100//n}%, replicating {tot['replicate']*100//n}%")

# ---------- C: each island's final best genome against its last held-out reading ----------
print()
print("C. Each island's best genome (last checkpoint) against its last held-out score (share of goal)")
last = {}
for x in rows:
    last[(x[0], x[2])] = x
feats = collections.defaultdict(list)
for r in RUNS:
    ck = parse_ckpt(f'runs/islands{r}/checkpoint.txt')
    for isl, pop in ck.items():
        if (r, isl) not in last or not pop:
            continue
        s = last[(r, isl)][6]
        w = [x for x in wiring(pop[0]) if x['src'] and x['dst']]
        n = max(len(w), 1)
        read = {x['src'] for x in w if x['src'][0] == 'c'}
        writers = collections.defaultdict(set)
        for x in w:
            writers[x['dst']].add('clear' if x['op'] in CLEARING else 'excite')
        f = {
            'loops': last[(r, isl)][4],
            'clearing competes': sum(1 for x in w if x['op'] in CLEARING and 'excite' in writers[x['dst']]) / n,
            'run-while gated': sum(1 for x in w if x['pause']) / n,
            'learn-gated': sum(1 for x in w if x['gate']) / n,
            'replicating': sum(1 for x in w if int(x['p'].get('replicate', '0')) > 0) / n,
            'repeat births': sum(1 for x in w if x['p'].get('repeat') == '1' and x['p'].get('grow') == '1') / n,
        }
        for k, v in f.items():
            feats[k].append((v, s))
for k, v in feats.items():
    v.sort()
    has = [s for x, s in v if x > 0]
    no = [s for x, s in v if x == 0]
    if has and no:
        print(f'   {k:<18}: with it n {len(has):2} mean {sum(has)/len(has):+.2f}; without n {len(no):2} mean {sum(no)/len(no):+.2f}')
