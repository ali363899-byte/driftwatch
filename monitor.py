"""Dependency-free classification monitoring with explicit statistical limits."""
import math
import bisect
from collections import defaultdict

def numeric(values):
    result=[float(v) for v in values]
    if not result or any(not math.isfinite(v) for v in result):
        raise ValueError('Need non-empty finite numeric observations')
    return result

def drift(reference,current,bins=10):
    reference=sorted(numeric(reference)); current=numeric(current)
    if not 2<=bins<=50:
        raise ValueError('bins must be between 2 and 50')
    edges=sorted(set(reference[min(len(reference)-1,int(i*len(reference)/bins))] for i in range(1,bins)))
    if reference[0] == reference[-1]:
        edges=[reference[0],math.nextafter(reference[0],math.inf)]
    def histogram(values):
        counts=[0]*(len(edges)+1)
        for v in values: counts[bisect.bisect_right(edges,v)]+=1
        return counts
    a,b=histogram(reference),histogram(current)
    # Laplace smoothing avoids zero buckets while preserving normalized probabilities.
    pa=[(v+0.5)/(len(reference)+0.5*len(a)) for v in a]
    pb=[(v+0.5)/(len(current)+0.5*len(b)) for v in b]
    psi=sum((q-p)*math.log(q/p) for p,q in zip(pa,pb))
    return {'psi':psi,'reference_counts':a,'current_counts':b,'edges':edges,
            'reference_n':len(reference),'current_n':len(current)}

def classification(rows,threshold=0.5):
    if not 0<=threshold<=1 or not rows:
        raise ValueError('Need rows and threshold within [0,1]')
    tp=tn=fp=fn=0; brier=0; calibration=[[] for _ in range(10)]
    for row in rows:
        y=row['label']; p=float(row['score'])
        if y not in (0,1) or not math.isfinite(p) or not 0<=p<=1:
            raise ValueError('Labels must be binary and scores finite probabilities')
        predicted=p>=threshold
        tp+=int(predicted and y==1); tn+=int(not predicted and y==0)
        fp+=int(predicted and y==0); fn+=int(not predicted and y==1)
        brier+=(p-y)**2
        calibration[min(9,int(p*10))].append((p,y))
    divide=lambda a,b:a/b if b else None
    points=[{'n':len(c),'confidence':sum(p for p,y in c)/len(c),
             'observed':sum(y for p,y in c)/len(c)} for c in calibration if c]
    return {'n':len(rows),'accuracy':(tp+tn)/len(rows),'precision':divide(tp,tp+fp),
            'recall':divide(tp,tp+fn),'f1':divide(2*tp,2*tp+fp+fn),
            'brier':brier/len(rows),'confusion':{'tp':tp,'tn':tn,'fp':fp,'fn':fn},
            'ece':sum(p['n']/len(rows)*abs(p['confidence']-p['observed']) for p in points),
            'calibration':points}

def report(reference,current,threshold=0.5,alert=0.2,min_slice=20):
    if not 0<=alert<=10 or min_slice<1:
        raise ValueError('Invalid alert or minimum slice size')
    base=classification(reference,threshold); now=classification(current,threshold)
    features=set(reference[0].get('features',{}))
    if not features or any(set(r.get('features',{}))!=features for r in reference+current):
        raise ValueError('All rows require identical non-empty feature keys')
    shifts={name:drift([r['features'][name] for r in reference],[r['features'][name] for r in current]) for name in sorted(features)}
    groups=defaultdict(list)
    for r in current: groups[str(r.get('segment','unspecified'))].append(r)
    slices={g:dict(classification(rs,threshold),low_sample=len(rs)<min_slice) for g,rs in sorted(groups.items())}
    return {'reference':base,'current':now,'features':shifts,'segments':slices,
            'alerts':[f'{name}: PSI exceeds {alert}' for name,v in shifts.items() if v['psi']>alert],
            'threshold':threshold,'psi_alert':alert,
            'limitations':'Synthetic demo is not production evidence. PSI thresholds are heuristics; slice comparisons are descriptive, not a fairness certification.'}
