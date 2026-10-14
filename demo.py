"""Seeded synthetic traffic, explicitly not real customers or production telemetry."""
import random
import math

def make_data(n=400,shift=1.0,seed=42):
    rng=random.Random(seed)
    def batch(offset):
        rows=[]
        for i in range(n):
            x=rng.gauss(offset,1)
            z=rng.gauss(0,1)
            score=1/(1+math.exp(-(1.2*x+0.4*z)))
            # Current outcomes change independently of the model: concept shift.
            truth=1/(1+math.exp(-(1.2*x+0.4*z-offset)))
            rows.append({'label':int(rng.random()<truth),'score':score,
                         'segment':'mobile' if i%2 else 'desktop',
                         'features':{'signal':x,'context':z}})
        return rows
    return batch(0),batch(shift)
