import json
from pathlib import Path
from scipy.special import expit
import numpy as np
from starter.click_model import fit
from starter.slates import recommend
p=Path(__file__).parent;t=json.loads((p/'fixtures/train.json').read_text());a=json.loads((p/'fixtures/allocation.json').read_text())
model=fit(t['rows'],t['dimension'],t['ridge']);v=expit(np.asarray(a['features'])@np.array(model['w']))
print(json.dumps({'ordinary_click_fit':model,'independent_lists':recommend(v,a['slots'])}))
