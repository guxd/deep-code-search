originfile='/data/use.codevecs.h5'
convertedfile='/data/use.codevecs.normalized.h5'

import tables 
import utils
import numpy as np
from utils import normalize
h5f = tables.open_file(originfile)
vecs= h5f.root.vecs
vecs= normalize(vecs)
            
            
            
npvecs=np.array(vecs)
fvec = tables.open_file(convertedfile, 'w')
atom = tables.Atom.from_dtype(npvecs.dtype)
filters = tables.Filters(complib='blosc', complevel=5)
ds = fvec.create_carray(fvec.root, 'vecs', atom, npvecs.shape,filters=filters)
ds[:] = npvecs
fvec.close()
