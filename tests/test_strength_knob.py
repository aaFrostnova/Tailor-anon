import numpy as np
from PIL import Image
import sys; sys.path.insert(0, ".")

def _scale_resid(cover, full, a):
    c=np.asarray(cover,np.float32); w=np.asarray(full,np.float32)
    return Image.fromarray(np.clip(c + a*(w-c),0,255).astype(np.uint8))

def test_scale_resid_endpoints():
    cover=Image.fromarray((np.ones((16,16,3))*100).astype(np.uint8))
    full =Image.fromarray((np.ones((16,16,3))*160).astype(np.uint8))
    assert np.array_equal(np.asarray(_scale_resid(cover,full,1.0)), np.asarray(full))      # a=1 -> full
    assert np.array_equal(np.asarray(_scale_resid(cover,full,0.0)), np.asarray(cover))     # a=0 -> cover
    mid=np.asarray(_scale_resid(cover,full,0.5)); assert abs(mid.mean()-130)<1.0            # halfway
