
import numpy as np

class Normalizer:

    def normalizeL1(X_imgs):
        return (255. - X_imgs.astype(np.float32)) / 255.

    normalizeL1 = staticmethod(normalizeL1)

