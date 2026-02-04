import numpy as np
import pandas as pd


class GradMapperUtils(object):
    @staticmethod
    def cut_mapping(grad: np.ndarray, exponent: float = 1.0):
        grad_df = pd.Series(np.abs(grad))
        grad_rank = grad_df.rank(method='dense', pct=True).to_numpy()
        grad_rank = grad_rank ** exponent
        grad_rank = grad_rank / np.max(np.abs(grad_rank))
        grad_mapped = np.sign(grad) * grad_rank
        return grad_mapped
