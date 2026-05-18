import numpy as np

import cflatfit.CostFuncMath as cf
from cflatfit.FoM import FigureOfMerit
from cflatfit.logger_config import make_logger
logger = make_logger(__name__)



class NLL(FigureOfMerit):
    def __init__(self, ignore_constraint: bool = False):
        super().__init__(ignore_constraint)
    
    def evaluate(self) -> float:
        return self.nll()
    

class NLLScaled(FigureOfMerit):
    def __init__(self, ignore_constraint: bool = False):
        super().__init__(ignore_constraint)
    
    def evaluate(self) -> float:
        fom = [cf.poisson_scaled_chi2(ich.data, ich.dataVar, ich.get_model()) for ich in self.channels.values()]
        return np.sum(fom) + self.get_constraint_term()

