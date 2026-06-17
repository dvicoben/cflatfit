import numpy as np
import copy

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

    def generate(self, rng: np.random.Generator = np.random.default_rng(), scale_err: float = 1.) -> FigureOfMerit:
        newFOM = copy.deepcopy(self)
        other = copy.deepcopy(self)
        toydata = {}
        for ichannel in other.channels.values():
            # Scaled Poisson distr, so scale to effective stats sum(weights)/sum(weights**2) 
            t = ichannel.data / ichannel.dataVar
            itoy = ichannel.get_model()
            itoy = rng.poisson(t*itoy).astype(float)
            itoyerr = np.ones_like(itoy)
            itoyerr[itoy > 0.0] = np.sqrt(itoy[itoy > 0.0])
            itoy = itoy/t
            itoyerr = itoyerr/t
            toydata[ichannel.name] = (itoy, scale_err*itoyerr)
        
        for ich, idat in toydata.items():
            newFOM.channels[ich].set_data(*idat)
            for ipar in newFOM.param_manager.params.values():
                ipar.set_resample_constraint(rng)
        
        return newFOM

