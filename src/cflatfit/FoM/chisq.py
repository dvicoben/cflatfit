import numpy as np
import copy

import cflatfit.CostFuncMath as cf
from cflatfit.FoM import FigureOfMerit
from cflatfit.FoM.utils import has_template_with_cons
from cflatfit.logger_config import make_logger
logger = make_logger(__name__)

class Chi2(FigureOfMerit):
    def __init__(self, ignore_constraint: bool = False):
        super().__init__(ignore_constraint)
    
    def evaluate(self) -> float:
        return self.chi2()
    

class Chi2DA(FigureOfMerit):
    def __init__(self, ignore_constraint: bool = False):
        super().__init__(ignore_constraint)

        if has_template_with_cons(self):
            logger.error(f"Chi2DA is a Barlow-Beeston lite figure of merit, cannot use with shape constraints in templates")
    
    def chi2DA(self):
        fom = [cf.chi2_template_DA(ich.data, ich.dataVar, ich.get_model(), ich.get_model_err()**2) for ich in self.channels.values()]
        return np.sum(fom) + self.get_constraint_term()
    
    def evaluate(self) -> float:
        return self.chi2DA()
    
    def generate(self, rng: np.random.Generator = np.random.default_rng()) -> FigureOfMerit:
        logger.warning("Generating for BB-lite method, not sure that things are properly fluctuated for the toy. Consider bootstrapping samples instead!")
        newFOM = copy.deepcopy(self)
        other = copy.deepcopy(self)
        toydata = {}
        for ichannel in other.channels.values():
            for itempl in ichannel.templates.values():
                itempl.fluctuate_template(rng)
            
            # for iyield in ichannel.pdf.get_yield_parameters().values():
            #     iyield.setVal(rng.poisson(iyield.getVal()))
            itoy = ichannel.get_model()
            itoy = rng.poisson(itoy).astype(float)
            itoyerr = np.ones_like(itoy, dtype=float)
            itoyerr[itoy > 0.0] = np.sqrt(itoy[itoy > 0.0])
            toydata[ichannel.name] = (itoy, itoyerr)
        
        for ich, idat in toydata.items():
            newFOM.channels[ich].set_data(*idat)
            for ipar in newFOM.param_manager.params.values():
                ipar.set_resample_constraint()
        
        return newFOM


class Chi2Conway(FigureOfMerit):
    def __init__(self, ignore_constraint: bool = False):
        super().__init__(ignore_constraint)

        if has_template_with_cons(self):
            logger.error(f"Chi2Conway is a Barlow-Beeston lite figure of merit, cannot use with shape constraints in templates")
    
    def chi2Conway(self):
        fom = [cf.chi2_template_JSC(ich.data, ich.get_model(), ich.get_model_err()**2) for ich in self.channels.values()]
        return np.sum(fom) + self.get_constraint_term()

    def evaluate(self) -> float:
        return self.chi2Conway()
    
    def generate(self, rng: np.random.Generator = np.random.default_rng()) -> FigureOfMerit:
        logger.warning("Generating for BB-lite method, not sure that things are properly fluctuated for the toy. Consider bootstrapping samples instead!")
        newFOM = copy.deepcopy(self)
        other = copy.deepcopy(self)
        toydata = {}
        for ichannel in other.channels.values():
            for itempl in ichannel.templates.values():
                itempl.fluctuate_template(rng)
            
            itoy = ichannel.get_model()
            itoy = rng.poisson(itoy).astype(float)
            itoyerr = np.ones_like(itoy)
            itoyerr[itoy > 0.0] = np.sqrt(itoy[itoy > 0.0])
            toydata[ichannel.name] = (itoy, itoyerr)
        
        for ich, idat in toydata.items():
            newFOM.channels[ich].set_data(*idat)
            for ipar in newFOM.param_manager.params.values():
                ipar.set_resample_constraint()
        
        return newFOM