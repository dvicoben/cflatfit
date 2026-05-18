from __future__ import annotations
import numpy as np
import copy

import cflatfit.CostFuncMath as cf
import cflatfit.Parameter as mp
from cflatfit.PDFBase import PDFBase
from cflatfit.FitChannel import Channel

from cflatfit.logger_config import make_logger
logger = make_logger(__name__)


class FigureOfMerit:
    def __init__(self, ignore_constraint: bool = False):
        self._channels: dict[str, Channel] = {}
        self._param_manager: mp.ParameterManager = mp.ParameterManager()
        self._ignore_constraint: bool = ignore_constraint
        self._offset: float = 0.0

    @property
    def channels(self) -> dict[str, Channel]: return self._channels
    @property
    def param_manager(self) -> mp.ParameterManager: return self._param_manager
    @property
    def params(self) -> dict[str, mp.Parameter]: return self.param_manager.params
    @property
    def ignore_constraint(self) -> bool: return self._ignore_constraint
    @property
    def param_list(self) -> list[str]: return self.param_manager.in_use
    @property
    def offset(self) -> float: return self._offset

    def add_channel(self, channel: Channel) -> None:
        if channel.name in self.channels:
            logger.error(f"Channel {channel.name} already in FoM")
            raise KeyError(f"Channel {channel.name} already in FoM")
        logger.info(f"Adding Channel {channel.name}")
        self.channels[channel.name] = channel
        logger.info("Updating FoM parameter manager")
        self.param_manager.merge(channel.pdf.param_manager)
        logger.info("Setting manager for all channels")
        for ichname, ich in self.channels.items():
            ich.pdf.set_param_manager(self.param_manager)

    def create_channel(self, 
                       name: str, 
                       pdf: PDFBase, 
                       data: np.ndarray, 
                       dataErr: np.ndarray = None,
                       normPDF_before_eval: bool = False,
                      ) -> None:
        ch = Channel(name=name, pdf=pdf, 
                     data=data, dataErr=dataErr,
                     normPDF_before_eval=normPDF_before_eval)
        self.add_channel(ch)

    def setVal(self, parname_man: str, val: float):
        self.param_manager.setVal(parname_man, val)

    def setVals(self, pars_man: dict[str, float]):
        self.param_manager.setVals(pars_man)

    def updateVals(self, paramvals: list[float]):
        params = {parname : parval for parname, parval in zip(self.param_list, paramvals)}
        self.param_manager.setVals(params)

    def set_offset(self, val: float):
        self._offset = val
    
    def set_offset_eval(self):
        self._offset = -self.evaluate()

    def get_constraint_term(self) -> float:
        """
        Fetch ALL constrained parameters in the manager and calculate their LLH constraint term.
        This is done at the FoM level rather than channel level to ensure constraint is only added once
        if shared by multiple channels.
        
        Calculation here will add constraint terms for parametrs that are not in use as well. This is to 
        include a constraint (e.g. sum of bkg yields) that the PDF is not directly parameterised by.
        Note that in principle this means random constrained parameters that have nothing to do with the fit
        also contribute. This is not an issue as they would be unchanged and contribute a constant only
        """
        if self.ignore_constraint:
            return 0.0

        consterm = 0.0
        for iparname, ipar in self.params.items():
            if ipar.constrain:
                consterm += ipar.get_constraint()
        return consterm
    
    def __call__(self, paramvals: list[float]) -> float:
        self.updateVals(paramvals)
        return self.evaluate() + self.offset
    
    def evaluate(self):
        # return self.nll()
        raise Exception("FOM evaluate specified in derived class")

    def nll(self):
        fom = [cf.poisson_chi2(ich.data, ich.get_model()) for ich in self.channels.values()]
        return np.sum(fom) + self.get_constraint_term()
    
    def nllW2(self):
        fom = [cf.poisson_chi2_W2(ich.data, ich.get_model(), ich.scaleFactorW2) for ich in self.channels.values()]
        return np.sum(fom) + self.get_constraint_term()
    
    def calc_nllW2(self, paramvals: list[float]):
        self.updateVals(paramvals)
        return self.nllW2()

    def chi2(self):
        chi2 = [np.sum((ich.data - ich.get_model())**2 / (ich.dataVar)) for ich in self.channels.values()]
        return np.sum(chi2) + self.get_constraint_term()
    
    def generate(self, rng: np.random.Generator = np.random.default_rng(), scale_err: float = 1.) -> FigureOfMerit:
        newFOM = copy.deepcopy(self)
        other = copy.deepcopy(self)
        toydata = {}
        for ichannel in other.channels.values():
            itoy = ichannel.get_model()
            itoy = rng.poisson(itoy).astype(float)
            itoyerr = np.ones_like(itoy)
            itoyerr[itoy > 0.0] = np.sqrt(itoy[itoy > 0.0])
            toydata[ichannel.name] = (itoy, scale_err*itoyerr)
        
        for ich, idat in toydata.items():
            newFOM.channels[ich].set_data(*idat)
            for ipar in newFOM.param_manager.params.values():
                ipar.set_resample_constraint(rng)
        
        return newFOM



