from __future__ import annotations
import numpy as np
# from hammer.hammerlib import Hammer, IOBuffer, RecordType

from cflatfit.Templates.modifiers import ModInterpSys
from cflatfit.Parameter import ParameterManager, Parameter, ConstraintType
from cflatfit.logger_config import make_logger
logger = make_logger(__name__)

        

class Template:
    def __init__(self,
                 name: str,
                 N: np.ndarray,
                 Nerr: np.ndarray,
                 isnorm: bool) -> None:
        self._name            : str                = name
        self._label           : str                = name
        self._shape           : tuple[int]         = None
        self._N               : np.ndarray         = None
        self._Nerr            : np.ndarray         = None
        self._isnorm          : bool               = isnorm
        self._param_manager   : ParameterManager   = ParameterManager()
        self._constraint      : ConstraintType     = ConstraintType.NONE
        self._additive_mod    : list[ModInterpSys] = []
        self._nuisance_params : list[str]          = []
        self._has_variance    : bool               = False
        self.set_templates(N, Nerr)

    @property
    def name(self) -> str: return self._name
    @property
    def label(self) -> str: return self._label
    @property
    def shape(self) -> tuple[int]: return self._shape
    @property
    def N(self) -> np.ndarray: return self._N
    @property
    def Nerr(self) -> np.ndarray: return self._Nerr
    @property
    def param_manager(self) -> ParameterManager: return self._param_manager
    @property
    def additive_mod(self) -> list[ModInterpSys]: return self._additive_mod
    @property
    def constraint(self) -> ConstraintType: return self._constraint
    @property
    def nuisance_params(self) -> list[str]: return self._nuisance_params
    
    def set_label(self, label: str) -> None:
        self._label = label

    def set_param_manager(self, param_manager: ParameterManager) -> None:
        self._param_manager = param_manager
    
    def set_templates(self, N: np.ndarray, Nerr: np.ndarray) -> None:
        N = np.array(N)
        # if np.sum(N) < 1e-3:
        #     self._isnull = True
        #     logger.warning(f"Template {self.name} is null")
        self._shape = np.shape(N)
        self._N = N.ravel()
        if Nerr is not None:
            Nerr = np.array(Nerr)
            if Nerr.shape != self.shape:
                logger.error(f"N {self.shape} and Nerr {Nerr.shape} have different shapes")
                raise IndexError(f"N {self.shape} and Nerr {Nerr.shape} have different shapes")
            self._Nerr = Nerr.ravel()
        else:
            logger.info("NOTE: No template errors provided, assuming poisson")
            self._Nerr = np.sqrt(self.N)
    
    def add_interpsys(self, modifier: ModInterpSys, nupar: Parameter, 
                      h_min: np.ndarray, h_max: np.ndarray) -> None:
        if nupar.constrainttype == ConstraintType.NONE:
            nupar.setlim(-5, 5)
            nupar.constrain_gauss(0.0, 1.0)
        self.param_manager.addParam(nupar)
        mod = modifier(nupar.name, h_min, h_max)
        self.additive_mod.append(mod)

    def get_additive_mod(self, hnom: np.ndarray) -> np.ndarray:
        delta = np.zeros_like(hnom)
        for imod in self.additive_mod:
            delta += imod.delta(hnom, self.param_manager)
        return delta
    
    def create_nuisance(self, ibin: int) -> None:
        # Implement in derived classes
        return

    def create_all_nuisance(self) -> None:
        for ibin in range(len(self.N)):
            self.create_nuisance(self, ibin)

    def update_nuisance(self, ibin: int):
        # Implement in derived classes
        return
    
    def update_all_nuisance(self, force: bool = False) -> None:
        if not self._has_variance and not force:
            return
        
        for ibin in range(self.N):
            self.update_nuisance(ibin)

    def get_nuisance_hist(self) -> np.ndarray:
        h = [self.param_manager.getVal(inu) for inu in self.nuisance_params]
        return np.array(h)
    
    def vary_hist(self) -> None:
        # For histograms where the values are allowed to fluctuate, override in derived class
        # This should set both self.N and self.Nerr when you vary the histogram
        return
    
    def fluctuate_template(self, rng: np.random.Generator) -> None:
        if not self._isnorm:
            N = rng.poisson(self.N).astype(float)
            self.set_templates(N, np.sqrt(N))
        else:
            vals = rng.normal(self.N, self.Nerr)
            vals[vals <= 0.0] = 1.e-10
            self.set_templates(vals, self.Nerr)
        self.update_all_nuisance(force = True)

    def calc_hist(self) -> tuple[np.ndarray, float]:
        self.vary_hist()
        self.update_all_nuisance()
        hist = self.N + self.get_additive_mod(self.N)
        norm = np.sum(hist)        
        return hist, norm
    
    def get_raw_norm(self) -> np.ndarray:
        arr = np.zeros_like(self.N)
        ma = self.N > 0
        arr[ma] = self.N[ma]
        return arr/np.sum(self.N)
    
    def get_raw_normerr(self) -> np.ndarray:
        arr = np.zeros_like(self.Nerr)
        ma = self.N > 0
        arr[ma] = self.Nerr[ma]/self.N[ma]
        return arr/np.sum(self.N)

    def H(self) -> np.ndarray:
        hist, norm = self.calc_hist()
        return hist
    
    def Herr(self) -> np.ndarray:
        h, norm = self.calc_hist()
        scale = np.ones_like(self.N)
        ma = self.N > 0.0
        scale[ma] = h[ma]/self.N[ma]
        return self.Nerr*scale

    def h(self) -> np.ndarray:
        h, norm = self.calc_hist()
        return h/norm

    def herr(self) -> np.ndarray:
        # if self._isnull:
        #     return np.zeros_like(self.N)
        h, norm = self.calc_hist()
        scale = np.ones_like(self.N)
        ma = self.N > 0.0
        # Not sure whether one should scale the errors with the nuisance parameter
        scale[ma] = h[ma]/self.N[ma]
        return self.Nerr*scale/(norm)

    def NormH(self) -> float:
        h, norm = self.calc_hist()
        return norm

    def SumH(self) -> float:
        return np.sum(self.H())
    
    def unrolled(self) -> np.ndarray:
        return self.N
    
    def unrollederr(self) -> np.ndarray:
        return self.Nerr




class TemplateBB(Template):
    def __init__(self,
                 name: str,
                 N: np.ndarray,
                 Nerr: np.ndarray,
                 isnorm: bool) -> None:
        if isnorm:
            logger.error(f"Cannot use {ConstraintType.POISSON} constraint if the template is normalised!")
            raise ValueError(f"Cannot use {ConstraintType.POISSON} constraint if the template is normalised!")
        super().__init__(name, N, Nerr, isnorm)
        self._constraint = ConstraintType.POISSON

    def create_nuisance(self, ibin: int) -> None:
        bin_N = self.N[ibin]
        xi = Parameter(f"{self.name}_nuisance[{ibin}]", bin_N, limlo = max(0.0, bin_N-10.*np.sqrt(bin_N)), limhi = bin_N+10.*np.sqrt(bin_N))
        xi.constrain_poisson(bin_N)
        xi.use()
        self.param_manager.addParam(xi)
        self.nuisance_params.append(xi.name)
    
    def update_nuisance(self, ibin) -> None:
        par = self.nuisance_params[ibin]
        val = self.N[ibin]
        self.param_manager.params[par].constrain_poisson(val)

    def calc_hist(self) -> tuple[np.ndarray, float]:
        self.vary_hist()
        self.update_all_nuisance()
        hist = self.get_nuisance_hist()
        hist += self.get_additive_mod(self.N)
        norm = np.sum(hist)        
        return hist, norm





class TemplateGauss(Template):
    def __init__(self,
                 name: str,
                 N: np.ndarray,
                 Nerr: np.ndarray,
                 isnorm: bool) -> None:
        super().__init__(name, N, Nerr, isnorm)
        self._constraint = ConstraintType.GAUSS
    
    def create_nuisance(self, ibin: int) -> None:
        bin_N = self.N[ibin]
        bin_err = self.Nerr[ibin]
        sigma = bin_err/bin_N if bin_N > 0. else 1.
        xi = Parameter(f"{self.name}_nuisance[{ibin}]", 1.0, limlo = max(0.0, 1.-10*sigma), limhi = 1.+10.*sigma)
        xi.constrain_gauss(1.0, sigma)
        if bin_N <= 0.0 or sigma > 1.0:
            logger.warning(f"Template {self.name} Invalid yield in bin {ibin} (N={bin_N}), {xi.name} is fixed to 1.0")
            xi.fix(1.0)
        self.param_manager.addParam(xi)
        self.nuisance_params.append(xi.name)
    
    def update_nuisance(self, ibin) -> None:
        par = self.nuisance_params[ibin]
        val = self.Nerr[ibin]/self.N[ibin]
        self.param_manager.params[par].constrain_gauss(1.0, val)

    def calc_hist(self) -> tuple[np.ndarray, float]:
        self.vary_hist()
        self.update_all_nuisance()
        hist = self.N + self.get_additive_mod(self.N)
        hist = self.get_nuisance_hist()*hist
        norm = np.sum(hist)
        return hist, norm