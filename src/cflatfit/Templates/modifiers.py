import numpy as np

from cflatfit.Parameter import ParameterManager

class ModLinearSys:
    def __init__(self, param: str, hmin: np.ndarray, hmax: np.ndarray):
        self._param : str        = param
        self._hmin  : np.ndarray = hmin
        self._hmax  : np.ndarray = hmax
    
    @property
    def hmin(self) -> np.ndarray: return self._hmin
    @property
    def hmax(self) -> np.ndarray: return self._hmax

    def delta(self, hnom: np.ndarray, param_manager: ParameterManager) -> np.ndarray:
        alpha  = param_manager.getVal(self._param)
        
        if alpha >= 0:
            return alpha*(self.hmax - hnom)
        return alpha*(hnom - self.hmin)