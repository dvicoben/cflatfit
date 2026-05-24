import numpy as np

from cflatfit.Parameter import ParameterManager

class ModInterpSys:
    def __init__(self, param: str, hmin: np.ndarray, hmax: np.ndarray):
        self._param : str        = param
        self._hmin  : np.ndarray = hmin
        self._hmax  : np.ndarray = hmax
    
    @property
    def hmin(self) -> np.ndarray: return self._hmin
    @property
    def hmax(self) -> np.ndarray: return self._hmax

    def delta(self, hnom: np.ndarray, param_manager: ParameterManager) -> np.ndarray:
        raise NotImplementedError("To implement in derived class")



class ModLinearInterpSys(ModInterpSys):
    def __init__(self, param: str, hmin: np.ndarray, hmax: np.ndarray):
        super().__init__(param, hmin, hmax)

    def delta(self, hnom: np.ndarray, param_manager: ParameterManager) -> np.ndarray:
        alpha  = param_manager.getVal(self._param)
        
        if alpha >= 0:
            return alpha*(self.hmax - hnom)
        return alpha*(hnom - self.hmin)
    


class ModQuadInterpSys(ModInterpSys):
    def __init__(self, param: str, hmin: np.ndarray, hmax: np.ndarray):
        super().__init__(param, hmin, hmax)
    
    def delta(self, hnom: np.ndarray, param_manager: ParameterManager) -> np.ndarray:
        alpha  = param_manager.getVal(self._param)
        a = 0.5*(self.hmax + self.hmin) - hnom
        b = 0.5*(self.hmax - self.hmin)

        if alpha > 1:
            return (alpha - 1.)*(b + 2*a)
        if alpha < 1:
            return (alpha + 1.)*(b - 2*a)
        return a*alpha*alpha + b*alpha



class ModExpInterpSys(ModInterpSys):
    def __init__(self, param: str, hmin: np.ndarray, hmax: np.ndarray):
        super().__init__(param, hmin, hmax)
    
    def delta(self, hnom: np.ndarray, param_manager: ParameterManager) -> np.ndarray:
        alpha  = param_manager.getVal(self._param)
        mult = 1.
        if alpha >= 0.:
            mult = (self.hmax/hnom)**alpha
        if alpha < 0.:
            mult = (self.hmin/hnom)**(-alpha)
        
        return hnom*(mult - 1.)
