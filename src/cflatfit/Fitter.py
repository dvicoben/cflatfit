import numpy as np
from iminuit import Minuit
from seaborn import heatmap
import matplotlib.pyplot as plt

from cflatfit.Parameter import Parameter, ParameterManager
from cflatfit.FoM import FigureOfMerit

from cflatfit.logger_config import make_logger
logger = make_logger(__name__)


def get_covariance_str(accurate: bool, forced_posdef: bool) -> str:
    if accurate:
        return "Accurate"
    if forced_posdef:
        return "Forced Pos-Def"
    return "Failed"


class imFitter:
    def __init__(self, 
                 FoM: FigureOfMerit):
        
        self._fom: FigureOfMerit = FoM
        self._m: Minuit = None

    @property
    def fom(self) -> FigureOfMerit: return self._fom
    @property
    def param_manager(self) -> ParameterManager: return self.fom.param_manager
    @property
    def m(self) -> Minuit: return self._m
    @property
    def param_list(self) -> list[str]: return self.fom.param_list

    def minuit_setup(self) -> None:
        vals = [self.param_manager.getVal(ipar) for ipar in self.param_list]
        self._m = Minuit(self.fom, vals)
        self.m.print_level = 2
        self.m.strategy = 2
        for i in range(len(self.param_list)):
            ipar = self.param_manager.getParam(self.param_list[i])
            self.m.limits[i] = (ipar.limlo, ipar.limhi)
            if ipar.fixed:
                self.m.fixto(i, ipar.val)
        
    def migrad(self, ncall: int = None) -> None:
        self.m.migrad(ncall=ncall)

    def hesse(self, ncall: int = None) -> None:
        self.m.hesse(ncall=ncall)

    def fit(self, ncall: int = None, sumW2error: bool = False) -> tuple[list[float], list[float]]:
        if not self.m:
            self.minuit_setup()
        self.migrad(ncall)
        self.hesse(ncall)
        
        res_list = [elem.value for elem in self.m.params]
        err = [elem.error for elem in self.m.params]
        if sumW2error:
            m2 = Minuit(self.fom.calc_nllW2, res_list)
            m2.hesse()
            c_inv = np.linalg.inv(m2.covariance)
            cov = np.matmul(self.m.covariance, np.matmul(c_inv, self.m.covariance))
            err = np.sqrt(np.diag(cov))

        # print("Function Minimum:", self.m.fval)
        # if self.m.valid:
        #     print("Fit converged")
        # else:
        #     print("Fit did not converge")

        # if self.m.accurate:
        #     print("Covariance matrix accurate")
        # else:
        #     print("Covariance matrix not accurate")
        self.fminlog()

        return res_list, err
    
    def fminlog(self) -> None:
        if not self.m:
            logger.info("No minuit object, please minimise first!")
            return
        
        fmin = self.m.fmin
        diag = [
            f"FMin: {fmin.fval} | Valid: {fmin.is_valid} | EDM: {self.m.fmin.edm}",
            f"Above EDM: {fmin.is_above_max_edm} | Call Limit: {fmin.has_reached_call_limit}",
            f"Covariance: {get_covariance_str(fmin.has_accurate_covar, fmin.has_made_posdef_covar)}",
        ]
        for ielem in diag:
            logger.info(ielem)


    def corr(self, savepath: str = None, parlist: list[str] = [], labels: list[str] = []) -> np.ndarray:
        corr = self.m.covariance.correlation()
        params_plot = self.param_list
        if len(parlist) > 0:
            paridx = [self.get_par_idx(ipar) for ipar in parlist]
            corr = corr[paridx]
            params_plot = parlist
        if savepath:
            fig, ax = plt.subplots(1, 1, figsize=(0.75*len(params_plot), 0.5*len(params_plot)))
            labs = labels if len(labels) > 0 else params_plot
            heatmap(data=corr, vmin=-1, vmax=1, annot=True, cmap='bwr', fmt=".3f", ax=ax,
                    xticklabels=labs, yticklabels=labs
                    )
            ax.tick_params(axis='x', labelrotation=90)
            fig.savefig(savepath, bbox_inches="tight")
            plt.close()
        return corr
    
    def get_par_idx(self, par: str) -> int:
        return int(np.where(np.array(self.param_list) == par)[0][0])

    def minos(self, par: str, print_level: int = 0) -> tuple[float, float]:
        plevel = self.m.print_level
        self.m.print_level = print_level
        paridx = self.get_par_idx(par)
        self.m.minos(paridx)
        self.m.print_level = plevel
        return self.m.params[paridx].merror

    def pull_hesse(self, par: str, trueval: float) -> float:
        paridx = self.get_par_idx(par)
        mpar = self.m.params[paridx]
        val = mpar.value
        err = mpar.error
        return (val - trueval) / err
    
    def pull_minos(self, par: str, trueval: float) -> float:
        paridx = self.get_par_idx(par)
        mpar = self.m.params[paridx]
        val = self.m.params[paridx].value
        errs = mpar.merror
        if mpar.merror is None:
            errs = self.minos(par)
        errs = np.abs(errs)
        errlo = errs[0]
        errhi = errs[1]

        pull = 0.0
        if trueval > val:
            pull = (val - trueval) / errhi
        if trueval < val:
            pull = (val - trueval) / errlo
        
        return pull
    
    def param_cov(self, pars: list[str]) -> np.ndarray:
        paridx = [self.get_par_idx(ipar) for ipar in pars]
        cov = self.m.covariance
        return cov[paridx]

    def param_corr(self, pars: list[str]) -> np.ndarray:
        paridx = [self.get_par_idx(ipar) for ipar in pars]
        corr = self.m.covariance.correlation()
        return corr[paridx]