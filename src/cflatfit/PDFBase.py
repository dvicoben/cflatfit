import numpy as np
import matplotlib.pyplot as plt

from cflatfit.Parameter import Parameter, ParameterManager
from cflatfit.Templates import Template

from cflatfit.logger_config import make_logger
logger = make_logger(__name__)


class PDFBase:
    def __init__(self, 
                 name: str,
                 param_manager: ParameterManager | None = None,
                 params_list: list[str] | None = None,
                ):
        
        self._name: str = name
        self._params: dict[str, str] = {ipar : None for ipar in params_list} if params_list is not None else {}
        self._param_manager: ParameterManager = param_manager if param_manager is not None else ParameterManager()
        self._templates: dict[str, Template] = {}
        self._external_constraints: list[str] = []
        self._poi_int: str = None

    @property
    def name(self) -> str: return self._name
    @property
    def param_manager(self) -> ParameterManager: return self._param_manager
    @property
    def params(self) -> dict[str, str]: return self._params
    @property
    def params_list(self) -> list[str]: return list(self.params.keys())
    @property
    def templates(self) -> dict[str, Template]: return self._templates
    @property
    def poi_int(self) -> str: return self._poi_int
    @property
    def poi_ext(self) -> str: return self.params[self.poi_int]
    @property
    def external_constraints(self) -> list[str]: return self._external_constraints

    def set_param_manager(self, 
                          param_manager: ParameterManager,
                          propagate_to_templates: bool = True):
        self._param_manager = param_manager
        if propagate_to_templates:
            for itemp in self.templates.values():
                itemp.set_param_manager(param_manager)

    def set_param(self, parname: str, par: Parameter) -> None:
        if par.name not in self.param_manager.params:
            logger.info(f"{par} not in manager, adding to manager")
            self.param_manager.addParam(par)
        
        if self.params[parname] is not None:
            logger.info(f"{parname} already linked to {self.params[parname]}, changing to {par.name}")
        self.link_param(parname, par.name)

    def set_params(self, param_dict: dict[str, Parameter]) -> None:
        for iparname, ipar in param_dict.items():
            self.set_param(iparname, ipar)

    def link_param(self, name_int: str, name_manager: str) -> None:
        if name_int not in self.params_list:
            logger.error(f"{name_int} is not a PDF parameter")
            raise KeyError(f"{name_int} is not a PDF parameter")
        self._params[name_int] = name_manager
    
    def link_params(self, link_dict: dict[str, str] = None) -> None:
        if link_dict is None:
            for ipar in self.params_list:
                self.link_param(ipar, ipar)
            return
        
        for ipar in link_dict:
            self.link_param(ipar, link_dict[ipar])

    def set_templates(self, templates: list[Template]) -> None:
        self._templates = {itemp.name : itemp for itemp in templates}
        for itemplate in self.templates.values():
            self.param_manager.merge(itemplate.param_manager)
            itemplate.set_param_manager(self.param_manager)

    def set_poi(self, int_name: str) -> None:
        if int_name not in self.params:
            logger.error(f"POI {int_name} not a (internal) PDF parameter!")
            raise KeyError("POI not a (internal) PDF parameter!")
        self._poi_int = int_name

    def add_external_constraint(self, par: Parameter) -> None:
        self.param_manager.addParam(par)
        self.external_constraints.append(par)

    def getParam(self, parname: str) -> Parameter:
        parname_man = self.params[parname]
        return self.param_manager.getParam(parname_man)

    def getVal(self, parname: str) -> float:
        parname_man = self.params[parname]
        return self.param_manager.getVal(parname_man)
    
    def setVal(self, parname: str, val: float) -> None:
        parname_man = self.params[parname]
        self.param_manager.setVal(parname_man, val)

    def setVals(self, pars: dict[str, float]) -> None:
        pars_man = {self.params[k] : pars[k] for k in pars}
        self.param_manager.setVals(pars_man)

    def get_yield_parameters(self) -> dict[str, Parameter]:
        raise NotImplementedError("Yield to template correspondence/parameterisation implemented in child class!")

    def get_yields(self) -> dict[str, float]:
        return {kname : kpar.getVal() for kname, kpar in self.get_yield_parameters().items()}

    def pdf(self) -> np.ndarray:
        N = self.get_yields()
        comps = {k : self.templates[k].h()*N[k] for k in N}
        return np.sum([icomp for icomp in comps.values()], axis=0)

    # def fluct_pdf(self, rng: np.random.Generator) -> np.ndarray:
    #     N = self.get_yields()
    #     comps = {k : rng.poisson(self.templates[k].h()*N[k]) for k in N}
    #     return np.sum([icomp for icomp in comps.values()], axis=0)

    def pdf_var(self) -> np.ndarray:
        N = self.get_yields()
        comps = {k : self.templates[k].herr()*N[k] for k in N}
        return np.sum([icomp**2 for icomp in comps.values()], axis=0)

    def pdf_err(self) -> np.ndarray:
        return np.sqrt(self.pdf_var())

    def normpdf(self) -> np.ndarray:
        pdf = self.pdf()
        return pdf/np.sum(pdf)
    
    def normpdf_err(self) -> np.ndarray:
        pdferr = self.pdf_err()
        Ntot = np.sum([iN for iN in self.get_yields()])
        return pdferr/Ntot

    def plot_templates(self, bin_edges: list[float], colors: dict[str, str]) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(1, 1)
        for icomp in self.get_yield_parameters():
            itemplate = self.templates[icomp]
            h = itemplate.h()
            herr = itemplate.herr()
            if icomp in colors:
                stair = ax.stairs(h, bin_edges, label = itemplate.label, color=colors[icomp], linewidth=2) 
            else:
                stair = ax.stairs(h, bin_edges, label = itemplate.label, linewidth=2) 
            stair_err = ax.stairs(h+herr, bin_edges, baseline=h-herr, 
                                  alpha=0.3, color=stair.get_edgecolor(), fill=True)
        ax.set(xlim=(bin_edges[0], bin_edges[-1]))
        # ax.legend()
        return fig, ax
    
    def plot_templates_raw(self, bin_edges: list[float]) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(1, 1)
        for icomp in self.get_yield_parameters():
            itemplate = self.templates[icomp]
            h = itemplate.get_raw_norm()
            herr = itemplate.get_raw_normerr()
            stair = ax.stairs(h, bin_edges, label = itemplate.label, linewidth=2) 
            stair_err = ax.stairs(h+herr, bin_edges, baseline=h-herr, 
                                  alpha=0.3, color=stair.get_edgecolor(), fill=True)
        ax.set(xlim=(bin_edges[0], bin_edges[-1]))
        ax.legend()
        return fig, ax

    def plot_projection(self, bin_edges: list[float], ax: plt.Axes, colors: dict[str, str] | None = None, use_norm_templates: bool = True) -> None:
        # bc = 0.5*(bin_edges[1:] + bin_edges[:-1])
        # bw = 0.5*(bin_edges[1:] - bin_edges[:-1])
        N = self.get_yields()
        if use_norm_templates:
            hres = [np.concatenate((self.templates[k].h()*N[k], [0])) for k in N]
        else:
            hres = [np.concatenate((self.templates[k].H()*N[k], [0])) for k in N]
        labels = [self.templates[k].label for k in N]
        if colors is not None:
            color = [colors[k] for k in N]
            ax.stackplot(bin_edges, hres[::-1], labels=labels[::-1], colors=color[::-1], step="post")
        else:
            ax.stackplot(bin_edges, hres[::-1], labels=labels[::-1], step="post")
        
        # ax.legend()
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles[::-1], labels[::-1])

        hres_sum = np.sum(hres, axis=0)[:-1]
        hres_err_total = self.pdf_err()
        ax.stairs(hres_sum+hres_err_total, bin_edges, baseline=hres_sum-hres_err_total, alpha=0.65, color="gray", fill=True)
        
