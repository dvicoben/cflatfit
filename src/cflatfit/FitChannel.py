import numpy as np
import matplotlib.pyplot as plt

import cflatfit.Parameter as mp
from cflatfit.PDFBase import PDFBase
from cflatfit.Templates import Template

from cflatfit.logger_config import make_logger
logger = make_logger(__name__)
    
def make_pyhf_sample_config(template: Template, normfactors: list[mp.Parameter]) -> dict:
    sample = {
        "name"      : template.name,
        "data"      : template.unrolled().tolist(),
        "modifiers" : []
    }
    
    for jpar in normfactors:
        imod = {
            "name" : jpar.name,
            "type" : "normfactor",
            "data" : None
        }
        sample["modifiers"].append(imod)
    
    # Now BB shapesys or gausian staterror
    templ_constr = template.constraint
    if templ_constr != mp.ConstraintType.NONE:
        mcstatmod = {
            "name" : f"{template.name}_mcsys",
            # "type" : "shapsys" or "staterror",
            "data" : template.unrollederr().tolist()
        }
        if templ_constr == mp.ConstraintType.POISSON:
            mcstatmod["type"] = "shapesys"
        if templ_constr == mp.ConstraintType.GAUSSIAN:
            mcstatmod["type"] = "staterror"

        sample["modifiers"].append(mcstatmod)
    
    return sample


def make_pyhf_parameter_config(par: mp.Parameter) -> dict:
    parconfig = {
        "name"   : par.name,
        "bounds" : [[par.limlo, par.limhi]],
        "inits"  : [par.val]
    }
    return parconfig



class Channel:
    def __init__(self, 
                 name: str,
                 pdf: PDFBase,
                 data: np.ndarray,
                 dataErr: np.ndarray = None,
                 normPDF_before_eval: bool = False):
        self._name: str = name
        self._pdf: PDFBase = pdf
        self._data_shape: tuple[int] = np.shape(data)
        self._data: np.ndarray = np.array(data).ravel()
        self._dataNorm = np.sum(self.data)
        self._dataErr: np.ndarray = np.sqrt(self.data) if dataErr is None else np.array(dataErr).ravel()
        self._dataVar: np.ndarray = self.dataErr**2
        self._normPDF_before_eval: bool = normPDF_before_eval
        self._scaleFactorW2: np.ndarray = (self.dataErr**2)/np.where(self.data != 0, self.data, 1.0)
        
        self.make_nuisance_constraints()
        # Set the PDF parameters here to in_use such that they will be propagated to minimization
        for iparint, iparman in self.pdf.params.items():
            self.pdf.param_manager[iparman].use()
        self.pdf.param_manager.fetch_base_and_use()

    @property
    def name(self) -> str: return self._name
    @property
    def pdf(self) -> PDFBase: return self._pdf
    @property
    def templates(self) -> dict[str, Template]: return self.pdf.templates
    @property
    def data(self) -> np.ndarray: return self._data
    @property
    def dataNorm(self) -> float: return self._dataNorm
    @property
    def dataErr(self) -> np.ndarray: return self._dataErr
    @property
    def dataVar(self) -> np.ndarray: return self._dataVar
    @property
    def normPDF_before_eval(self) -> bool: return self._normPDF_before_eval
    @property
    def scaleFactorW2(self) -> np.ndarray: return self._scaleFactorW2

    def set_data(self, data: np.ndarray, dataErr: np.ndarray) -> None:
        self._data_shape: tuple[int] = np.shape(data)
        self._data: np.ndarray = data.ravel()
        self._dataNorm = np.sum(self.data)
        self._dataErr: np.ndarray = np.sqrt(self.data) if dataErr is None else dataErr.ravel()
        self._dataVar: np.ndarray = self.dataErr**2
        self._scaleFactorW2: np.ndarray = (self.dataErr**2)/np.where(self.data != 0, self.data, 1.0)

    def get_model(self):
        if self._normPDF_before_eval:
            model = self.pdf.normpdf()
            model_scaled = self.dataNorm*model
            return model_scaled
        return self.pdf.pdf()

    def get_model_err(self):
        if self._normPDF_before_eval:
            err = self.pdf.normpdf_err()
            return self.dataNorm*err
        return self.pdf.pdf_err()

    def make_nuisance_constraints(self) -> None:
        for itemplate in self.templates.values():
            itemplate.create_all_nuisance()
    
    def fluctuate_templates(self, rng: np.random.Generator) -> None:
        for itemplate in self.templates.values():
            itemplate.fluctuate_template(rng)

    def plot_projection(self, bin_edges: list[float], 
                        include_template_err: bool = True,
                        colors: dict[str, str] | None = None,
                        remove_pull: bool = False
                        ) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
        if remove_pull:
            fig, ax1 = plt.subplots(1, 1)
        
        bin_edges = np.array(bin_edges)
        bc = 0.5*(bin_edges[1:] + bin_edges[:-1])
        bw = 0.5*(bin_edges[1:] - bin_edges[:-1])
        
        self.pdf.plot_projection(bin_edges, ax1, colors)
        kwargs = {"fmt" : 'ko', "markersize" : 2, "label" : "Data"}
        ax1.errorbar(bc, self.data, yerr=self.dataErr, **kwargs)

        # ax1.legend()
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles[::-1], labels[::-1])
        
        xlim = (bin_edges[0], bin_edges[-1])
        ax1.set(xlim=xlim, ylabel="Candidates")
        if remove_pull:
            return fig, ax1

        hres_sum = self.pdf.pdf()
        hres_err_total = self.pdf.pdf_err()
        residual = hres_sum - self.data
        pulls = (residual)/np.sqrt(self.dataErr**2 + hres_err_total**2) if include_template_err else (residual)/self.dataErr
        # ax2.errorbar(bc, pulls, xerr=bw, yerr=1., fmt='ko', markersize=2)
        ax2.stairs(pulls, bin_edges, fill=True, color='k', alpha = 0.85)
        ax2.axhline(0, 0, 1, color='grey', alpha=0.5, zorder = -2)
        ax2.axhline(3, 0, 1, color='grey', alpha=0.5, zorder = -2, linestyle='--')
        ax2.axhline(-3, 0, 1, color='grey', alpha=0.5, zorder = -2, linestyle='--')
        ax2.set(xlim=xlim, ylabel="Pulls")
        return fig, ax1, ax2
    
    def plot_projection_total(
            self, bin_edges: list[float] = None, remove_pull: bool = False
        ) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
        if remove_pull:
            fig, ax1 = plt.subplots(1, 1)
        
        if bin_edges is None:
            bin_edges = np.arange(len(self.data)+1)
        else:
            bin_edges = np.array(bin_edges)
        bc = 0.5*(bin_edges[1:] + bin_edges[:-1])
        bw = 0.5*(bin_edges[1:] - bin_edges[:-1])
        
        kwargs = {"fmt" : 'ko', "markersize" : 2, "label" : "Data"}
        ax1.errorbar(bc, self.data, yerr=self.dataErr, xerr=bw, **kwargs)

        hres_sum = self.get_model()
        ax1.stairs(hres_sum, bin_edges, color="b", label="Fit")

        # ax1.legend()
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles[::-1], labels[::-1], fontsize="xx-large")
        
        xlim = (bin_edges[0], bin_edges[-1])
        ax1.set(xlim=xlim, ylabel="Candidates")

        if remove_pull:
            return fig, ax1
        
        residual = hres_sum - self.data
        pulls = (residual)/self.dataErr
        # ax2.errorbar(bc, pulls, xerr=bw, yerr=1., fmt='ko', markersize=2)
        ax2.stairs(pulls, bin_edges, fill=True, color='k', alpha = 0.85)
        ax2.axhline(0, 0, 1, color='grey', alpha=0.5, zorder = -2)
        ax2.axhline(3, 0, 1, color='grey', alpha=0.5, zorder = -2, linestyle='--')
        ax2.axhline(-3, 0, 1, color='grey', alpha=0.5, zorder = -2, linestyle='--')
        ax2.set(xlim=xlim, ylabel="Pulls")

        return fig, ax1, ax2


    def get_pyhf_config(self) -> tuple[dict, dict, dict]:
        yields_for_templ = self.pdf.get_yield_parameters()
        parameters: set[mp.Parameter] = set()
        poi: str = self.pdf.poi_ext

        channel = {
            "name"    : self.pdf.name,
            "samples" : [],
        }
        
        for itempl, ipar in yields_for_templ.items():
            
            op_ch, ijpars = mp.get_operation_chain(ipar)
            if not mp.is_product_only(ipar):
                logger.error(f"Using a compound parameter ({ipar.name}) that is not multiplicative only is not supported for pyhf fitting!")
                raise Exception("Using a compound parameter that is not multiplicative only is not supported for pyhf fitting!")
            if poi is None:
                logger.warning(f"PDF poi not set, setting poi to {ijpars[0].name}")
                poi = ijpars[0].name

            isample = make_pyhf_sample_config(self.templates[itempl], ijpars)
            channel["samples"].append(isample)
            parameters.update(ijpars)

        # Note data errors are assumed Poisson here!
        obs = {
            "name"    : self.pdf.name,
            "data"    : self.data.tolist()
        }
        meas = {
            "name"    : "Meas_"+self.pdf.name,
            "config"  : {
                "poi"        : None,
                "parameters" : []
            }
        }
        meas["config"]["poi"] = poi

        logger.warning("Cannot Gaussian constrain normfactor in pyhf, need to be handled outside. Currently not implemented")
        for upar in list(parameters):
            parconfig = make_pyhf_parameter_config(upar)            
            meas["config"]["parameters"].append(parconfig)

        return channel, obs, meas
