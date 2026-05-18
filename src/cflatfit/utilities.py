import numpy as np
import scipy.stats as stats
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL

def gaussian(x: np.ndarray | float, mu: float, sigma: float) -> np.ndarray | float:
    return stats.norm.pdf(x, loc=mu, scale=sigma)

def make_gaussian_pulls_fit(pulls: list[float]) -> tuple[float]:
    cost = UnbinnedNLL(pulls, gaussian)
    mg = Minuit(cost, mu=0.0, sigma=1.0)
    mg.migrad()
    mg.hesse()
    return mg.params["mu"].value, mg.params["mu"].error, mg.params["sigma"].value, mg.params["sigma"].error, 

def make_gaussian_pull_label(mu: float, 
                             muerr: float, 
                             sigma: float, 
                             sigmaerr: float,
                             label_head: str = "Fit") -> str:
    plabel = (
        label_head
        + r" $\mu = "
        + f"{mu:.2f}"
        + r"\pm"
        + f"{muerr:.2f}"
        + r"$, "
        + r"$\sigma = "
        + f"{sigma:.2f}"
        + r"\pm"
        + f"{sigmaerr:.2f}"
        + r"$"
    )
    return plabel