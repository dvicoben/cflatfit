from __future__ import annotations
from enum import StrEnum
import numpy as np

from typing import Callable
import cflatfit.CostFuncMath as cf

from cflatfit.logger_config import make_logger
logger = make_logger(__name__)


class ParameterOperation(StrEnum):
    ADD       = "+"
    SUBTRACT  = "-"
    SUB       = "-"
    MULTIPLY  = "*"
    MULT      = "*"
    DIVIDE    = "/"
    DIV       = "/"
    EXP       = "**"
    POW       = "**"
    ARBITRARY = "ARB"
    COMPOSITE = "COMP"

class ConstraintType(StrEnum):
    NONE          = "None"
    GAUSSIAN      = "Gaussian"
    GAUSS         = "Gaussian"
    GAUSSIANSCALE = "GaussianScale"
    POISSON       = "Poisson"
    BB            = "Poisson"


def parse_constraint(constraint_type: ConstraintType, constraint_vals: list[float]) -> bool:
    match constraint_type:
        case ConstraintType.NONE:
            return True
        case ConstraintType.GAUSSIAN:
            if len(constraint_vals) != 2:
                logger.error(f"{constraint_type} constraint expects vals [mu, sigma]")
                return False
            return True
        case ConstraintType.POISSON:
            if len(constraint_vals) != 1:
                logger.error(f"{constraint_type} constraint expects vals [mu]")
                return False
            return True
    logger.error(f"Invalid constraint type {constraint_type}")
    return False


def get_entire_chain_from_compound(par: CompoundParameter) -> set[Parameter]:
    pars = set()
    if type(par) == Parameter:
        pars.update([par])
        return pars

    for ipar in par.pars:
        pars.update([ipar])
        if type(ipar) != Parameter:
            pars.update(get_entire_chain_from_compound(ipar))

    return pars


def get_basic_from_compound(par: CompoundParameter) -> set[Parameter]:
    base_pars = set()
    if type(par) == Parameter:
        base_pars.update([par])
        return base_pars

    for ipar in par.pars:
        if type(ipar) == Parameter:
            base_pars.update([ipar])
        else:
            base_pars.update(get_basic_from_compound(ipar))
    return base_pars


def get_all_basic_params(manager: ParameterManager) -> set[Parameter]:
    base_pars = set()
    for ipar in manager.params.values():
        base_pars.update(get_basic_from_compound(ipar))
    return base_pars


def is_compound(par: Parameter | CompoundParameter) -> bool:
    return type(par) == CompoundParameter


def get_operation_chain(par: Parameter | CompoundParameter) -> tuple[str, list[Parameter]]:
    if type(par) == Parameter:
        opstr = par.name
        return opstr, [par]
    
    opstr = "" + par.opstr
    parlist = []
    for ipar in par.pars:
        if type(ipar) == CompoundParameter:
            iopstr, ilist = get_operation_chain(ipar)
            opstr = opstr.replace(ipar.name, iopstr)
            parlist.extend(ilist)
        else:
            parlist.append(ipar)
    return opstr, parlist


def is_product_only(par: Parameter | CompoundParameter) -> bool:
    if type(par) == Parameter:
        return True
    
    is_prod = par.op == ParameterOperation.MULTIPLY
    for ipar in par.pars:
        if not is_prod:
            break
        if type(ipar) != Parameter:
            is_prod = is_product_only(ipar)
    
    return is_prod



class Parameter:
    def __init__(self, 
                 name: str,
                 val: float, 
                 limlo: float = None, 
                 limhi: float = None,
                 fixed: bool = False):

        self._name: str = name
        self._val: float = val
        self._limlo: float = limlo
        self._limhi: float = limhi
        self._fixed: bool = fixed
        self._constrainttype: ConstraintType = ConstraintType.NONE
        self._constraintvals: list[float] = []
        self._in_use: bool = False

    @property
    def name(self) -> str: return self._name
    @property
    def val(self) -> float: return self.getVal()
    @property
    def limlo(self) -> float: return self._limlo
    @property
    def limhi(self) -> float: return self._limhi
    @property
    def fixed(self) -> bool: return self._fixed
    @property
    def constrain(self) -> bool: return self._constrainttype != ConstraintType.NONE
    @property
    def constrainttype(self) -> ConstraintType: return self._constrainttype
    @property
    def constraintvals(self) -> list[float]: return self._constraintvals
    @property
    def in_use(self) -> bool: return self._in_use

    def fix(self, val: float = None) -> None:
        if val is not None:
            self._val = val
        self._fixed = True
    
    def setlim(self, lo: float = None, hi: float = None) -> None:
        if lo is not None:
            self._limlo = lo
        if hi is not None:
            self._limhi = hi

    def set_constraint(self, constraint_type: ConstraintType, constraint_vals: list[float]) -> None:
        if not parse_constraint(constraint_type, constraint_vals):
            logger.error(f"Constraint for {self.name} could not be parsed")
            raise ValueError("Could not parse specified constraint")
        
        self._constrainttype = constraint_type
        self._constraintvals = constraint_vals
    
    def constrain_gauss(self, mu: float, sigma: float) -> None:
        self.set_constraint(ConstraintType.GAUSSIAN, [mu, sigma])

    def constrain_poisson(self, mu: float) -> None:
        self.set_constraint(ConstraintType.POISSON, [mu])

    def get_constraint(self) -> float:
        match self.constrainttype:
            case ConstraintType.GAUSSIAN:
                return cf.nll_gaussian(self.constraintvals[0], self.getVal(), self.constraintvals[1])
            case ConstraintType.POISSON:
                return cf.nll_poisson(self.constraintvals[0], self.getVal())
        return 0.0

    def setVal(self, val: float) -> None:
        self._val = val

    def getVal(self) -> float:
        return self._val
    
    def get_basic(self) -> set[Parameter]:
        return get_basic_from_compound(self)
    
    def get_chain(self) -> set[Parameter]:
        return get_entire_chain_from_compound(self)
    
    def set_resample_constraint(self, rng: np.random.Generator = np.random.default_rng()) -> None:
        match self.constrainttype:
            case ConstraintType.NONE:
                return
            case ConstraintType.POISSON:
                val = float(rng.poisson(self.constraintvals[0]))
                self._constraintvals[0] = val
            case ConstraintType.GAUSSIAN:
                val = rng.normal(self.constraintvals[0], self.constraintvals[1])
                self._constraintvals[0] = val

    def use(self):
        # Could set to always be false if parameter is fixed, no need to pass minuit a fixed parameter
        # if self.fixed:
        #     logger.info(f"Parameter {self.name} is fixed so forced unused in fitter")
        #     self._in_use = False
        #     return
        self._in_use = True
    
    def unuse(self):
        self._in_use = False

    def __repr__(self):
        return f"<Parameter {self.name}={self.val}, at {hex(id(self))}>"
    
    # def __add__(self, param: Parameter) -> CompoundParameter:
    #     return CompoundParameter(f"({self.name}+{param.name})", self, ParameterOperation.ADD, param)
    
    # def __sub__(self, param: Parameter) -> CompoundParameter:
    #     return CompoundParameter(f"({self.name}-{param.name})", self, ParameterOperation.SUBTRACT, param)
    
    # def __mul__(self, param: Parameter) -> CompoundParameter:
    #     return CompoundParameter(f"({self.name}*{param.name})", self, ParameterOperation.MULTIPLY, param)

    # def __truediv__(self, param: Parameter) -> CompoundParameter:
    #     return CompoundParameter(f"({self.name}/{param.name})", self, ParameterOperation.DIVIDE, param)

    # def __pow__(self, param: Parameter) -> CompoundParameter:
    #     return CompoundParameter(f"({self.name}^{param.name})", self, ParameterOperation.EXP, param)

    def __add__(self, other: float | np.ndarray) -> float | np.ndarray:
        return self.getVal() + other
    
    def __radd__(self, other: float | np.ndarray) -> float | np.ndarray:
        return other + self.getVal()
    
    def __sub__(self, other: float | np.ndarray) -> float | np.ndarray:
        return self.getVal() - other
    
    def __rsub__(self, other: float | np.ndarray) -> float | np.ndarray:
        return other - self.getVal()
    
    def __mul__(self, other: float | np.ndarray) -> float | np.ndarray:
        return self.getVal()*other
    
    def __rmul__(self, other: float | np.ndarray) -> float | np.ndarray:
        return other*self.getVal()

    def __truediv__(self, other: float | np.ndarray) -> float | np.ndarray:
        return self.getVal()/other
    
    def __rtruediv__(self, other: float | np.ndarray) -> float | np.ndarray:
        return other/self.getVal()
    
    def __pow__(self, other: float | np.ndarray) -> float | np.ndarray:
        return self.getVal()**other
    
    def __rpow__(self, other: float | np.ndarray) -> float | np.ndarray:
        return other**self.getVal()


class CompoundParameter(Parameter):
    def __init__(self, name: str, par1: Parameter, op: ParameterOperation, par2: Parameter):
        self._op: ParameterOperation = op
        self._pars: tuple[Parameter] = (par1, par2)
        super().__init__(name, self.getVal())

    @property
    def pars(self) -> tuple[Parameter]: return self._pars
    @property
    def op(self) -> ParameterOperation: return self._op
    @property
    def opstr(self) -> str: return f"({self.pars[0].name} {self.op} {self.pars[1].name})"

    def getVal(self) -> float:
        match self.op:
            case ParameterOperation.ADD:
                self.setVal(self.pars[0].getVal() + self.pars[1].getVal())
            case ParameterOperation.SUBTRACT:
                self.setVal(self.pars[0].getVal() - self.pars[1].getVal())
            case ParameterOperation.MULTIPLY:
                self.setVal(self.pars[0].getVal() * self.pars[1].getVal())
            case ParameterOperation.DIVIDE:
                self.setVal(self.pars[0].getVal() / self.pars[1].getVal())
            case ParameterOperation.EXP:
                self.setVal(self.pars[0].getVal() ** self.pars[1].getVal())
            case _:
                logger.error(f"Invalid parameter operation ({self.op}) for {self.name}")
                raise ValueError("Not a valid operation")
        return self._val

    def __repr__(self) -> str:
        return f"<CompoundParameter {self.name}={self.pars[0].name}{self.op}{self.pars[1].name}={self.val}, at {hex(id(self))}>"

    def use(self) -> None:
        for ipar in list(get_basic_from_compound(self)):
            ipar.use()



class CompositeParameter(Parameter):
    def __init__(self, name: str, 
                 parlist: list[Parameter], 
                 fcn: Callable[[list[float]], float]):
        self._op: ParameterOperation = ParameterOperation.COMPOSITE
        self._pars: list[Parameter] = parlist
        self._fcn = fcn
        super().__init__(name, self.getVal())

    @property
    def pars(self) -> list[Parameter]: return self._pars
    @property
    def op(self) -> ParameterOperation: return self._op
    @property
    def fcn(self) -> Callable[[list[float]], float]:
        return self._fcn
    
    def getVal(self) -> float:
        # self.setVal(self.fcn(self.pars))
        self.setVal(self.fcn([ipar.getVal() for ipar in self.pars]))
        return self._val
    
    def __repr__(self) -> str:
        return f"<CompositeParameter {self.name}={self.op}({len(self.pars)} Params)={self.val}, at {hex(id(self))}>"

    def use(self) -> None:
        for ipar in list(get_basic_from_compound(self)):
            ipar.use()



class SummedParameter(CompositeParameter):
    def __init__(self, name: str, 
                 parlist: list[Parameter]):
        super().__init__(name, parlist, np.sum)



class ProductParameter(CompositeParameter):
    def __init__(self, name: str, 
                 parlist: list[Parameter]):
        super().__init__(name, parlist, np.prod)



class ParameterManager:
    def __init__(self):
        self._params: dict = {}
        self._base_params: list[str] = []
        self._in_use: list[str] = []

    @property
    def params(self) -> dict[str, Parameter]: return self._params
    @property
    def base_params(self) -> list[str]: return self._base_params
    @property
    def in_use(self) -> list[str]: return self._in_use

    def __getitem__(self, parname: str) -> Parameter:
        return self.getParam(parname)

    def update(self, pars: dict[str, Parameter]):
        for iparname, ipar in pars.items():
            if iparname not in self.params:
                continue
            # Deal with parameters that already exist in manager
            if ipar is not self.params[iparname]:
                logger.warning(f"{ipar} parameter has same name as {self.params[iparname]}, will only keep {ipar} and assuming shared. " 
                               +"If this is not intended please ensure the parameters have different names.")
            # else:
            #     logger.info(f"{ipar} is duplicate, assuming same/shared and keeping once")
                
        self.params.update(pars)
    
    def fetch_base_and_use(self):
        self._base_params = [iparam.name for iparam in list(get_all_basic_params(self))]
        self._in_use = [ipar for ipar in self.base_params if self.params[ipar].in_use]

    def merge(self, other_manager: ParameterManager) -> None:
        if other_manager is self:
            return
        self.update(other_manager.params)
        self.fetch_base_and_use()

    def addParam(self, par: Parameter) -> None:
        parchain = par.get_chain()
        parchain.update({par})
        for ipar in list(parchain):
            self.update({ipar.name : ipar})
        self.fetch_base_and_use()
    
    def getVal(self, parname: str) -> float:
        return self.params[parname].getVal()
    
    def setVal(self, parname: str, val: float) -> None:
        par = self.params[parname]
        if is_compound(par):
            logger.warning("Setting value for a compound parameter, likely ignored in minimisation")
        par.setVal(val)

    def setVals(self, parvals: dict) -> None:
        for ipar in parvals:
            self.setVal(ipar, parvals[ipar])
    
    def getParam(self, parname: str) -> Parameter:
        return self.params[parname]