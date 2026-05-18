from cflatfit.FoM import FigureOfMerit
from cflatfit.Parameter import ConstraintType

def has_template_with_cons(fom: FigureOfMerit) -> bool:
    for ichannel in fom.channels.values():
            for itempl in ichannel.templates.values():
                if itempl.constraint != ConstraintType.NONE:
                    return True