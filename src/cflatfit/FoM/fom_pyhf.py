from cflatfit.FoM import FigureOfMerit


class FoMPyHF(FigureOfMerit):
    def __init__(self):
        super().__init__(ignore_constraint = False)
        self._w = None

    def get_spec(self) -> dict:
        spec = {
            "channels"     : [],
            "observations" : [],
            "measurements" : [],
            "version" : "1.0.0"
        }

        for ichannel in self.channels.values():
            ich, iobs, imeas = ichannel.get_pyhf_config()
            spec["channels"].append(ich)
            spec["observations"].append(iobs)
            spec["measurements"].append(imeas)
        
        return spec