import numpy as np
from relaylab import characteristics
from relaylab.signals import AnalogSignal, ComplexSignal, DiscreteSignal
import pytest


class TestDifRelay:
    dif_data = [
        (0.199, 0, False), (0.201, 0, True),
        (0.199, 0.5, False), (0.201, 0.5, True),
        (0.299, 1, False), (0.301, 1, True),
        (0.399, 1.5, False), (0.401, 1.5, True),
        (0.899, 2.5, False), (0.901, 2.5, True)
    ]

    @pytest.mark.parametrize('dif, bias, expected', dif_data)
    def test_start(self, dif, bias, expected):
        relay = characteristics.DifRelay(st0=0.2, slope_st1=0.5, slope1=0.2, slope_st2=1.5, slope2=0.5)
        Idif = AnalogSignal(val=np.array([dif]))
        Ibias = AnalogSignal(val=np.array([bias]))
        assert relay.start(Idif, Ibias, dif_dft=False).val[0] == expected
