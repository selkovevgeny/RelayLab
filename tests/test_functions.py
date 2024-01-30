from relaylab import functions as func
from relaylab import signals as sg
import pytest

def test_symmetrical_comp_seq1():
    Ia = sg.sin(1, -30, 'Ia', tmax=0.1)
    Ib = sg.sin(1, 210, 'Ib', tmax=0.1)
    Ic = sg.sin(1, 90, 'Ic', tmax=0.1)
    I1, I2, I0 = func.symmetrical_comp(Ia, Ib, Ic)
    assert I1.abs().val[-1] == pytest.approx(1, 0.001)
    assert I1.angle_deg().val[-1] == pytest.approx(-30, 0.001)
    assert I2.abs().val[-1] == pytest.approx(0, 0.001)
    assert I0.abs().val[-1] == pytest.approx(0, 0.001)


def test_symmetrical_comp_seq2():
    Ia = sg.sin(1, -30, 'Ia', tmax=0.1)
    Ib = sg.sin(1, 90, 'Ib', tmax=0.1)
    Ic = sg.sin(1, 210, 'Ic', tmax=0.1)
    I1, I2, I0 = func.symmetrical_comp(Ia, Ib, Ic)
    assert I1.abs().val[-1] == pytest.approx(0, 0.001)
    assert I2.abs().val[-1] == pytest.approx(1, 0.001)
    assert I2.angle_deg().val[-1] == pytest.approx(-30, 0.001)
    assert I0.abs().val[-1] == pytest.approx(0, 0.001)


def test_symmetrical_comp_seq0():
    Ia = sg.sin(1, -30, 'Ia', tmax=0.1)
    Ib = sg.sin(1, -30, 'Ib', tmax=0.1)
    Ic = sg.sin(1, -30, 'Ic', tmax=0.1)
    I1, I2, I0 = func.symmetrical_comp(Ia, Ib, Ic)
    assert I1.abs().val[-1] == pytest.approx(0, 0.001)
    assert I2.abs().val[-1] == pytest.approx(0, 0.001)
    assert I0.abs().val[-1] == pytest.approx(1, 0.001)
    assert I0.angle_deg().val[-1] == pytest.approx(-30, 0.001)


def test_symmetrical_comp_seq1_ab():
    Ia = sg.sin(1, -30, 'Ia', tmax=0.1)
    Ib = sg.sin(1, 210, 'Ib', tmax=0.1)
    Ic = sg.sin(1, 90, 'Ic', tmax=0.1)
    I1, I2 = func.symmetrical_comp(Ia-Ib, Ib-Ic)
    assert I1.abs().val[-1] == pytest.approx(1, 0.001)
    assert I1.angle_deg().val[-1] == pytest.approx(-30, 0.001)
    assert I2.abs().val[-1] == pytest.approx(0, 0.001)


def test_symmetrical_comp_seq2_ab():
    Ia = sg.sin(1, -30, 'Ia', tmax=0.1)
    Ib = sg.sin(1, 90, 'Ib', tmax=0.1)
    Ic = sg.sin(1, 210, 'Ic', tmax=0.1)
    I1, I2 = func.symmetrical_comp(Ia-Ib, Ib-Ic)
    assert I1.abs().val[-1] == pytest.approx(0, 0.001)
    assert I2.abs().val[-1] == pytest.approx(1, 0.001)
    assert I2.angle_deg().val[-1] == pytest.approx(-30, 0.001)