from relaylab import signals as sg
import pytest


def test_sin():
    I = sg.sin(1, 0, 'Ia', f=50, tmax=0.021, Fs=1000)
    assert len(I.val) == 21
    assert I.val[0] == pytest.approx(0, 0.001)
    assert I.val[10] == pytest.approx(0, 0.001)
    assert I.val[20] == pytest.approx(0, 0.001)
    assert I.val[5] == pytest.approx(2 ** 0.5, 0.001)
    assert I.val[15] == pytest.approx(-2 ** 0.5, 0.001)


def test_dft():
    I = sg.sin(1, -30, 'Ia', f=50, tmax=0.041, Fs=1000).dft()
    assert I.abs().val[20] == pytest.approx(1, 0.001)
    assert I.angle_deg().val[20] == pytest.approx(-30, 0.001)
    assert I.abs().val[25] == pytest.approx(1, 0.001)
    assert I.angle_deg().val[30] == pytest.approx(-30, 0.001)


def test_dft_2h():
    I = sg.sin(1, -30, 'Ia', f=100, tmax=0.041, Fs=1000).dft(harm=2)
    assert I.abs().val[20] == pytest.approx(1, 0.001)
    assert I.angle_deg().val[20] == pytest.approx(-30, 0.001)
    assert I.abs().val[25] == pytest.approx(1, 0.001)
    assert I.angle_deg().val[30] == pytest.approx(-30, 0.001)


def test_dft_rot():
    I = sg.DFT(sg.sin(1, -30, 'Ia', f=50, tmax=0.041, Fs=1000), rot=True)
    assert I.abs().val[20] == pytest.approx(1, 0.001)
    assert I.angle_deg().val[20] == pytest.approx(-30, 0.001)
    assert I.abs().val[25] == pytest.approx(1, 0.001)
    assert I.angle_deg().val[25] == pytest.approx(60, 0.001)
    assert I.angle_deg().val[30] == pytest.approx(150, 0.001)
    assert I.angle_deg().val[35] == pytest.approx(-120, 0.001)


def test_impulses():
    imp = sg.impulses(vals=[0, 220, 0, 0], t=[0, 0.1, 0.11, 0.2], name='signal', Fs=100)
    assert imp.val[0] == 0
    assert imp.val[9] == 0
    assert imp.val[10] == 220
    assert imp.val[11] == 0
    assert imp.val[-1] == 0

