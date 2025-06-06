"""
relaylab.functions
================

Модуль содержит функции расчета, получающие на вход несколько сигналов, например, расчет симметричных составляющих,
расчет дифференциальных токов и пр.
"""

from relaylab.signals import AnalogSignal as _AnalogSignal, ComplexSignal as _ComplexSignal, DFT as _DFT
import numpy as np
from relaylab.signals import const
from typing import Union as _Union
from scipy.signal import lfilter as _lfilter
from relaylab.equipment import Line

def _check_type(var, var_types):
    if type(var) not in var_types:
        raise TypeError(f'Неверный тип входных данных. Ожидается {var_types}, получено {type(var)}')


def _check_type_equality(args):
    if not all(map(lambda var: type(var) == type(args[0]), args)):
        raise ValueError(f'Аргументы {args} должны быть одного типа')


def symmetrical_comp(*args):
    """Расчет симметричных составляющих токов или напряжений.

    :param args: A, B, C или AB, BC, type: ComplexSignal или AnalogSignal
    :return: seq1, seq2 , seq0 или seq1, seq2, type: ComplexSignal
    """
    for arg in args:
        _check_type(arg, (_AnalogSignal, _ComplexSignal))
    _check_type_equality(args)
    Fs = args[0].Fs
    args = _DFT(*args) if type(args[0] == _AnalogSignal) else args
    if len(args) == 3:
        a, b, c = map(lambda s: s.val, args)
        seq1 = _ComplexSignal(name='A1', val=(a + b * const.a + c * const.a ** 2) / 3, Fs=Fs)
        seq2 = _ComplexSignal(name='A2', val=(a + b * const.a ** 2 + c * const.a) / 3, Fs=Fs)
        seq0 = _ComplexSignal(name='A0', val=(a + b + c) / 3, Fs=Fs)
        return seq1, seq2, seq0
    elif len(args) == 2:
        ab, bc = map(lambda s: s.val, args)
        seq1 = _ComplexSignal(name='A1', val=(ab - bc * const.a ** 2) / 3, Fs=Fs)
        seq2 = _ComplexSignal(name='A2', val=(ab - bc * const.a) / 3, Fs=Fs)
        return seq1, seq2
    else:
        raise ValueError('Неверное количество аргументов. Должно быть 2 или 3.')


def dif_single_phase(I1: _AnalogSignal, I2: _AnalogSignal, Inom1=1, Inom2=1, CT1=None, CT2=None):
    """ Расчет дифференциального тока и тока торможения для одной фазы
    Если CT1 или CT2 не заданы, то приведение токов сторон выполняется по формулам:
        val= I1 * CT.n / Inom
    В противном случае:
        val= I1 / Inom
    Диф. ток равен:
        Idif = val1 + val2
    Ток торможения равен:
        Ibias = 0.5(abs(val1.dft()) + abs(val2.dft()))

    :param I1: аналоговый сигнал тока стороны 1, type: AnalogSignal
    :param I2: аналоговый сигнал тока стороны 2, type: AnalogSignal
    :param Inom1: номинальный ток стороны 1, type: float, int
    :param Inom2: номинальный ток стороны 2, type: float, int
    :param CT1: трансформатор тока стороны 1, type: CT
    :param CT2: трансформатор тока стороны 2, type: CT
    :return: list(Idif, Ibias), где Idif - мгновенные значения диф. тока, type: AnalogSignal
                                    Ibias - ток торможения, type: AnalogSignal
    """
    if CT1 is not None and CT1 is not None:
        val1, val2 = I1 * CT1.n.val / Inom1, I2 * CT2.n.val / Inom2
    else:
        val1, val2 = I1 / Inom1, I2 / Inom1
    if I1.__class__ == _AnalogSignal and I2.__class__ == _AnalogSignal:
        Idif = _AnalogSignal(name='Iдиф', Fs=I1.Fs)
        Idif.val = (val1 + val2).val
        Ibias = _AnalogSignal(name='Iторм', Fs=I1.Fs)
        Ibias.val = (np.abs(_DFT(val1).val) + np.abs(_DFT(val2).val)) / 2
    elif I1.__class__ == _ComplexSignal and I2.__class__ == _ComplexSignal:
        Idif = _ComplexSignal(name='Iдиф', Fs=I1.Fs)
        Idif.val = (val1 + val2).val
        Ibias = _AnalogSignal(name='Iторм', Fs=I1.Fs)
        Ibias.val = (abs(val1) + abs(val2)).val / 2
    else:
        raise ValueError('Неверный тип входных данных. Сигналы должны быть типа: AnalogSignal или ComplexSignal')
    return Idif, Ibias


def dif_three_phase(IA1: _AnalogSignal, IB1: _AnalogSignal, IC1: _AnalogSignal,
              IA2: _AnalogSignal, IB2: _AnalogSignal, IC2: _AnalogSignal,
              Inom1=1, Inom2=1, CT_high=None, CT_low=None):
    """Расчет дифференциального тока и тока торможения для двигателя.
    При расчете осуществляется приведение сторон:
    I1_norm = I1 * CT.n / Inom_motor
    Диф. ток равен:
        Idif = val1 + val2
    Ток торможения равен:
        Ibias = 0.5(abs(val1.dft()) + abs(val2.dft()))

    :param IA1: аналоговый сигнал тока стороны 1, type: AnalogSignal
    :param IB1: аналоговый сигнал тока стороны 1, type: AnalogSignal
    :param IC1: аналоговый сигнал тока стороны 1, type: AnalogSignal
    :param IA2: аналоговый сигнал тока стороны 2, type: AnalogSignal
    :param IB2: аналоговый сигнал тока стороны 2, type: AnalogSignal
    :param IC2: аналоговый сигнал тока стороны 2, type: AnalogSignal
    :param motor: двигатель, type: Motor
    :param CT_high: трансформатор тока стороны 1, type: CT
    :param CT_low: трансформатор тока стороны 2, type: CT
    :return: list(IdifA, IdifB, IdifC, IbiasA, IbiasB, IbiasC),
                                    где Idif - мгновенные значения диф. тока, type: AnalogSignal
                                        Ibias - ток торможения, type: AnalogSignal
    """
    IdifA, IbiasA = dif_single_phase(IA1, IA2, Inom1=Inom1, Inom2=Inom2,
                                     CT1=CT_high, CT2=CT_low)
    IdifB, IbiasB = dif_single_phase(IB1, IB2, Inom1=Inom1, Inom2=Inom2,
                                     CT1=CT_high, CT2=CT_low)
    IdifC, IbiasC = dif_single_phase(IC1, IC2, Inom1=Inom1, Inom2=Inom2,
                                     CT1=CT_high, CT2=CT_low)
    IdifA.name, IbiasA.name = 'IдифA', 'IтормA'
    IdifB.name, IbiasB.name = 'IдифB', 'IтормB'
    IdifC.name, IbiasC.name = 'IдифC', 'IтормC'
    return IdifA, IdifB, IdifC, IbiasA, IbiasB, IbiasC


def dif_two_winding_transformer(IA1: _AnalogSignal, IB1: _AnalogSignal, IC1: _AnalogSignal,
                                IA2: _AnalogSignal, IB2: _AnalogSignal, IC2: _AnalogSignal,
                                transformer, CT_high=None, CT_low=None):
    """Расчет дифференциального тока и тока торможения для трех фаз двухобмоточного трансформатора
    При расчете учитывается группа и схема соединения трансформатора, осуществляется приведение сторон.
    Диф. ток равен:
        Idif = val1 + val2
    Ток торможения равен:
        Ibias = 0.5(abs(val1.dft()) + abs(val2.dft()))

    :param IA1: аналоговый сигнал тока стороны 1, type: AnalogSignal
    :param IB1: аналоговый сигнал тока стороны 1, type: AnalogSignal
    :param IC1: аналоговый сигнал тока стороны 1, type: AnalogSignal
    :param IA2: аналоговый сигнал тока стороны 2, type: AnalogSignal
    :param IB2: аналоговый сигнал тока стороны 2, type: AnalogSignal
    :param IC2: аналоговый сигнал тока стороны 2, type: AnalogSignal
    :param transformer: силовой трансформатор, type: Transformer
    :param CT_high: трансформатор тока стороны высокого напряжения, type: CT
    :param CT_low: трансформатор тока стороны низкого напряжения, type: CT
    :return: list(IdifA, IdifB, IdifC, IbiasA, IbiasB, IbiasC),
                                    где Idif - мгновенные значения диф. тока, type: AnalogSignal
                                        Ibias - ток торможения, type: AnalogSignal
    """
    if transformer.scheme_high.val == 'Y' and transformer.scheme_low.val == 'D' and transformer.group.val == 11:
        IA1_rot = (IA1 - IB1) / np.sqrt(3)
        IB1_rot = (IB1 - IC1) / np.sqrt(3)
        IC1_rot = (IC1 - IA1) / np.sqrt(3)
        IA2_rot, IB2_rot, IC2_rot = IA2, IB2, IC2
    elif transformer.scheme_high.val == 'Y' and transformer.scheme_low.val == 'D' and transformer.group.val == 1:
        IA1_rot = (IA1 - IC1) / np.sqrt(3)
        IB1_rot = (IB1 - IA1) / np.sqrt(3)
        IC1_rot = (IC1 - IB1) / np.sqrt(3)
        IA2_rot, IB2_rot, IC2_rot = IA2, IB2, IC2
    elif transformer.group.val == 1 and ((transformer.scheme_high.val == 'Y' and transformer.scheme_low.val == 'Y') or
                                         transformer.scheme_high.val == 'D' and transformer.scheme_low.val == 'D'):
        IA1_rot, IB1_rot, IC1_rot = IA1, IB1, IC1
        IA2_rot, IB2_rot, IC2_rot = IA2, IB2, IC2
    else:
        raise ValueError(f'Не поддерживаемая группа соединений трансформатора')
    IdifA, IbiasA = dif_single_phase(IA1_rot, IA2_rot, Inom1=transformer.Ihigh.val, Inom2=transformer.Ilow.val,
                                     CT1=CT_high, CT2=CT_low)
    IdifB, IbiasB = dif_single_phase(IB1_rot, IB2_rot, Inom1=transformer.Ihigh.val, Inom2=transformer.Ilow.val,
                                     CT1=CT_high, CT2=CT_low)
    IdifC, IbiasC = dif_single_phase(IC1_rot, IC2_rot, Inom1=transformer.Ihigh.val, Inom2=transformer.Ilow.val,
                                     CT1=CT_high, CT2=CT_low)
    IdifA.name, IbiasA.name = 'IдифA', 'IтормA'
    IdifB.name, IbiasB.name = 'IдифB', 'IтормB'
    IdifC.name, IbiasC.name = 'IдифC', 'IтормC'
    return IdifA, IdifB, IdifC, IbiasA, IbiasB, IbiasC


def impedance(*args, imin=0.05, vol='phase'):
    """Расчет междуфазных сопротивлений.

    :param args: IA, IB, IC, UAB, UBC, UCA (UA, UB, UC) type: ComplexSignal или AnalogSignal
    :param vol: напряжения фазные (phase), линейные (line) type: str
    :param imin: минимальное значение тока при котором осуществлятся расчет, type: float
    :return: ZAB, ZBC , ZCA type: ComplexSignal
    """
    for arg in args:
        _check_type(arg, (_AnalogSignal, _ComplexSignal))
    _check_type_equality(args)
    Fs = args[0].Fs
    length = len(args[0].val)
    args = _DFT(*args) if type(args[0] == _AnalogSignal) else args
    if len(args) == 6:
        if vol=='line':
            ia, ib, ic, uab, ubc, uca = map(lambda s: s.val, args)
        elif vol=='phase':
            ia, ib, ic, ua, ub, uc = map(lambda s: s.val, args)
            uab, ubc, uca = ua - ub, ub - uc, uc - ua
    elif len(args) == 5:
        ia, ib, ic, uab, ubc = map(lambda s: s.val, args)
        uca = - uab - ubc
    else:
        raise ValueError('Неверное количество аргументов. Должно быть 5 или 6.')
    iab, ibc, ica = ia - ib, ib - ic, ic - ia
    zab = _ComplexSignal(name='Zab', Fs=Fs)
    zab.val = np.where(abs(iab) > imin, uab / (iab + 1e-9), np.full(length, np.nan))
    zbc = _ComplexSignal(name='Zbc', Fs=Fs)
    zbc.val = np.where(abs(ibc) > imin, ubc / (ibc + 1e-9), np.full(length, np.nan))
    zca = _ComplexSignal(name='Zca', Fs=Fs)
    zca.val = np.where(abs(ica) > imin, uca / (ica + 1e-9), np.full(length, np.nan))
    return zab, zbc, zca

def impedance_phase(*args, imin=0.05, line:Line=Line()):
    """Расчет фазных сопротивлений.

    :param args: IA, IB, IC, UA, UB, UC type: ComplexSignal или AnalogSignal
    :param imin: минимальное значение тока при котором осуществляется расчет, type: float
    :param line: модель линии электропередач с параметрами, type: Line
    :return: ZA, ZB , ZC type: ComplexSignal
    """
    for arg in args:
        _check_type(arg, (_AnalogSignal, _ComplexSignal))
    _check_type_equality(args)
    Fs = args[0].Fs
    length = len(args[0].val)
    args = _DFT(*args) if type(args[0] == _AnalogSignal) else args
    if len(args) == 6:
        ia, ib, ic, ua, ub, uc = map(lambda s: s.val, args)
        i03 = ia + ib + ic
        k_comp = ((line.r0.val - line.r1.val) + 1j * (line.x0.val - line.x1.val)) / (3 * (line.r1.val + 1j * line.x1.val))
    else:
        raise ValueError('Неверное количество аргументов. Должно быть 6: IA, IB, IC, UA, UB, UC')
    za = _ComplexSignal(name='Za', Fs=Fs)
    za.val = np.where(abs(ia) > imin, ua / (ia + k_comp * i03 + 1e-9), np.full(length, np.nan))
    zb = _ComplexSignal(name='Zb', Fs=Fs)
    zb.val = np.where(abs(ib) > imin, ub / (ib + k_comp * i03 + 1e-9), np.full(length, np.nan))
    zc = _ComplexSignal(name='Zc', Fs=Fs)
    zc.val = np.where(abs(ic) > imin, uc / (ic + k_comp * i03 + 1e-9), np.full(length, np.nan))
    return za, zb, zc


def impedance_single_phase(i: _Union[_AnalogSignal, _ComplexSignal], u: _Union[_AnalogSignal, _ComplexSignal],
                           imin=0.05, method='fourier')->_ComplexSignal:
    """Расчет сопротивления.

    :param i: ток, type: ComplexSignal или AnalogSignal
    :param u: напряжение, type: ComplexSignal или AnalogSignal
    :param imin: минимальное значение тока при котором осуществлятся расчет, type: float
    :param method: метод расчета: fourier - расчет по Фурье;
                                  sanderson - расчет по уравнению линии, формула Сандерсона

    :return: сопротивление, type: ComplexSignal
    """
    for arg in (i, u):
        _check_type(arg, (_AnalogSignal, _ComplexSignal))
    _check_type_equality((i, u))
    Fs = i.Fs
    z = _ComplexSignal(name='Z', Fs=Fs)
    length = len(i.val)
    if method == 'fourier':
        i, u = _DFT(i, u) if type(i) == _AnalogSignal else (i, u)
        i, u = map(lambda s: s.val, (i, u))
        z.val = np.where(abs(i) > imin, u / (i + 1e-9), np.full(length, np.nan))
    elif method == 'sanderson' and type(i) == _AnalogSignal:
        i, u = map(lambda s: s.val, (i, u))
        N = np.int32(Fs / const.fbase)
        N_length = len(i)
        b_low = np.sqrt(2) * np.sin(2 * np.pi * np.arange(0.5, N, 1) / N) / N
        b_f = np.array((1, 1, 1)) / 2.982889722747621
        b_d = np.array((1, 0, -1)) / 0.261052384440103
        b_dd = np.array((1, -2, 1)) / 0.017110277252379
        # фильтрация гармоник выше 50 Гц синусным фильтром
        u_filt = np.convolve(u, b_low)
        i_filt = np.convolve(i, b_low)
        fu = np.convolve(b_f, u_filt)
        du = np.convolve(b_d, u_filt) * const.w
        fi = np.convolve(b_f, i_filt)
        di = np.convolve(b_d, i_filt) * const.w
        ddi = np.convolve(b_dd, i_filt) * const.w**2
        L = (fu * di - du * fi) / (di * di - ddi * fi)
        R = (du * di - fu * ddi) / (di * di - ddi * fi)
        zcalc = R + 1j * 2 * np.pi * 50 * L
        zcalc = zcalc[0:N_length]
        zcalc[0:N+3] = np.nan
        z.val = np.where(abs(fi[:N_length]+1j*di[:N_length] / const.w) > imin, zcalc, np.full(N_length, np.nan))
    else:
        raise ValueError('Ошибка исходных данных в формуле расчета сопротивления')
    return z


if __name__ == '__main__':
    from relaylab.signals import sin

    U = sin(10, 0, 'U', tmax=0.1)
    I = sin(1, 0, 'I', tmax=0.1)
    impedance_single_phase(U, I)

