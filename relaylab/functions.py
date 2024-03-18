"""
relaylab.functions
================

Модуль содержит функции расчета, получающие на вход несколько сигналов, например, расчет симметричных составляющих,
расчет дифференциальных токов и пр.
"""

from relaylab.signals import AnalogSignal as _AnalogSignal, ComplexSignal as _ComplexSignal, DFT as _DFT
import numpy as np
from relaylab.signals import const


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
        Ibias.val = (np.abs(val1) + np.abs(val2)).val / 2
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


if __name__ == '__main__':
    from relaylab.signals import sin

    Ia = sin(1, 0, 'Ia', tmax=0.1)
    Ib = sin(1, 240, 'Ib', tmax=0.1)
    Ic = sin(1, 120, 'Ic', tmax=0.1)
    I1, I2, I0 = symmetrical_comp(Ia, Ib, Ic)
    print(abs(I1.val[49]), abs(I2.val[49]), abs(I0.val[49]))
    print(Ia.dft().val[49] / Ia.dft().val[49], Ib.dft().val[49] / Ia.dft().val[49], Ic.dft().val[49] / Ia.dft().val[49])
