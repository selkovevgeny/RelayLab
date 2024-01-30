"""
relaylab.characteristics
================

Модуль содержит характеристики срабатывания защит.
"""
import numpy as np
from relaylab.signals import AnalogSignal as _AnalogSignal, ComplexSignal as _ComplexSignal, \
    DiscreteSignal as _DiscreteSignal
from relaylab.signals import const


class _Relay:
    def __init__(self):
        pass


class DifRelay(_Relay):
    """Характеристика срабатывания дифференциальной защиты с двумя участками торможения"""

    def __init__(self, start=0.2, slope_st1=0.5, slope1=0.2, slope_st2=1.5, slope2=0.5):
        """ Задание характеристики срабатывания

        :param start: начальный ток срабатывания
        :param slope_st1: ток начала торможения участка 1
        :param slope1: коэффициент наклона участка 1
        :param slope_st2: ток начала торможения участка 2
        :param slope2: коэффициент наклона участка 2
        """
        super().__init__()
        self.start = start
        self.slope_st1 = slope_st1
        self.slope1 = slope1
        self.slope_st2 = slope_st2
        self.slope2 = slope2

    def get_points(self, bias_max=10):
        """Возвращает координаты точек характеристики торможения на плоскости: Ibias, Idif"""
        bias = np.array((0, self.slope_st1, self.slope_st2, bias_max))
        dif = np.array((self.start, self.start, self.start + self.slope1 * (self.slope_st2 - self.slope_st1),
                      self.start + self.slope1 * (self.slope_st2 - self.slope_st1) + self.slope2 * (bias_max - self.slope_st2)))
        return bias, dif


class MaxRelay(_Relay):
    def __init__(self, setting, return_coef=0.95):
        """Инициализация максимального реле

        :param setting: уставка срабатывания, type:float
        :param return_сoef: коэффициент возврата, type:float
        """
        super().__init__()
        self.setting = setting
        self.return_coef = return_coef

    def start(self, signal: _AnalogSignal):
        """Пусковой орган максимального типа. Срабатывание происходит при превышении уставки срабатывания,
        возврат при снижении ниже уставки, умноженной на коэффициент возврата.

        :param signal: входной сигнал, type: AnalogSignal
        :return: логический сигнал, type: DiscreteSignal
        """
        vals = signal.val[::const.cycle]
        st = False
        st_list = []
        for val in vals:
            if val >= self.setting:
                st = True
            elif val < self.setting * self.return_coef:
                st = False
            st_list.append(st)
        arr = np.array(st_list).repeat(const.cycle)[0:len(signal)]
        start = _DiscreteSignal(name=f'{signal.name} > {self.setting}', val=arr, Fs=signal.Fs)
        return start


class MinRelay(_Relay):
    def __init__(self, setting, return_coef=1.05):
        """Инициализация минимального реле

        :param setting: уставка срабатывания, type:float
        :param return_сoef: коэффициент возврата, type:float
        """
        super().__init__()
        self.setting = setting
        self.return_coef = return_coef

    def start(self, signal: _AnalogSignal):
        """Пусковой орган минимального типа. Срабатывание происходит при снижении сигнала ниже уставки срабатывания,
        возврат при повышении сигнала выше уставки, умноженной на коэффициент возврата.

        :param signal: входной сигнал, type: AnalogSignal
        :return: логический сигнал, type: DiscreteSignal
        """
        vals = signal.val[::const.cycle]
        st = False
        st_list = []
        for val in vals:
            if val <= self.setting:
                st = True
            elif val > self.setting * self.return_coef:
                st = False
            st_list.append(st)
        arr = np.array(st_list).repeat(const.cycle)[0:len(signal)]
        arr[0:int(signal.Fs/const.fbase)] = False
        start = _DiscreteSignal(name=f'{signal.name} < {self.setting}', val=arr, Fs=signal.Fs)
        return start