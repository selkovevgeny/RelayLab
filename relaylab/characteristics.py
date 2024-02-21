"""
relaylab.characteristics
================

Модуль содержит характеристики срабатывания защит.
"""
import numpy as np
from relaylab.signals import AnalogSignal as _AnalogSignal, ComplexSignal as _ComplexSignal, \
    DiscreteSignal as _DiscreteSignal
from relaylab.signals import const
from relaylab.signals import _resample_discrete


class _Relay:
    def __init__(self):
        pass

    @staticmethod
    def _check_type(vals, types):
        """Проверка соответствия типа данных. В случае несоответсвия выдается ошибка"""
        if type(vals) not in (list, tuple):
            vals = (vals, )
        if type(types) not in (list, tuple):
            types = (types, )
        for val in vals:
            if type(val) not in types:
                raise f'Не подходящий аргумент функции. Ожидается: {types}, получен: {type(val)}'

    @staticmethod
    def _is_type(val, types):
        """Проверка соответствия типа данных. Возвращает bool"""
        if type(types) not in (list, tuple):
            types = (types, )
        return type(val) in types


class DifRelay(_Relay):
    """Характеристика срабатывания дифференциальной защиты с двумя участками торможения"""

    def __init__(self, st0=0.2, slope_st1=0.5, slope1=0.2, slope_st2=1.5, slope2=0.5):
        """ Задание характеристики срабатывания

        :param st0: начальный ток срабатывания
        :param slope_st1: ток начала торможения участка 1
        :param slope1: коэффициент наклона участка 1
        :param slope_st2: ток начала торможения участка 2
        :param slope2: коэффициент наклона участка 2
        """
        super().__init__()
        self.st0 = st0
        self.slope_st1 = slope_st1
        self.slope1 = slope1
        self.slope_st2 = slope_st2
        self.slope2 = slope2

    def get_points(self, bias_max=10):
        """Возвращает координаты точек характеристики торможения на плоскости: Ibias, Idif"""
        bias = np.array((0, self.slope_st1, self.slope_st2, bias_max))
        dif = np.array((self.st0, self.st0, self.st0 + self.slope1 * (self.slope_st2 - self.slope_st1),
                      self.st0 + self.slope1 * (self.slope_st2 - self.slope_st1) + self.slope2 * (bias_max - self.slope_st2)))
        return bias, dif

    def start(self, *signals, dif_dft=True):
        """Пусковой орган с торможением. Срабатывание происходит при превышении уставки срабатывания с учетом торможения

        :param signals: кортеж входных сигналов (Idif, Ibias) или (Idif1, ... , Idifn, Ibias1, ..., Ibiasn),
                type: tuple of AnalogSignal
                        Idif: мгновенные значения тока, type: AnalogSignal
                        Ibias: действущее значение тока, type: AnalogSignal
        :param dif_dft: флаг делать DFT диффеенциального тока или нет
        :return: логические сигналы, type: tuple of DiscreteSignal
        """
        length = len(signals)
        if length % 2:
            raise 'Количество входных аргументов должно быть четное число в формате: ' \
                  '(Idif1, ... , Idifn, Ibias1, ..., Ibiasn)'
        Idifs, Ibiases = signals[0:length//2], signals[length//2:]
        self._check_type(Idifs, _AnalogSignal)
        self._check_type(Idifs, _AnalogSignal)
        res_arr = []
        for Idif, Ibias in zip(Idifs, Ibiases):
            Idif_val = Idif.dft_abs().val if dif_dft else Idif.val
            Ibias_val = Ibias.val
            start = (Idif_val > self.st0) & \
                  (Idif_val > (Ibias_val * self.slope1 + self.st0 - self.slope1 * self.slope_st1)) & \
                  (Idif_val > (Ibias_val * self.slope2 + self.st0 + self.slope1 *
                               (self.slope_st2 - self.slope_st1) - self.slope2 * self.slope_st2))
            start_resampled = _resample_discrete(start)
            res_arr.append(_DiscreteSignal(name=f'{Idif.name}>', val=start_resampled, Fs=Idif.Fs))
        return res_arr[0] if len(res_arr) == 1 else tuple(res_arr)


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


if __name__ == '__main__':
    relay = DifRelay(st0=0.2, slope_st1=0.5, slope1=0.2, slope_st2=1.5, slope2=0.5)
    Idif = _AnalogSignal(val=np.array([0.199, 0.201, 0.199, 0.201, 0.299, 0.301, 0.399, 0.401, 0.899, 0.901]))
    Ibias = _AnalogSignal(val=np.array([0, 0, 0.5, 0.5, 1, 1, 1.5, 1.5, 2.5, 2.5]))
    res = all(relay.start(Idif, Ibias, dif_dft=False).val | ~ np.array([False, True, False, True, False, True, False, True, False, True]))
    print(res)