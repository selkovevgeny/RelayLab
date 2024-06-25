"""
relaylab.signals
================

Модуль содержит три базовых класса сигналов:
    DiscreteSignal - дискретные сигналы;
    AnalogSignal - аналоговые сигналы с вещественными значениями;
    ComplexSignal - аналоговые сигналы с комплексными значениями.

В составе модуля содержатся функции для генерации сигналов, типичных для энергосистемы (см. функции sin, transient и пр.)

В составе модуля содержатся базовые функции, выполяющие преобрзования сигналов (см. DFT, RMS и пр.)
"""
import copy as _copy
import numpy as np
import relaylab as _relaylab
from scipy.signal import lfilter as _lfilter

class _Const:
    def __init__(self):
        self.__fbase = 50
        self.__w = 2 * np.pi * self.fbase
        self.cycle = 1
        self.a = np.exp(2j * np.pi / 3)

    @property
    def fbase(self):
        return self.__fbase

    @fbase.setter
    def fbase(self, val):
        self.__fbase = val
        self.__w = 2 * np.pi * self.__fbase

    @property
    def w(self):
        return self.__w

const = _Const()


class _Color:
    def __init__(self, color:dict):
        for key, value in color.items():
            self.__setattr__(key, value)


color = {'grey': '#888b8c', 'yellow': '#FFB02E', 'green': '#50C878', 'red': '#FD7C6E', 'blue': '#0095B6',
         'grey_dark': '#293133', 'yellow_deep': '#D5713F', 'green_deep': '#006633', 'red_deep': '#E32636', 'blue_deep': '#1A4876',
         'orange': '#F3A505'}
color = _Color(color) #Доступ к цветам через точку, например: color.blue


def _resample_discrete(channel):
    """Учет того, что обработка сигнала проиходит один раз за программный цикл"""
    return channel[::const.cycle].repeat(const.cycle)


class _Signal:
    """Родительский класс для всех сигналов"""
    _self_types = (np.float_, np.double, np.single, np.intc, np.int_)

    def __init__(self, name='signal', val=None, Fs=2400):
        """Создание сигнала

        :param name: наименование
        :param val: массив значений
        :param Fs: частота дискретизации, Гц
        """
        self.name = name
        self.Fs = Fs
        self.val = np.array([], dtype=self._self_types[0]) if val is None else val

    @property
    def val(self):
        return self.__val

    @val.setter
    def val(self, val):
        if type(val) != np.ndarray:
            raise ValueError(f'Неверный тип данных {type(val)} для класса {self.__class__}. '
                             f'Ожидается тип данных numpy.ndarray.')
        else:
            if val.dtype not in self._self_types:
                raise ValueError(f'Неверный тип данных в  массиве numpy.ndarray {type(val)} для класса {self.__class__}.'
                                 f'Ожидаются типы данных {self._self_types}.')
        self.__val = val

    @property
    def time(self):
        """Метки времени"""
        return np.arange(0, len(self.val)) / self.Fs

    def change_name(self, name):
        """Изменение имени сигнала"""
        self.name = str(name)
        return self

    @property
    def plt_data(self):
        """Данные для построения графиков в plotly.

        return: dict(name=self.name, x=self.time, y=self.val)"""
        return dict(name=self.name, x=self.time, y=self.val)

    def __len__(self):
        """Возвращает количество элементов в массиве ndarray"""
        return len(self.val)

    def __getitem__(self, item):
        """Возвращает срез сигнала S[start:stop], где start и stop - время в секундах."""
        res = self.__class__(name=self.name, Fs=self.Fs)
        start = max(0, int(self.Fs * item.start))
        stop = min(len(self), int(self.Fs * item.stop))
        res.val = self.val[start:stop]
        return res

    def __str__(self):
        """Описание экземпляра класса"""
        return f'{self.name}: Fs= {self.Fs}\n{self.val[0:min(10, len(self.val))]}...'

    def __repr__(self):
        return f'{self.__class__}: {self.name}'


class DiscreteSignal(_Signal):
    """Класс дискретных сигналов"""
    _data_types = (np.bool_, bool)
    _self_types = (np.bool_, )

    def __and__(self, other):
        """Логическое И

        Пример:
        res = discrete_signal1 & discrete_signal2
        """
        if type(other) == self.__class__:
            signal = self.__class__(name=self.name + f'and {other.name}', Fs=self.Fs)
            signal.val = self.val & other.val
        else:
            raise ValueError(f'Не поддерживаемый тип данных: {type(other)}')
        return signal

    def __or__(self, other):
        """Логическое ИЛИ

        Пример:
        res = discrete_signal1 | discrete_signal2
        """
        if type(other) == self.__class__:
            signal = self.__class__(name=self.name + f'or {other.name}', Fs=self.Fs)
            signal.val = self.val | other.val
        else:
            raise ValueError(f'Не поддерживаемый тип данных: {type(other)}')
        return signal

    def __invert__(self):
        """Логическая инверсия сигнала

        Пример:
        res = ~ discrete_signal1
        """
        signal = self.__class__(name=f'not {self.name}', Fs=self.Fs)
        signal.val = ~ self.val
        return signal

    def delay(self, t, delay_type='front'):
        """Задержка срабатывания логичесого сигнала на время t, c.
        По умолчанию задержка на срабатывание, если type==back - задержка на возврат"""
        if delay_type == 'front':
            return delay(self, t=t)
        elif delay_type == 'back':
            return delay_to_return(self, t=t)
        else:
            raise ValueError(f'Не верное значение параметра "delay_type". Возможные значения: "front" и "back"')

    def impulse(self, t, impulse_type='front'):
        """Формирователь импульса длительностью  t, c.

        :param t: длительность импульса, с
        :param impulse_type: тип формирователя: front - формирователь по переднему фронту,
                                                back - формирователь по заднему фронту
                                                both - формирователь по переднему и по заднему фронту
        :return: импульс длительностью t, type: DiscreteSignal
        """
        return impulse_former(self, t, impulse_type=impulse_type)


class _CommonSignal(_Signal):
    _data_types = (float, int, np.float_, np.single, np.double, np.intc, np.int_)
    _self_types = (np.float_, np.double, np.single, np.intc, np.int_)

    def __neg__(self):
        """Инверсия сигнала

        Пример:
        res = - signal
        """
        deep_copy = _copy.deepcopy(self)
        deep_copy.val = - self.val
        deep_copy.name = f'-{self.name}'
        return deep_copy

    def __sub__(self, other):
        """Вычитание сигналов

        Примеры:
        res = signal1 - signal2
        res2 = signal1 - 0.5
        """
        if type(other) in self._data_types:
            signal = self.__class__(name=self.name+f'-{other:2f}', Fs=self.Fs)
            signal.val = self.val - other
        elif type(other) == self.__class__:
            signal = self.__class__(name=self.name + f'-{other.name}', Fs=self.Fs)
            signal.val = self.val - other.val
        else:
            raise ValueError(f'Не поддерживаемый тип данных: {type(other)}')
        return signal

    def __rsub__(self, other):
        """Вычитание сигналов

        Примеры:
        res2 = 0.5 - signal1
        """
        if type(other) in self._data_types:
            signal = self.__class__(name=f'{other:2f}-' + self.name, Fs=self.Fs)
            signal.val = other - self.val
        else:
            raise ValueError(f'Не поддерживаемый тип данных: {type(other)}')
        return signal

    def __add__(self, other):
        """Сложение сигналов

        Примеры:
        res = signal1 + signal2
        res2 = signal1 + 0.5
        """
        if type(other) in self._data_types:
            signal = self.__class__(name=self.name+f'+{other:2f}', Fs=self.Fs)
            signal.val = self.val + other
        elif type(other) == self.__class__:
            signal = self.__class__(name=self.name + f'+{other.name}', Fs=self.Fs)
            signal.val = self.val + other.val
        else:
            raise ValueError(f'Не поддерживаемый тип данных: {type(other)}')
        return signal

    def __radd__(self, other):
        """Сложение сигналов

        Примеры:
        res2 = 0.5 + signal1
        """
        if type(other) in self._data_types:
            signal = self.__class__(name=f'{other:2f}+' + self.name, Fs=self.Fs)
            signal.val = self.val + other
        else:
            raise ValueError(f'Не поддерживаемый тип данных: {type(other)}')
        return signal

    def __mul__(self, other):
        """Умножение сигналов

        Примеры:
        res = signal1 * signal2
        res2 = signal1 * 0.5
        """
        if type(other) in self._data_types:
            signal = self.__class__(name=self.name+f'*{other:2f}', Fs=self.Fs)
            signal.val = self.val * other
        elif type(other) == self.__class__:
            signal = self.__class__(name=self.name + f'*{other.name}', Fs=self.Fs)
            signal.val = self.val * other.val
        else:
            raise ValueError(f'Не поддерживаемый тип данных: {type(other)}')
        return signal

    def __rmul__(self, other):
        """Умножение сигналов

        Примеры:
        res2 = 0.5 * signal1
        """
        if type(other) in self._data_types:
            signal = self.__class__(name=f'{other:2f}*' + self.name, Fs=self.Fs)
            signal.val = self.val * other
        else:
            raise ValueError(f'Не поддерживаемый тип данных: {type(other)}')
        return signal

    def __truediv__(self, other):
        """Деление сигналов

        Примеры:
        res = signal1 / signal2
        res2 = signal1 / 2
        """
        if type(other) in self._data_types:
            signal = self.__class__(name=self.name+f'/{other:2f}', Fs=self.Fs)
            signal.val = self.val / other
        elif type(other) == self.__class__:
            signal = self.__class__(name=self.name + f'/{other.name}', Fs=self.Fs)
            signal.val = np.where(abs(other.val)>0.05, self.val / (other.val + 1e-9), np.full(len(self.val), np.nan))
        else:
            raise ValueError(f'Не поддерживаемый тип данных: {type(other)}')
        return signal

    def __rtruediv__(self, other):
        """Деление сигналов

        Примеры:
        res2 = 5 / signal1
        """
        if type(other) in self._data_types:
            signal = self.__class__(name=f'{other:2f}/' + self.name, Fs=self.Fs)
            signal.val = other / self.val
        else:
            raise ValueError(f'Не поддерживаемый тип данных: {type(other)}')
        return signal


class AnalogSignal(_CommonSignal):
    """Класс аналоговых сигналов

    Parameters:
    name: наименование
    val: массив значений ndarray
    Fs: частота дискретизации, Гц
    """
    def rms(self):
        """Расчет среднекадратичного значения сигнала."""
        return RMS(self)

    def dft(self, harm=1):
        """Расчет комплексного значения сигнала, выбранной гармоники."""
        return DFT(self, harm=harm)

    def dft_abs(self, harm=1):
        """Расчет модуля комплексного значения сигнала, выбранной гармоники."""
        return DFT(self, harm=harm).abs()

    def noise(self, snr=20):
        """Функция добавления шума в cигнал"""
        return noise_snr(self, snr=snr)

    def max_relay(self, setting, return_coef=0.95):
        """Пусковой орган максимального типа"""
        relay = _relaylab.characteristics.MaxRelay(setting=setting, return_coef=return_coef)
        return relay.start(self)

    def min_relay(self, setting, return_coef=1.05):
        """Пусковой орган максимального типа"""
        relay = _relaylab.characteristics.MinRelay(setting=setting, return_coef=return_coef)
        return relay.start(self)

    def diff(self):
        """Возвращает приращение сигнала за период промышленной частоты"""
        res = AnalogSignal(name=f'Δ({self.name})', Fs=self.Fs)
        N = int(self.Fs / const.fbase)
        res.val = np.concatenate([np.zeros(2*N), self.val[2*N:] - self.val[N:-N]])
        return res

    def __gt__(self, other):
        """Сравнение с уставкой

        Пример:
        res = signal1 > 5.0
        """
        if type(other) in self._data_types:
            return self.max_relay(other)
        else:
            raise ValueError(f'Не поддерживаемый тип данных: {type(other)}')

    def __lt__(self, other):
        """Сравнение с уставкой

        Пример:
        res = signal1 < 80.0
        """
        if type(other) in self._data_types:
            return self.min_relay(other)
        else:
            raise ValueError(f'Не поддерживаемый тип данных: {type(other)}')

    def dfilter(self, b, a=1):
        """Фильтрация сигнала. Передаточная характеристика задается коэффициентами b и a:
        H(z) = (b[0] + b[1]*z^-1 +..+ b[n]*z^-n) / (1 + a[1]*z^-1 +..+ a[n]*z^-n)

        :param b: коэфициенты цифрового фильтра, type: numpy ndarray
        :param a: коэфициенты цифрового фильтра, type: numpy ndarray
        :return: сигнал класса ComplexSignal, если b - комплекные значения,
                               AnalogSignal, если b - вещественные значения
        """
        return dfilter(self, b=b, a=a)


class ComplexSignal(_CommonSignal):
    """Класс комлексных сигналов"""
    _data_types = (float, int, np.single, np.double, np.intc, np.int_,
                   np.complex_, complex, np.complex128, np.csingle, np.cdouble, np.clongdouble)
    _self_types = (np.complex_, np.complex128, np.csingle, np.cdouble, np.clongdouble)

    def abs(self):
        """Возвращает модуль комплексных значений"""
        res = AnalogSignal(name=f'abs({self.name })', Fs=self.Fs)
        res.val = np.abs(self.val)
        return res

    def real(self):
        """Возвращает вещественную часть комплексных значений"""
        res = AnalogSignal(name=f'real({self.name })', Fs=self.Fs)
        res.val = np.real(self.val)
        return res

    def imag(self):
        """Возвращает мнимую часть комплексных значений"""
        res = AnalogSignal(name=f'imag({self.name })', Fs=self.Fs)
        res.val = np.imag(self.val)
        return res

    def __abs__(self):
        """Возвращает модуль комплексных значений"""
        return self.abs()

    def angle(self):
        """Возвращает угол в радианах"""
        res = AnalogSignal(name=f'ang({self.name })', Fs=self.Fs)
        res.val = np.angle(self.val)
        return res

    def diff(self, abs=False):
        """Возвращает приращение сигнала за период промышленной частоты"""
        res = AnalogSignal(name=f'Δ({self.name})', Fs=self.Fs)
        N = int(self.Fs / const.fbase)
        if abs:
            res.val = np.concatenate([np.zeros(2 * N), np.abs(self.val[2 * N:]) - np.abs(self.val[N:-N])])
        else:
            res.val = np.concatenate([np.zeros(2*N), np.abs(self.val[2*N:] - self.val[N:-N])])
        return res

    def angle_deg(self):
        """Возвращает угол в градусах"""
        res = AnalogSignal(name=f'ang({self.name })', Fs=self.Fs)
        res.val = np.rad2deg(np.angle(self.val))
        return res

    @property
    def plt_data(self):
        """Данные для построения графиков в plotly.

        return: dict(name=self.name, x=self.time, y=self.val)"""
        return dict(name=self.name, x=self.time, y=np.abs(self.val))


def impulses(vals, t, name='signal', Fs=2400):
    """Генерация импульсов аналогового сигнала амплитудой vals в моменты времени t

    :param vals: амплитуда сигнала, type: list
    :param t: список меток времени, type: list
    :param name: наименование сигнала
    :param Fs: частота дискретизации, Гц
    :return: аналоговый сигнал класса AnalogSignal

    Пример:
    impulses(vals=[0, 220, 0, 0], t=[0, 0.1, 0.11, 0.2], name='signal', Fs=2400)
    создается импульс 220 В в промежуток времени от 0.1 до 0.11
    """

    res = AnalogSignal(name=name, Fs=Fs)
    time = np.arange(0, t[-1], 1/Fs)
    signal = np.zeros(len(time))
    signal1 = np.ones(len(time))
    for tst, tend, val in zip(t[0:-1], t[1:], vals[0:len(t)-1]):
        signal += signal1 * ((time >= tst) & (time < tend)) * val
    signal[-1] = signal[-2]
    res.val = signal
    return res


def sin(val, fi, name='signal', f=50, tmax=1.0, tstart=0, tstop=None, Fs=2400):
    """Задание синусоидального сигнала

    :param val: действующее значение синусоидального сигнала
    :param fi: фаза синусоидального сигнала, град
    :param name: наименование сигнала
    :param f: частота сигнала, Гц
    :param tmax: длительность сигнала, с
    :param tstart: время начала синусоиды, с
    :param tstop: время окончания синусоиды, с
    :param Fs: частота дискретизации, Гц
    :return: аналоговый сигнал класса AnalogSignal
    """
    res = AnalogSignal(name=name, Fs=Fs)
    time = np.arange(0, tmax, 1/Fs)
    tstop = tmax if tstop is None else tstop
    res.val = val * np.sqrt(2) * np.sin(2 * np.pi * f * time + np.deg2rad(fi)) * ((time >= tstart) & (time <= tstop))
    return res


def transient(val_load, fi_load, val_fault, fi_fault, tau, name='signal', tfault=0.1, tstop=1.0, tmax=1.0, f=50, Fs=2400):
    """Задание сигнала тока, состоящего из двух частей:
    1 - предаварийный режим - синусоидальный сигнал
    2 - аварийный режим - синусоидальный сигнал и апериодика

    :param val_load: действующее значение тока в предаварийном режиме
    :param fi_load: фаза сигнала в предаварийном режиме, град
    :param val_fault: действующее значение тока в аварийном режиме
    :param fi_fault: фаза сигнала в аварийном режиме, град
    :param tau: постоянная времени апериодической составляющей тока в аварийном режиме
    :param name: наименование сигнала
    :param tfault: время возникновения аварийного режима, с
    :param tstop: время прекращения тока, с
    :param tmax: длительность сигнала, с
    :param f: частота сигнала, Гц
    :param Fs: частота дискретизации, Гц
    :return: аналоговый сигнал класса AnalogSignal
    """
    res = AnalogSignal(name=name, Fs=Fs)
    Atau = val_load * np.sqrt(2) * np.sin(2 * np.pi * f * tfault + np.deg2rad(fi_load)) - np.sqrt(
        2) * val_fault * np.sin(2 * np.pi * f * tfault + np.deg2rad(fi_fault))
    time = np.arange(0, tmax, 1 / Fs)
    val = (val_load * np.sqrt(2) * np.sin(2 * np.pi * f * time + np.deg2rad(fi_load)) * (time <= tfault) +
                 (val_fault * np.sqrt(2) * np.sin(2 * np.pi * f * time + np.deg2rad(fi_fault)) + Atau * np.exp(
                     -(time - tfault) / tau)) * (time > tfault))
    if tstop < tmax:
        nstop = int(tstop * Fs)
        nmax = len(val)
        dn = int(Fs/f/2) + 1 #пол периода+1 отсчет
        n_cross = np.argmin(np.abs(val[nstop:min(nstop+dn, nmax)]))
        val[nstop + n_cross:] = 0
    res.val = val
    return res


def DFT(*signals, harm=1, rot=False):
    """Расчет комплексных значений сигналов, выбранной гармоники.

    :param signals: аналоговые сигналы класса AnalogSignal
    :param harm: номер гармоники сигнала
    :return: список комплексных сигналов класса ComplexSignal
    """
    res_array = []
    Fs = signals[0].Fs
    N = np.int32(Fs / const.fbase)
    N_length = len(signals[0].val)
    bsin = np.sqrt(2) * np.sin(2 * np.pi * harm * np.arange(0.5, N, 1) / N) / N
    bcos = np.sqrt(2) * np.cos(2 * np.pi * harm * np.arange(0.5, N, 1) / N) / N
    for signal in signals:
        Ssin = np.convolve(signal.val, bsin)
        Scos = np.convolve(signal.val, bcos)
        Sfourier = Scos + 1j * Ssin
        Sfourier = Sfourier[0:-N + 1]
        Sfourier[0:N] = 0
        if not rot:
            rot_vector = np.exp(- 2j * np.pi * np.arange(0, N_length) / N * harm) #остановка вектора
            Sfourier = Sfourier * rot_vector
        cor = np.exp(-1j * np.pi * 50 / Fs * harm + 1j * np.pi / 2)
        Sfourier = Sfourier * cor #угловая коррекция вектора
        res = ComplexSignal(name='DFT ' + signal.name, Fs=signal.Fs)
        res.val = Sfourier
        res_array.append(res)
    return res_array[0] if len(res_array) == 1 else res_array


def RMS(*signals):
    """Расчет среднеквадратичных значений сигналов.

    :param signals: аналоговые сигналы класса AnalogSignal
    :return: список аналоговых сигналов класса AnalogSignal
    """
    res_array = []
    N = np.int32(signals[0].Fs / const.fbase)
    window = np.ones(N) / N
    for signal in signals:
        Srms = np.sqrt(np.convolve(signal.val * signal.val, window))
        Srms = Srms[0:-N + 1]
        Srms[0:N] = 0
        res = AnalogSignal(name='RMS ' + signal.name, Fs=signal.Fs)
        res.val = Srms
        res_array.append(res)
    return res_array[0] if len(res_array) == 1 else res_array


def ct_tf(*signals, ct, Kr=0):
    """Расчет вторичного тока трансформатора тока методом ПХН.

    :param signals: аналоговые сигналы класса AnalogSignal
    :param ct: модель трансформатора тока
    :param Kr: остаточная индукция от -1 до 1
    :return: список аналоговых сигналов класса AnalogSignal
    """
    res_array = []
    for signal in signals:
        # Исходные данные
        Fs = signal.Fs
        w = 2 * np.pi * 50
        r2, x2 = ct.r2.val + ct.rload.val, ct.x2.val + ct.xload.val
        z2nom = np.sqrt((ct.r2.val + ct.rnom.val)**2 + (ct.x2.val + ct.xnom.val)**2)
        L2 = x2 / w
        Kpriv = w / (np.sqrt(2) * ct.I1faultnom.val * z2nom)
        # Исходные сигналы
        i1 = signal.val * Kpriv
        didt = np.diff(i1, append=i1[-1]) * Fs
        # Расчетные сигналы
        i2 = np.array([])
        # Инициализация
        flux_point = Kr
        i2_point = 0
        is_saturated = False
        i0_prev = 0
        for i_point, didt_point in zip(i1, didt):
            flux_point = flux_point + (L2 * didt_point + r2*i_point)/Fs
            # Насыщение наступает при потоке больше 1 или меньше -1
            if flux_point > 1:
                flux_point = 1
                is_saturated = True
            elif flux_point < -1:
                flux_point = -1
                is_saturated = True
            if is_saturated:
                i2_point = 0 if L2 == 0 or (1 - r2/Fs/L2) < 0 else i2_point * (1 - r2/Fs/L2)
            else:
                i2_point = i_point
            i0 = i_point - i2_point
            # Выход из насыщения при изменении знака тока намагничивания
            if np.sign(i0) != np.sign(i0_prev):
                is_saturated = False
                i2_point = i_point
            i0_prev = i0
            i2 = np.append(i2, i2_point)
        i2_res = AnalogSignal(name=signal.name + ' втор.', Fs=signal.Fs)
        i2_res.val = i2 / Kpriv
        res_array.append(i2_res)
    return res_array[0] if len(res_array) == 1 else res_array


def noise_snr(*signals, snr=20):
    """ Функция добавления шума в исходный сигнал. Шум рассчитывается исходя из мощности входного сигнала.

    :param signals: аналоговые сигналы класса AnalogSignal
    :param snr: signal-to-noise ratio, отношение сигнал/шум
    :return: список аналоговых сигналов класса AnalogSignal
    """
    res_array = []
    for signal in signals:
        # Signal in watts
        signal_watts = signal.val ** 2
        # Calculate signal power and convert to dB
        signal_avg_watts = np.mean(signal_watts)
        signal_avg_db = 10 * np.log10(signal_avg_watts)
        # Calculate noise then convert to watts
        noise_avg_db = signal_avg_db - snr
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        # Generate an sample of white noise
        mean_noise = 0
        noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(signal.val))
        # Noise up the original signal
        res = AnalogSignal(name='noised ' + signal.name, Fs=signal.Fs)
        res.val = signal.val + noise
        res_array.append(res)
    return res_array[0] if len(res_array) == 1 else res_array


def dfilter(*signals, b, a=1):
    """Фильтрация сигнала. Передаточная характеристика задается коэффициентами b и a:
    H(z) = (b[0] + b[1]*z^-1 +..+ b[n]*z^-n) / (1 + a[1]*z^-1 +..+ a[n]*z^-n)

    :param signals: аналоговые сигналы класса AnalogSignal
    :param b: коэфициенты цифрового фильтра, type: numpy ndarray
    :param a: коэфициенты цифрового фильтра, type: numpy ndarray
    :return: список комплексных сигналов класса ComplexSignal, если b - комплекные значения,
                                                AnalogSignal, если b - вещественные значения
    """
    res_array = []
    N = len(b)
    for signal in signals:
        signal_filtered = _lfilter(b, a, signal.val)
        signal_filtered[0:N] = 0
        if (b.dtype in ComplexSignal._self_types) or signal.__class__ == ComplexSignal:
            res = ComplexSignal(name=f'Фильтр({signal.name})', Fs=signal.Fs)
        else:
            res = AnalogSignal(name=f'Фильтр({signal.name})', Fs=signal.Fs)
        res.val = signal_filtered
        res_array.append(res)
    return res_array[0] if len(res_array) == 1 else res_array


def delay(signal: DiscreteSignal, t):
    """ Задержка на срабатывание логического сигнала

    :param signal: логический сигнал, type: DiscreteSignal
    :param t: время задержки, с
    :return: задержанный логический сигнал, type: DiscreteSignal
    """
    Fs = signal.Fs
    dn = np.int(np.ceil(t * Fs))
    val = signal.val
    delayed_val = val.copy()
    with np.nditer(delayed_val, op_flags=['readwrite']) as it:
        for num, x in enumerate(it):
            x[...] = True if np.all(val[max(0, num - dn):num + 1]) else False
    res = DiscreteSignal(name=f'{signal.name} + tср={t:.3f},c', val=delayed_val, Fs=signal.Fs)
    return res


def delay_to_return(signal: DiscreteSignal, t):
    """ Задержка на возврат логического сигнала

    :param signal: логический сигнал, type: DiscreteSignal
    :param t: время на возврат, с
    :return: задержанный логический сигнал, type: DiscreteSignal
    """
    Fs = signal.Fs
    dn = np.int(np.ceil(t * Fs))
    val = signal.val
    val_int = np.array(signal.val, dtype=np.int_)
    dif = np.diff(val_int, prepend=val[0])
    imp = np.where(dif < 0, True, False)
    imp_val = imp.copy()
    with np.nditer(imp_val, op_flags=['readwrite']) as it:
        for num, x in enumerate(it):
            x[...] = True if np.any(imp[max(0, num - dn):num + 1]) else False
    res_val = val | imp_val
    res = DiscreteSignal(name=f'{signal.name} + tв={t:.3f},c', val=res_val, Fs=signal.Fs)
    return res


def impulse_former(signal: DiscreteSignal, t, impulse_type='front'):
    """Формирователь импульса длительностью  t, c. 
    
    :param signal: логический сигнал, type: DiscreteSignal
    :param t: длительность импульса, с
    :param type: тип формирователя: front - формирователь по переднему фронту, back - формирователь по заднему фронту
                                    both - формирователь по переднему и по заднему фронту
    :return: импульс длительностью t, type: DiscreteSignal
    """
    val = np.array(signal.val, dtype=np.int_)
    dif = np.diff(val, prepend=val[0])
    if impulse_type == 'front':
        imp = np.where(dif>0, True, False)
    elif impulse_type == 'back':
        imp = np.where(dif<0, True, False)
    elif impulse_type == 'both':
        imp = np.array(dif, dtype=np.bool_)
    else:
        raise ValueError(f'Не верное значение параметра "impulse_type". Возможные значения: "front", "back" и "both"')
    imp_ds = DiscreteSignal(name='Импульс', val=imp, Fs=signal.Fs)
    return delay_to_return(imp_ds, t).change_name(f'{signal.name} {impulse_type} {t:.3f},c')


def discrete2analog(*signals):
    """Преобразование дискретных сигналов в аналоговые.

    :param signals: дискретные сигналы класса DiscreteSignal
    :return: список аналоговых сигналов класса AnalogSignal
    """

    res_array = []
    for signal in signals:
        res = AnalogSignal(name=signal.name, Fs=signal.Fs)
        res.val = np.array(signal.val, dtype=np.float_)
        res_array.append(res)
    return res_array[0] if len(res_array) == 1 else res_array


if __name__ == '__main__':
    # import plotly.graph_objects as go
    # i1 = transient(name='IA', val_load=1, fi_load=0, val_fault=10, fi_fault=0, tau=0.03, tfault=0.04, tmax=0.1,
    #                  f=50, Fs=1200)
    # i2 = transient(name='IB', val_load=1, fi_load=0, val_fault=10, fi_fault=-90, tau=0.03, tfault=0.04, tmax=0.1,
    #                  f=50, Fs=1200)
    # i1_dft, i2_dft = DFT(i1, i2)
    # kfi = np.exp(-1j * np.pi * 50 / 2400)
    # print(i1_dft.angle_deg().val)
    # a = ComplexSignal(name='Ia', val=np.array([1j, 10]))
    # a.val = np.array([1j, 1, 2])
    # print(a)

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(name=i1.name, x=i1.time, y=i1.noise().dft().abs().val,
    #                          mode="lines+markers", line=dict(color=color.blue)))
    # fig.add_trace(go.Scatter(name=i2.name, x=i2.time, y=i2.val,
    #                          mode="lines+markers", line=dict(color=color.green)))
    # fig.show()

    # i = transient(name='I', val_load=0, fi_load=0, val_fault=1000, fi_fault=0, tau=0.05, tfault=0.04,
    #               tmax=0.2, tstop=0.15, f=50, Fs=1200)
    imp = impulses(vals=[0, 220, 0, 0], t=[0, 0.1, 0.15, 0.2], name='signal', Fs=100)
    sin_signal = sin(1, 0, tmax=0.1, tstart=0.02, tstop=0.06, Fs=1000)
    print(sin_signal.val)

