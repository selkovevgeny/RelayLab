"""
relaylab.equipment
================

Модуль содержит модели электрооборудования и расчет параметров электрооборудования.
"""

import numpy as np
import pandas as _pd
from relaylab.signals import const


class _Var:
    __data_types = (float, int, np.single, np.double, np.intc, np.int_)
    # Допустимые единицы измерения
    __mul = {'': 1, 'о.е.': 1, 'с': 1, 'мс': 1e-3, 'гр': 1, 'мкФ': 1e-6, 'Гн': 1,
             'А': 1, 'В': 1, 'Ом': 1, 'ВА': 1, 'Вт': 1, 'вар': 1,
             'кА': 1000, 'кВ': 1000, 'кОм': 1000, 'кВА': 1000, 'кВт': 1000, 'квар': 1000,
             'МВт': 1e6, 'МВА': 1e6, 'Мвар': 1e6,
             'км': 1000, 'Ом/км': 0.001}

    def __init__(self, val=None, name='', desc='', unit='', n_digits=2):
        self.val = val * self.__mul[unit]
        self.name = name
        self.desc = desc
        self.unit = unit
        self.n_digits = n_digits

    def __add__(self, other):
        if type(other) == _Var:
            return self.val + other.val
        elif type(other) in self.__data_types:
            return self.val + other
        else:
            raise ValueError(f'Неверный тип данных: {other}')

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if type(other) == _Var:
            return self.val - other.val
        elif type(other) in self.__data_types:
            return self.val - other
        else:
            raise ValueError(f'Неверный тип данных: {other}')

    def __rsub__(self, other):
        return (self - other) * (-1)

    def __truediv__(self, other):
        if type(other) == _Var:
            return self.val / other.val
        elif type(other) in self.__data_types:
            return self.val / other
        else:
            raise ValueError(f'Неверный тип данных: {other}')

    def __rtruediv__(self, other):
        if type(other) == _Var:
            return other.val / self.val
        elif type(other) in self.__data_types:
            return other / self.val
        else:
            raise ValueError(f'Неверный тип данных: {other}')

    def __mul__(self, other):
        if type(other) == _Var:
            return self.val * other.val
        elif type(other) in self.__data_types:
            return self.val * other
        else:
            raise ValueError(f'Неверный тип данных: {other}')

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power, modulo=None):
        if type(power) in self.__data_types:
            return self.val ** power
        else:
            raise ValueError(f'Неверный тип данных: {power}')

    def __str__(self):
        return f'{self.name} = {round(self.val / self.__mul[self.unit], self.n_digits)} - {self.desc}, {self.unit}'

    def __repr__(self):
        return f'{str(self.__class__)}: {self.name}'


class _StrVar:
    def __init__(self, val=None, name='', desc=''):
        self.val = val
        self.name = name
        self.desc = desc

    def __str__(self):
        return f'{self.name} = {self.val} - {self.desc}'

    def __repr__(self):
        return f'{str(self.__class__)}: {self.name}'


class Equipment:
    def __str__(self):
        """Вывод описания экземлпяра класса при использовании функции
        print(equipment instance)"""
        res = ''
        for param in self.__dict__:
            if param[0] != '_':
                res = res + str(param) + ': ' + str(self.__getattribute__(param)) + '\n'
        return res


class Transformer(Equipment):
    """Модель трансформатора:
    S - мощность, кВА;
    Uhigh - напряжение стороны ВН, кВ;
    Ulow - напряжение обмотки НН, кВ;
    Usc - напряжение короткого замыкания, о.е.;
    Inl - ток холостого хода, о.е.;
    Psc - активные потери в опыте короткого замыкания, кВт;
    Pnl - активные потери в опыте холостого хода, кВт;
    scheme_high - схема соединения обмотки ВН;
    scheme_low - схема соединения обмотки НН;
    group - группа соединения трансформатора;

    Расчетные величины:
    Ihigh - номинальный ток стороны ВН, А;
    Ilow - номинальный ток стороны НН, А;
    Zsc - сопротивление трансформатора в режиме КЗ, Ом;
    Rsc - активное сопротивление трансформатора в режиме КЗ, Ом;
    Xsc - реактивное сопротивление трансформатора в режиме КЗ, Ом.
    """

    def __init__(self, S, Uhigh, Ulow, Usc=0.105, Inl=0.1, Psc=0.1, Pnl=0.1, scheme_high='Y', scheme_low='D', group=11):
        self.S = _Var(val=S, name='Sтр', desc='мощность силового трансформатора', unit='кВА', n_digits=0)
        self.Uhigh = _Var(val=Uhigh, name='Uвн', desc='напряжение обмотки ВН', unit='кВ', n_digits=0)
        self.Ulow = _Var(val=Ulow, name='Uнн', desc='напряжение обмотки НН', unit='кВ', n_digits=0)
        self.Ihigh = _Var(val=self.S / (3 ** 0.5 * self.Uhigh), name='Iном.вн', desc='номинальный ток стороны ВН',
                          unit='А', n_digits=0)
        self.Ilow = _Var(val=self.S / (3 ** 0.5 * self.Ulow), name='Iном.нн', desc='номинальный ток стороны НН',
                         unit='А', n_digits=0)
        self.Usc = _Var(val=Usc, name='Uкз', desc='напряжение короткого замыкания', unit='о.е.', n_digits=3)
        self.Inl = _Var(val=Inl, name='Iхх', desc='ток холостого хода', unit='о.е.', n_digits=4)
        self.Psc = _Var(val=Psc, name='Pкз', desc='активные потери в опыте короткого замыкания', unit='кВт', n_digits=1)
        self.Pnl = _Var(val=Pnl, name='Pхх', desc='активные потери в опыте холостого хода', unit='кВт', n_digits=1)
        self.scheme_high = _StrVar(val=scheme_high, name='Схема ВН', desc='схема соединения обмотки ВН')
        self.scheme_low = _StrVar(val=scheme_low, name='Схема НН', desc='схема соединения обмотки НН')
        self.group = _StrVar(val=group, name='Группа', desc='группа соединения трансформатора')
        self.__calc_impedances()

    def __calc_impedances(self):
        self.Zsc = _Var(val=self.Usc * self.Uhigh ** 2 / self.S, name='Zкз',
                        desc='сопротивление трансформатора в режиме КЗ', unit='Ом', n_digits=2)
        self.Rsc = _Var(val=self.Psc / 3 / self.Ihigh ** 2, name='Rкз',
                        desc='активное сопротивление трансформатора в режиме КЗ', unit='Ом', n_digits=2)
        self.Xsc = _Var(val=(self.Zsc ** 2 - self.Rsc ** 2) ** 0.5, name='Xкз',
                        desc='реактивное сопротивление трансформатора в режиме КЗ', unit='Ом', n_digits=2)
        self.Znl = _Var(val=self.Uhigh ** 2 / self.S / self.Inl, name='Zхх',
                        desc='сопротивление трансформатора в режиме ХХ', unit='Ом', n_digits=0)
        self.Rnl = _Var(val=self.Uhigh ** 2 / self.Pnl, name='Rхх',
                        desc='активное сопротивление трансформатора в режиме ХХ', unit='Ом', n_digits=0)

        def get_x(z, r):
            """Расчет реактивного сопротивления в параллельной схеме"""
            a = r ** 2 - z ** 2
            b = r ** 4 - 2 * r ** 2 * z ** 2
            c = - r ** 4 * z ** 2
            D = b ** 2 - 4 * a * c
            return ((- b + D ** 0.5) / (2 * a)) ** 0.5

        self.Xnl = _Var(val=get_x(self.Znl.val, self.Rnl.val), name='Xхх',
                        desc='реактивное сопротивление трансформатора в режиме ХХ', unit='Ом', n_digits=0)


class System(Equipment):
    """Модель системы:
    Unom - номинальное напряжение системы, кВ;
    Isc3 - ток трехфазного КЗ на шинах системы, А;
    Isc2 - ток двухфазного КЗ на шинах системы, А;
    tau - постоянная времени сети, c;
    psi - начальный угол, гр;
    Расчетные величины:
    R, X, Z - сопротивление системы, Ом;
    """

    def __init__(self, Unom, Isc3, tau=0.03, psi=0):
        self.Unom = _Var(val=Unom, name='Uном', desc='номинальное напряжение системы', unit='кВ', n_digits=0)
        self.Isc3 = _Var(val=Isc3, name='Iкз(3ф)', desc='ток трехфазного КЗ на шинах системы', unit='А', n_digits=0)
        self.Isc2 = _Var(val=Isc3 * 3 ** 0.5 / 2, name='Iкз(2ф)', desc='ток двухфазного КЗ на шинах системы', unit='А',
                         n_digits=0)
        self.tau = _Var(val=tau, name='tau', desc='постоянная времени сети', unit='с', n_digits=2)
        self.psi = _Var(val=tau, name='psi', desc='начальный угол', unit='гр', n_digits=2)
        self.__calc_impedances()

    def __calc_impedances(self):
        self.Z = _Var(val=self.Unom / 3 ** 0.5 / self.Isc3, name='Z', desc='сопротивление системы', unit='Ом',
                      n_digits=2)
        self.R = _Var(val=self.Z / (1 + (2 * np.pi * 50 * self.tau.val) ** 2) ** 0.5, name='R',
                      desc='активное сопротивление системы', unit='Ом', n_digits=2)
        self.X = _Var(val=(self.Z ** 2 - self.R ** 2) ** 0.5, name='X', desc='реактивное сопротивление системы',
                      unit='Ом', n_digits=2)


class Line(Equipment):
    """Модель линии:
    L - длина линии, км
    x1 - удельное индуктивное сопротивление прямой последовательности, Ом/км;
    r1 - удельное активное сопротивление прямой последовательности, Ом/км;
    x0 - удельное индуктивное сопротивление нулевой последовательности, Ом/км;
    r0 - удельное активное сопротивление нулевой последовательности, Ом/км
    """

    def __init__(self, L=70, x1=0.41, r1=0.1, x0=1.2, r0=0.4):
        self.L = _Var(val=L, name='L', desc='длина линии', unit='км', n_digits=2)
        self.x1 = _Var(val=x1, name='x1 уд', desc='удельное индуктивное сопротивление прямой последовательности',
                       unit='Ом/км', n_digits=3)
        self.r1 = _Var(val=r1, name='r1 уд', desc='удельное активное сопротивление прямой последовательности',
                       unit='Ом/км', n_digits=3)
        self.x0 = _Var(val=x0, name='x0 уд', desc='удельное индуктивное сопротивление нулевой последовательности',
                       unit='Ом/км', n_digits=3)
        self.r0 = _Var(val=r0, name='r0 уд', desc='удельное активное сопротивление нулевой последовательности',
                       unit='Ом/км', n_digits=3)
        self.X1 = _Var(val=x1 * L, name='X1', desc='активное сопротивление прямой последовательности',
                            unit='Ом', n_digits=3)
        self.R1 = _Var(val=r1 * L, name='R1', desc='индуктивное сопротивление прямой последовательности',
                            unit='Ом', n_digits=3)
        self.X0 = _Var(val=x0 * L, name='X0', desc='активное сопротивление нулевой последовательности',
                            unit='Ом', n_digits=3)
        self.R0 = _Var(val=r0 * L, name='R0', desc='индуктивное сопротивление нулевой последовательности',
                            unit='Ом', n_digits=3)


class ShortCircuit(Equipment):
    """Модель дуги в точке короткого замыкания:
    R - активное сопротивление дуги, Ом
    """

    def __init__(self, R=1):
        self.R = _Var(val=R, name='Rдуги', desc='активное сопротивление дуги', unit='Ом', n_digits=3)


class Load(Equipment):
    """Модель нагрузки, последовательное соединение RLC
    Snom - номинальное напряжение нагрузки, кВА;
    Unom - номинальное напряжение нагрузки, кВ;
    cos_nom - номинальная коэффициент мощности;
    Расчетные величины:
    R, X, Z - сопротивление системы, Ом;
    """

    def __init__(self, Snom, Unom, cos_nom, type='L'):
        self.Unom = _Var(val=Unom, name='Uном', desc='номинальное напряжение нагрузки', unit='кВ', n_digits=0)
        self.Snom = _Var(val=Snom, name='Sнагр', desc='мощность нагрузки', unit='кВА', n_digits=0)
        self.cos_nom = _Var(val=cos_nom, name='cos ном', desc='номинальный коэффициент мощности', unit='', n_digits=2)
        self.__type = type
        self.__calc_impedances()

    def __calc_impedances(self):
        self.Z = _Var(val=self.Unom ** 2 / self.Snom, name='Zнагр', desc='сопротивление нагрузки', unit='Ом',
                      n_digits=2)
        self.R = _Var(val=self.Z * self.cos_nom, name='Rнагр', desc='активное сопротивление нагрузки', unit='Ом',
                      n_digits=2)
        if self.__type == 'L':
            X = self.Z * np.sqrt(1 - self.cos_nom ** 2)
            L, C = X / const.w, 0
        elif self.__type == 'C':
            X = (-1) * self.Z * np.sqrt(1 - self.cos_nom ** 2)
            L, C = 0, -1 / const.w / X
        else:
            raise TypeError(f'Неверный тип нагрузки: {self.__type}')
        self.X = _Var(val=X, name='Xнагр', desc='реактивное сопротивление нагрузки', unit='Ом', n_digits=2)
        self.L = _Var(val=L, name='Lнагр', desc='индуктивность нагрузки', unit='Гн', n_digits=4)
        self.C = _Var(val=C / 1e-6, name='Cнагр', desc='емкость нагрузки', unit='мкФ', n_digits=2)


class CT(Equipment):
    """Модель трансформатора тока:
    I1nom - номинальный первичный ток, А;
    I2nom- номинальный вторичный ток, А;
    Knom - номинальная допустимая предельная кратность;
    Snom - номинальная вторичная мощность, ВА;
    cos_nom - номинальная коэффициент мощности;
    r2 - активное сопротивление вторичной обмотки, Ом;
    x2 - реактивное сопротивление вторичной обмотки, Ом;
    Sload - фактическая вторичная мощность, ВА;
    cos_load - фактический коэффициент мощности;
    Расчетные величины:
    n - коэффициент трансформации ТТ;
    rnom - номинальное активное сопротивление вторичной нагрузки;
    xnom - номинальное реактивное сопротивление вторичной нагрузки;
    rload - фактическое активное сопротивление вторичной нагрузки;
    xload - фактическое реактивное сопротивление вторичной нагрузки.
    """

    def __init__(self, I1nom, I2nom, Knom=20, Snom=50, znom =None, cos_nom=0.8, r2=10., x2=0., Sload=6.3, zload=None, cos_load=1.0):
        self.I1nom = _Var(val=I1nom, name='I1ном', desc='номинальный первичный ток', unit='А', n_digits=0)
        self.I2nom = _Var(val=I2nom, name='I2ном', desc='номинальный вторичный ток', unit='А', n_digits=0)
        self.Knom = _Var(val=Knom, name='Кном', desc='номинальная допустимая предельная кратность', unit='', n_digits=0)
        self.I1faultnom = _Var(val=self.I1nom * self.Knom, name='I1ном кз',
                               desc='номинальный предельный допустимый ток КЗ', unit='А', n_digits=0)
        self.cos_nom = _Var(val=cos_nom, name='cos ном', desc='номинальный коэффициент мощности', unit='', n_digits=2)
        self.r2 = _Var(val=r2, name='R2', desc='активное сопротивление вторичной обмотки', unit='Ом', n_digits=3)
        self.x2 = _Var(val=x2, name='X2', desc='реактивное сопротивление вторичной обмотки', unit='Ом', n_digits=3)
        self.n = _Var(val=self.I1nom / self.I2nom, name='n', desc='коэффициент трансформации', unit='', n_digits=0)
        if znom is None:
            self.Snom = _Var(val=Snom, name='Sном', desc='номинальная вторичная мощность', unit='ВА', n_digits=0)
            Rnom, Xnom = self.Snom.val * self.cos_nom.val / self.I2nom.val ** 2, self.Snom.val * np.sqrt(
                1 - self.cos_nom.val ** 2) / self.I2nom.val ** 2
            self.rnom = _Var(val=Rnom, name='Rном', desc='номинальное активное сопротивление вторичной нагрузки',
                             unit='Ом', n_digits=3)
            self.xnom = _Var(val=Xnom, name='Xном', desc='номинальное реактивное сопротивление вторичной нагрузки',
                             unit='Ом', n_digits=3)
            self.znom = _Var(val=np.sqrt(Xnom ** 2 + Rnom ** 2), name='Zном',
                             desc='номинальное полное сопротивление вторичной нагрузки', unit='Ом', n_digits=3)
        else:
            self.znom = _Var(val=znom, name='Zном',
                             desc='номинальное полное сопротивление вторичной нагрузки', unit='Ом', n_digits=3)
            self.rnom = _Var(val=znom * self.cos_nom.val, name='Rном', desc='номинальное активное сопротивление вторичной нагрузки',
                             unit='Ом', n_digits=3)
            self.xnom = _Var(val=znom * np.sqrt(1 - self.cos_nom.val**2), name='Xном', desc='номинальное реактивное сопротивление вторичной нагрузки',
                             unit='Ом', n_digits=3)
            self.Snom = _Var(val=znom * self.I2nom.val ** 2, name='Sном', desc='номинальная вторичная мощность', unit='ВА', n_digits=0)
        if zload is None:
            self.Sload = _Var(val=Sload, name='Sнагр', desc='фактическая вторичная мощность', unit='ВА', n_digits=2)
            self.cos_load = _Var(val=cos_load, name='cos нагр', desc='фактический коэффициент мощности', unit='',
                             n_digits=2)
            Rload, Xload = self.Sload.val * self.cos_load.val / self.I2nom.val ** 2, self.Sload.val * np.sqrt(
                1 - self.cos_load.val ** 2) / self.I2nom.val ** 2
            self.rload = _Var(val=Rload, name='Rфакт', desc='фактическое активное сопротивление вторичной нагрузки',
                              unit='Ом', n_digits=3)
            self.xload = _Var(val=Xload, name='Xфакт', desc='фактическое реактивное сопротивление вторичной нагрузки',
                              unit='Ом', n_digits=3)
            self.zload = _Var(val=np.sqrt(Xload**2 + Rload**2), name='Zфакт', desc='фактическое полное сопротивление вторичной нагрузки',
                             unit='Ом', n_digits=3)
        else:
            self.zload = _Var(val=zload, name='Zфакт', desc='фактическое полное сопротивление вторичной нагрузки',
                             unit='Ом', n_digits=3)
            self.rload = _Var(val=zload * self.cos_load.val, name='Rфакт', desc='фактическое активное сопротивление вторичной нагрузки',
                              unit='Ом', n_digits=3)
            self.xload = _Var(val=zload* np.sqrt(1 - self.cos_load.val**2), name='Xфакт', desc='фактическое реактивное сопротивление вторичной нагрузки',
                              unit='Ом', n_digits=3)
            self.Sload = _Var(val=zload * self.I2nom.val ** 2, name='Sнагр', desc='фактическая вторичная мощность', unit='ВА', n_digits=2)

    def calc_saturation(self, Isc, tau, Kr=0.86):
        """Расчет времени до насыщения ТТ

        :param Isc: ток короткого замыкания, type: float
        :param tau: постоянная времени,с , type: float
        :param Kr: остаточная намагниченность
        :return:pnt, plt, type: pandas.DataFrame
                pnt - таблица kr, A, t_sat, fi_sat - время до насыщения с/без остсточной намагниченности
                plt - таблица kpr, t - зависимость магнитного потока от времени без учета насыщения
        """
        T_MAX_CONST = 1
        Rnom, Xnom = self.rnom.val, self.xnom.val
        Rload, Xload = self.rload.val, self.xload.val
        R2, X2 = self.r2.val, self.x2.val
        # Параметр режима
        A = self.I1nom.val * self.Knom.val * ((R2 + Rnom) ** 2 + (X2 + Xnom) ** 2) ** 0.5 / (
                    Isc * ((R2 + Rload) ** 2 + (X2 + Xload) ** 2) ** 0.5)
        # угол нагрузки
        cos_alfa = (R2 + Rload) / ((R2 + Rload) ** 2 + (X2 + Xload) ** 2) ** 0.5
        alfa = np.arccos(cos_alfa)
        w = 2 * np.pi * 50  # угловая частота
        # Проверим произойдет ли насыщение без остаточной намагниченности
        is_saturated = True if w * tau + 1 > A else False
        # Проверим произойдет ли насыщение c наличием остаточной намагниченности
        A_Kr = A * (1 - Kr)
        is_saturated_Kr = True if w * tau + 1 > A_Kr else False
        # Точный расчет
        t = np.arange(0, T_MAX_CONST, 0.0001)
        # Расчет угла fi при котором достигается максимальное насыщение
        num = np.cos(alfa) - np.cos(w * t + alfa)
        denom = np.sin(alfa) * np.exp(-t / tau) + np.cos(alfa) * w * tau * (1 - np.exp(-t / tau)) - np.sin(
            w * t + alfa) + 0.000001
        fi = np.arctan(num / denom)
        # Построим характеристику Kпр(t)
        kpr = np.sin(alfa) * np.cos(fi) * np.exp(-t / tau) + \
              np.cos(alfa) * np.cos(fi) * w * tau * (1 - np.exp(-t / tau)) - \
              np.sin(w * t + alfa + fi) + \
              np.cos(alfa) * np.sin(fi)
        # Найдем время до насыщения
        tnas = t[np.argmin(np.abs(kpr - A))] if is_saturated else T_MAX_CONST
        tnas_Kr = t[np.argmin(np.abs(kpr - A_Kr))] if is_saturated_Kr else T_MAX_CONST
        fi_nas = np.rad2deg(fi[np.argmin(np.abs(kpr - A))]) - 90 if is_saturated else None
        fi_nas_Kr = np.rad2deg(fi[np.argmin(np.abs(kpr - A_Kr))]) - 90 if is_saturated_Kr else None
        return _pd.DataFrame(dict(kr=[0, Kr], A=[A, A_Kr], t_sat=[tnas, tnas_Kr], fi_sat=[fi_nas, fi_nas_Kr])), \
               _pd.DataFrame({'kpr': kpr, 't': t})


if __name__ == '__main__':
    tr = Transformer(16000, 110, 11, Usc=0.105, Inl=0.007, Psc=85, Pnl=19, scheme_high='Y', scheme_low='D', group=11)
    system = System(110, 10000, 0.03)
    ct = CT(100, 5, Knom=10, Snom=15, cos_nom=0.8, r2=0.12, x2=0, Sload=3, cos_load=0.8)
    pt, plt = ct.calc_saturation(Isc=500, tau=0.2, Kr=0.86)
    print(plt)
    # load = Load(Snom=10000, Unom=10, cos_nom=0.8, type='C')
    # print(load)
