"""
relaylab.characteristics
================

Модуль содержит характеристики срабатывания защит.
"""
import numpy as np
from relaylab.signals import AnalogSignal as _AnalogSignal, ComplexSignal as _ComplexSignal, \
    DiscreteSignal as _DiscreteSignal
from relaylab.signals import const, color
from relaylab.signals import _resample_discrete
import plotly.graph_objects as go


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
        self._check_type(Ibiases, _AnalogSignal)
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


class DistanceRelay(_Relay):
    """Четырехугольная характеристика срабатывания дистанционной защиты с вырезом нагрузки"""

    def __init__(self, z_line=10, fi_line=70, r_right=5, fi_right=70, offset=0.1, r_load=None, fi_load=30,
                 backward:bool=False, directed:bool=False):
        """ Задание характеристики срабатывания

        :param z_line: Полное сопротивление срабатывания, type:float
        :param fi_line: Угол линии, type:float
        :param r_right: Уставка по активному сопротивлению, type:float
        :param fi_right: Угол наклона правой стороны характеристики срабатывания, type:float
        :param offset: Коэффициент смещения за спину характеристики срабатывания, type:float
        :param r_load: Угол нагрузки, гр, type:float
        :param fi_load: Уставка по активному сопротивлению зоны нагрузки, type:float
        :param backward: Направленность характеристики в обратную сторону, type:bool
        :param directed: Флаг направленности пускового органа, type:bool
        """
        super().__init__()
        self.z_line = z_line
        self.fi_line = fi_line
        self.r_right = r_right
        self.fi_right = fi_right
        self.offset = offset
        self.r_load = r_load
        self.fi_load = fi_load
        self.backward = backward
        self.directed = directed

    @staticmethod
    def __solve_eq(eq1, eq2):
        """Вычисление точек пересечения двух линий (x, y), eq = (k, b), y=k*x+b"""
        a = np.array([[eq1[0], -1], [eq2[0], -1]])
        b = np.array([-eq1[1], -eq2[1]])
        return np.linalg.solve(a, b)

    def get_points(self):
        """Возвращает координаты точек характеристики срабатывания на плоскости: R, X"""
        #TODO: реализовать направленность при расчете точек для отрисовки
        fi_line = np.deg2rad(self.fi_line)
        fi_right = np.deg2rad(self.fi_right)
        fi_load = np.deg2rad(self.fi_load)
        # Уравнения линий
        up = 0,  self.z_line * np.sin(fi_line) #верхняя сторона
        right = np.tan(fi_right), -self.r_right * np.tan(fi_right) #правая сторона
        down = 0,  - self.offset * self.z_line * np.sin(fi_line) #нижняя сторона
        left = np.tan(fi_line), self.r_right * np.tan(fi_line) #левая сторона
        # Уравнения лучей направленной характеристики
        up_ray = np.tan(np.deg2rad(105)), 0
        right_ray = np.tan(np.deg2rad(-15)), 0
        # Вычисляем точки пересечения
        t1 = self.__solve_eq(up, right)
        t2 = self.__solve_eq(down, right)
        t3 = self.__solve_eq(down, left)
        t4 = self.__solve_eq(up, left)
        t0 = np.array([0, 0])
        if self.r_load is None:
            if self.directed:
                t2a = self.__solve_eq(right_ray, right)
                t2b = self.__solve_eq(right_ray, down)
                t4a = self.__solve_eq(up_ray, left)
                t4b = self.__solve_eq(up_ray, up)
                points_array = [t1]
                if t2a[1] > t2[1]:
                    points_array.append(t2a)
                else:
                    points_array.append(t2)
                    points_array.append(t2b)
                points_array.append(t0)
                if t4a[1] < t4[1]:
                    points_array.append(t4a)
                    points_array.append(t4)
                else:
                    points_array.append(t4b)
                points_array.append(t1)
                points = np.array(points_array)
            else:
                points = np.array([t1, t2, t3, t4, t1])
        else:
            points_lst = []
            # Уравнения линий нагрузки
            load_pos = np.tan(fi_load), 0  # нагрузка положительный наклон
            load_neg = np.tan(-fi_load), 0  # нагрузка отрицательный наклон
            r_pos = 100000, -100000 * self.r_load  # нагрузка справа
            r_neg = 100000, 100000 * self.r_load  # нагрузка слева
            t1a = self.__solve_eq(up, r_pos)
            t1b = self.__solve_eq(up, load_pos)
            t1c = self.__solve_eq(right, load_pos)
            t1d = self.__solve_eq(r_pos, load_pos)
            t1e = self.__solve_eq(right, r_pos)
            t2a = self.__solve_eq(down, r_pos)
            t2b = self.__solve_eq(down, load_neg)
            t2c = self.__solve_eq(right, load_neg)
            t2d = self.__solve_eq(r_pos, load_neg)
            t2e = self.__solve_eq(right, r_pos)
            t3a = self.__solve_eq(down, r_neg)
            t3b = self.__solve_eq(down, load_pos)
            t3c = self.__solve_eq(left, load_pos)
            t3d = self.__solve_eq(r_neg, load_pos)
            t3e = self.__solve_eq(left, r_neg)
            t4a = self.__solve_eq(up, r_neg)
            t4b = self.__solve_eq(up, load_neg)
            t4c = self.__solve_eq(left, load_neg)
            t4d = self.__solve_eq(r_neg, load_neg)
            t4e = self.__solve_eq(left, r_neg)
            # Обработка 1 четверти
            if t1a[1] < t1d[1] and t1a[0] < t1[0]:
                # область нагрузки больше характеристики сверху и снизу
                points_lst.append(t1a)
                #print(f'Вар.1: t1a[1] = {t1a[1]},  t1d[1] = {t1d[1]}')
            else:
                #print(f'Вар.2: t1a[1] = {t1a[1]},  t1d[1] = {t1d[1]}')
                if t1e[1] > t1c[1]:
                    # область нагрузки не пересекается с характеристикой
                    points_lst.append(t1)
                    #print(f'Вар.1: t1e[1] = {t1e[1]},  t1c[1] = {t1c[1]}')
                else:
                    #print(f'Вар.2: t1e[1] = {t1e[1]},  t1c[1] = {t1c[1]}')
                    if t1b[0] < t1[0]:
                        # область нагрузки съела сторону 2
                        points_lst.append(t1b)
                        if t1d[1] < t1[1]:
                            points_lst.append(t1d)
                    else:
                        points_lst.append(t1)
                        if t1c[1] < t1[1]:
                            points_lst.append(t1c)
                            points_lst.append(t1d)
                    if (t2c[0] < t1e[0] < t1c[0]) and t1e[1] > t2[1]:
                        points_lst.append(t1e)
            # Обработка 2 четверти
            if t2a[1] > t2d[1] and t2a[0] < t2[0]:
                if t2a[0] < t2[0]:
                    points_lst.append(t2a)
                else:
                    points_lst.append(t2)
            else:
                if t2e[1] > t2c[1]:
                    # область нагрузки не пересекается с характеристикой
                    points_lst.append(t2)
                else:
                    if t2b[0] < t2[0]:
                        if t2d[1] > t2[1]:
                            points_lst.append(t2d)
                        points_lst.append(t2b)
                    else:
                        if t2c[1] > t2[1]:
                            points_lst.append(t2d)
                            points_lst.append(t2c)
                        points_lst.append(t2)
            # Обработка 3 четверти
            if t3a[1] > t3d[1] and t3a[0] > t3[0]:
                # область нагрузки больше характеристики сверху и снизу
                points_lst.append(t3a)
            else:
                if t3e[1] < t3c[1]:
                    # область нагрузки не пересекается с характеристикой
                    points_lst.append(t3)
                else:
                    if t3b[0] > t3[0]:
                        # область нагрузки съела сторону 2
                        points_lst.append(t3b)
                        if t3d[1] > t3[1]:
                            points_lst.append(t3d)
                    else:
                        points_lst.append(t3)
                        if t3c[1] > t3[1]:
                            points_lst.append(t3c)
                            points_lst.append(t3d)
                    if (t4c[0] > t3e[0] > t3c[0]) and t3e[1] < t4[1]:
                        points_lst.append(t3e)
            # Обработка 4 четверти
            if t4a[1] < t4d[1] and t4a[0] > t4[0]:
                if t4a[0] > t4[0]:
                    points_lst.append(t4a)
                else:
                    points_lst.append(t4)
            else:
                if t4e[1] < t4c[1]:
                    # область нагрузки не пересекается с характеристикой
                    points_lst.append(t4)
                else:
                    if t4b[0] > t4[0]:
                        if t4d[1] < t4[1]:
                            points_lst.append(t4d)
                        points_lst.append(t4b)
                    else:
                        if t4c[1] < t4[1]:
                            points_lst.append(t4d)
                            points_lst.append(t4c)
                        points_lst.append(t4)
            points_lst.append(points_lst[0])
            points = np.array(points_lst)
        if self.backward:
            points = - points
        return points[:, 0], points[:, 1]

    def start(self, *signals: _ComplexSignal):
        """Пусковой орган сопротивления без контура памяти. Срабатывание происходит при попадании в характеристику срабатывания.
        Учитывает направленность, обратное направление, вырез нагрузки

        :param signals: входные сигналы, type: AnalogSignal
        :return: логический сигнал, type: DiscreteSignal
        """
        self._check_type(signals, _ComplexSignal)
        fi_line = np.deg2rad(self.fi_line)
        fi_right = np.deg2rad(self.fi_right)
        fi_load = np.deg2rad(self.fi_load)
        # Уравнения линий
        up = 0,  self.z_line * np.sin(fi_line) #верхняя сторона
        right = np.tan(fi_right), -self.r_right * np.tan(fi_right) #правая сторона
        down = 0,  - self.offset * self.z_line * np.sin(fi_line) #нижняя сторона
        left = np.tan(fi_line), self.r_right * np.tan(fi_line) #левая сторона
        # Уравнения лучей направленной характеристики
        up_ray = np.tan(np.deg2rad(105)), 0
        right_ray = np.tan(np.deg2rad(-15)), 0
        # Уравнения линий нагрузки
        if self.r_load is not None:
            load_pos = np.tan(fi_load), 0  # нагрузка положительный наклон
            load_neg = np.tan(-fi_load), 0  # нагрузка отрицательный наклон
            r_pos = 100000, -100000 * self.r_load  # нагрузка справа
            r_neg = 100000, 100000 * self.r_load  # нагрузка слева
        res_arr = []
        for signal in signals:
            signal_val  = signal.val
            if self.backward:
                signal_val = - signal_val
            xp = np.real(signal_val)
            yp = np.imag(signal_val)
            start = ((((up[0] * xp + up[1]) > yp) &
                      (yp > (down[0] * xp + down[1]))) &
                     (((yp - right[1]) / right[0]) > xp) &
                     (xp > ((yp - left[1]) / left[0])))
            if self.directed:
                start = start & ((((yp - up_ray[1]) / up_ray[0]) < xp) &
                                 ((right_ray[0] * xp + right_ray[1]) < yp))
            if self.r_load is not None:
                start = (start &
                         np.logical_not(((load_pos[0] * xp + load_pos[1]) > yp) &
                                         (yp > (load_neg[0] * xp + load_neg[1])) &
                                        (((yp - r_pos[1]) / r_pos[0]) < xp)) &
                         np.logical_not(((load_pos[0] * xp + load_pos[1]) < yp) &
                                         (yp < (load_neg[0] * xp + load_neg[1])) &
                                        (((yp - r_neg[1]) / r_neg[0]) > xp)))
            start_resampled = _resample_discrete(start)
            res_arr.append(_DiscreteSignal(name=f'{signal.name}<', val=start_resampled, Fs=signal.Fs))
        return res_arr[0] if len(res_arr) == 1 else tuple(res_arr)


if __name__ == '__main__':
    relay = DistanceRelay(z_line=10, fi_line=70, r_right=10, fi_right=70, offset=0.2, r_load=5, fi_load=35, backward=False)
    relay2 = DistanceRelay(z_line=10, fi_line=70, r_right=10, fi_right=70, offset=0.2, r_load=5, fi_load=35, directed=False, backward=False)
    r, x = relay.get_points()
    r2, x2 = relay2.get_points()
    ang = np.arange(0, 360, 10)
    z = np.arange(1, 20, 1)
    ang_z_ar = np.array(np.meshgrid(z, ang)).T.reshape(-1, 2)
    z = ang_z_ar[:,0] * np.exp(1j*np.deg2rad(ang_z_ar[:, 1]))
    trip = relay2.start(_ComplexSignal(val=z))
    trip_color = [color.green if t else color.grey for t in trip.val]
    print(trip)
    fig = go.Figure()
    fig.add_scatter(x=r, y=x, name='Первая ступень ДЗ', mode='lines', fill="toself")
    fig.add_scatter(x=r2, y=x2, name='Первая ступень ДЗ направл.', mode='lines', fill="toself")
    fig.add_scatter(x=np.real(z), y=np.imag(z), name='точки', mode='markers', marker=dict(color=trip_color))
    fig.update_xaxes(range=(-40, 60))
    fig.update_yaxes(range=(-20, 40))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, )
    fig.update_layout(height=800, width=1200, template='plotly_white', margin=dict(b=10, t=10, l=10, r=10))
    fig.show()