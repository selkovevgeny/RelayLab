"""
relaylab.record
================

Модуль содержит инструменты импорта осциллограмм в формате COMTRADE.
В состав модуля подключены примеры осциллограмм: relaylab.record.examples
"""
import numpy as np
import pandas as _pd
from string import ascii_letters as _ascii_letters, digits as _digits

import pandas as pd

from relaylab.comtrade import Comtrade as _Comtrade
from relaylab.signals import AnalogSignal as _AnalogSignal, DiscreteSignal as _DiscreteSignal, \
    ComplexSignal as _ComplexSignal
from relaylab import signals as _sg
from pathlib import Path as _Path
import relaylab as _lb
from datetime import datetime as _datetime
from typing import Union

_trans_dict = {'Ь': '', 'ь': '', 'Ъ': '', 'ъ': '', 'А': 'A', 'а': 'a', 'Б': 'B', 'б': 'b', 'В': 'V', 'в': 'v',
               'Г': 'G', 'г': 'g', 'Д': 'D', 'д': 'd', 'Е': 'E', 'е': 'e', 'Ё': 'E', 'ё': 'e', 'Ж': 'Zh', 'ж': 'zh',
               'З': 'Z', 'з': 'z', 'И': 'I', 'и': 'i', 'Й': 'I', 'й': 'i', 'К': 'K', 'к': 'k', 'Л': 'L', 'л': 'l',
               'М': 'M', 'м': 'm', 'Н': 'N', 'н': 'n', 'О': 'O', 'о': 'o', 'П': 'P', 'п': 'p', 'Р': 'R', 'р': 'r',
               'С': 'S', 'с': 's', 'Т': 'T', 'т': 't', 'У': 'U', 'у': 'u', 'Ф': 'F', 'ф': 'f', 'Х': 'Kh', 'х': 'kh',
               'Ц': 'Tc', 'ц': 'tc', 'Ч': 'Ch', 'ч': 'ch', 'Ш': 'Sh', 'ш': 'sh', 'Щ': 'Shch', 'щ': 'shch', 'Ы': 'Y',
               'ы': 'y', 'Э': 'E', 'э': 'e', 'Ю': 'Iu', 'ю': 'iu', 'Я': 'Ya', 'я': 'ya', ".": "", ",": "", "!": "",
               "?": "", "-": "_", ":": "_", ";": "", "...": "", ">": "b", "<": "m",
               '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', '0': '0'}

_trans_dict = _trans_dict | dict(zip(list(_ascii_letters), list(_ascii_letters)))


def _translate(word):
    return ''.join(_trans_dict[letter] if letter in _trans_dict else '_' for letter in word)


def _define_encoding(bytes_string: bytes):
    """Define encoding of the string"""
    # print("bytes_string = ", bytes_string)
    # print("cp1251 = ", bytes_string.decode("cp1251"))
    # print("cp866 = ", bytes_string.decode("cp866"))
    is_cp1251 = True
    is_cp866 = True
    for int_val in bytes_string:
        byte_val = int_val.to_bytes(1, byteorder='big')
        if not (int_val >= int("0xc0", 16) and int_val <= int("0xff", 16)) and not (byte_val.isascii()):
            is_cp1251 = False
        if not (int_val >= int("0x80", 16) and int_val <= int("0xaf", 16)) and \
                not (int_val >= int("0xe0", 16) and int_val <= int("0xf1", 16)) and \
                not (byte_val.isascii()) and not (int_val == int("0xfc", 16)):
            is_cp866 = False
    if is_cp1251:
        return "cp1251"
    elif is_cp866:
        return "cp866"
    else:
        return "undefined"


class Record:
    """Работа с осциллограммами в формате COMTRADE.

    При инициализации происходит загрузка осциллограммы.
    После загрузки все аналоговые и дискретные сигналы осциллограммы становятся атрибутами данного класса.
    То есть доступ к сигнала становится доступен слеудющим образом:

    >>> from relaylab import record
    >>> rec = record.Record('C:\Работа\База осциллограмм\Двигатель\Пуски двигателей\Насыщение ТТ при пуске\\1.cfg')
    >>> rec.DTO_srab
    <class 'relaylab.signals.DiscreteSignal'>: ДТО сраб.

    Для русскоязычных символов происходит транслитерация.

    Доступ к описанию осциллограммы проиходит следующим образом:
    >>>     print(rec)
    Наименование осциллограммы: C:\python\RelayLab\relaylab\records\motor_start.cfg
    Ревизия стандарта: 2000
    Объект установки устройства: ЛПДС Лысьва/ МНА 8000 кВт
    Наименование устройства: БМРЗ-УЗД_00
    Начало записи: 2015-10-29 06:01:06.571000
    Длительность записи: 1.10, c
    Количество аналоговых каналов: 11
    Количество дискретных каналов: 61
    """

    def __init__(self, *args):
        """Инициализация экземпляра класса. Загрузка осциллограммы в формате COMTRADE или
        создание на основе аналоговых или дискретных сигналов

        :param args: путь к файлу осциллограммы или список каналов
        """
        if len(args) == 1 and type(args[0]) == str:
            self.__comtrade = self.__open_comtrade(args[0])
            self.Fs = self.__get_Fs()
            self.nmax = self.__comtrade.total_samples
            self.tmax = self.nmax / self.Fs
            self.__parse_channels()
        elif len(args) == 0:
            self.Fs, self.nmax, self.tmax = None, 0, 0
            self.__comtrade = None
        else:
            self.Fs, self.nmax, self.tmax = None, 0, 0
            self.__comtrade = None
            self.append(*args)

    @staticmethod
    def __open_comtrade(comtrade_file):
        if comtrade_file:
            with open(comtrade_file, "rb") as cfg:
                cfg_encoding = _define_encoding(cfg.read())
            if cfg_encoding != "undefined":
                """Downloading comtrade record and create analog and digital channels"""
                record = _Comtrade(use_numpy_arrays=True)
                record.load(comtrade_file, encoding=cfg_encoding)
                return record
            else:
                raise ImportError(f"Не определена кодировка файла: {comtrade_file}")

    def __get_Fs(self):
        freqs_tpl = tuple(
            {200, 300, 400, 600, 800, 1200, 1600, 2400, 3200, 4800, 6400, 9600, 19200, 200, 400, 500, 800, 1000,
             1600, 2000, 2500, 3200, 4000, 5000, 6400, 8000, 10000, 16000, 20000, 32000, 40000, 80000, 160000})
        freqs = np.array(sorted(freqs_tpl))
        sample_rates = self.__comtrade.cfg.sample_rates
        if len(sample_rates) == 1:
            if sample_rates[0][0] != 0:
                Fs = sample_rates[0][0]
            else:
                Fs = int(self.__comtrade.time.shape[0] / max(self.__comtrade.time))  # рассчитаем из dat файла
                Fs = self.__find_nearest(freqs, Fs)
        else:
            n_start = 0
            rate_weighted = []
            for rate, n in sample_rates:
                rate_weighted.append(rate * (n - n_start))
                n_start = n
            Fs = sum(rate_weighted) / n_start  # рассчитаем и усредним из cfg файла
            Fs = self.__find_nearest(freqs, Fs)
        return Fs

    @staticmethod
    def __find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def __parse_channels(self):
        for name, data in zip(self.__comtrade.analog_channel_ids, self.__comtrade.analog):
            signal = _AnalogSignal(name=name, Fs=self.Fs)
            signal.val = data
            self.__set_channel(signal)
        for name, data in zip(self.__comtrade.status_channel_ids, self.__comtrade.status):
            signal = _DiscreteSignal(name=name, Fs=self.Fs)
            signal.val = np.array(data, dtype=np.bool_)
            self.__set_channel(signal)

    def append(self, *args):
        """Функция добавлят в экземпляр класса входные сигналы

        :param args: перечень сигналов, type: AnalogSignal, DiscreteSignal
        :return: None
        """
        if all(map(lambda arg: type(arg) in (_AnalogSignal, _DiscreteSignal, _ComplexSignal), args)):
            if len(args) >= 1 and self.Fs is None:
                self.Fs = args[0].Fs
                self.nmax = len(args[0])
                self.tmax = self.nmax / self.Fs
            for ch in args:
                if ch.Fs == self.Fs and len(ch.val) == self.nmax:
                    ch = ch.abs() if type(ch) == _ComplexSignal else ch
                    self.__set_channel(ch, raise_error=True)
                else:
                    raise ValueError(f'Неверные входные каналы. У присоединяемых каналов частота дискретизации '
                                     f'Fs должна быть {self.Fs}, '
                                     f'длительность записи должна быть {self.tmax} ({self.nmax} точек)')
        else:
            raise ValueError('Неверные входные аргументы. Входные аргументы должны быть классов'
                             ' AnalogSignal или DiscreteSignal')

    def __set_channel(self, ch, raise_error=False):
        trans_name0 = _translate(ch.name).strip('_').lstrip(_digits)
        trans_name = trans_name0
        if raise_error:
            if trans_name in self.__dict__:
                raise ValueError(
                    f'Канал с именем {ch.name} ({trans_name}) был добавлен ранее. Измените наименование сигнала')
        else:
            cnt = 0
            while trans_name in self.__dict__:
                trans_name = trans_name0 + str(cnt)
                cnt += 1
        self.__setattr__(trans_name, ch)

    def to_comtrade(self, name, path=None):
        """"Экспорт в формат COMTRADE 2001. Генерируется два файла: *.cfg, *.dat

        :param name: наименование осциллограммы, type: str
        :param path: путь, по которому будет сохранена осциллограмма
        :return: None
        """
        achs = self.__get_analog_channels()
        dchs = self.__get_discrete_channels()
        if path is None:
            cfg_path = name + '.cfg'
            dat_path = name + '.dat'
        else:
            cwd = _Path(path)
            cfg_path = cwd / f'{name}.cfg'
            dat_path = cwd / f'{name}.dat'

        with open(cfg_path, "w") as cfg:
            cfg.write('relaylab comtrade generator, 0, 2000\n')
            cfg.write(f'{len(achs) + len(dchs)}, {len(achs)}A, {len(dchs)}D\n')
            for nach, ach in enumerate(achs, start=1):
                cfg.write(f'{nach},{ach.name},,,,1,0,0,{min(ach.val)},{max(ach.val)},1,1,S\n')
            for ndch, dch in enumerate(dchs, start=1):
                cfg.write(f'{ndch},{dch.name},,,1\n')
            cfg.write('50\n')
            cfg.write('1\n')
            cfg.write(f'{self.Fs},{self.nmax}\n')
            tm = _datetime.now()
            mcs = int(str(tm.microsecond)[0:-3])
            time_str = f'{tm.day}/{tm.month}/{tm.year},{tm.hour:02}:{tm.minute:02}:{tm.second:02}:{mcs:03}\n'
            cfg.write(time_str)
            cfg.write(time_str)
            cfg.write('ASCII\n')
            cfg.write('1.0')
        # Создание и запись dat файла
        n = np.arange(1, self.nmax + 1)
        t = np.array((n / self.Fs * 1e6), dtype=np.int_)
        matrix = [n, t] + list(map(lambda ch: ch.val, achs)) + list(map(lambda ch: ch.val, dchs))
        matrix_nd = np.stack(matrix, axis=1)
        format = ['%1.1i'] * 2 + ['%1.6f'] * len(achs) + ['%1.1i'] * len(dchs)
        np.savetxt(dat_path, matrix_nd, fmt=format, delimiter=',')

    def __get_analog_channels(self):
        """Получение кортежа аналоговых каналов"""
        achs = []
        for ch in self.__dict__.values():
            if type(ch) == _AnalogSignal:
                achs.append(ch)
        return tuple(achs)

    def __get_discrete_channels(self):
        """Получение кортежа дискретных каналов"""
        dchs = []
        for ch in self.__dict__.values():
            if type(ch) == _DiscreteSignal:
                dchs.append(ch)
        return tuple(dchs)

    @property
    def anch(self) -> tuple[_AnalogSignal]:
        """Кортеж аналоговых каналов"""
        return self.__get_analog_channels()

    @property
    def dch(self) -> tuple[_DiscreteSignal]:
        """Кортеж аналоговых каналов"""
        return self.__get_discrete_channels()

    @property
    def time(self) -> np.ndarray:
        """Время"""
        return self.__get_analog_channels()[0].time

    def get_dataframe(self, analog_type: str = 'inst', hide_unchanged: bool = True) -> pd.DataFrame:
        """Функция возвращает DataFrame с аналоговыми и дискретными сигналами"

        :param analog_type: тип возвращаемых каналов 'inst', 'rms', 'dft', 'comp'
        :param hide_unchanged: скрыть не изменяющиеся каналы
        :return: DataFrame
        """
        #Выбор функции для преобразования аналоговых каналов в датафрейме
        analog_func_dict = {'inst': lambda ch: ch.val,
                            'rms': lambda ch: _sg.RMS(ch).val,
                            'dft': lambda ch: _sg.DFT(ch).abs().val,
                            'comp': lambda ch: _sg.DFT(ch).val}
        analog_func = analog_func_dict.setdefault(analog_type, lambda ch: ch.val)
        df = pd.DataFrame()
        df['time'] = self.__get_analog_channels()[0].time
        for ch in self.__get_analog_channels():
            df[ch.name] = analog_func(ch)
        for ch in self.__get_discrete_channels():
            if hide_unchanged and (max(ch.val) ^ min(ch.val)):
                df[ch.name] = ch.val
        return df

    def describe_analogs(self):
        """Вывод информации по аналоговым каналам осциллограммы"""
        achs = self.__get_analog_channels()
        df_achs = _pd.DataFrame()
        for ach in achs:
            df_achs[ach.name] = ach.dft_abs().val[int(self.Fs / 50):]
        df_achs_agg = df_achs.aggregate(func=(min, max)).transpose()
        return df_achs_agg.style.format('{:.3f}')

    def describe_discretes(self):
        """Вывод информации по дискретным каналам осциллограммы"""
        dchs = self.__get_discrete_channels()
        df_dchs = _pd.DataFrame()
        for dch in dchs:
            df_dchs[dch.name] = dch.val
        df_dchs_agg = df_dchs.aggregate(func=(min, max)).transpose()
        df_dchs_agg['change'] = (df_dchs_agg['max'] != df_dchs_agg['min'])
        return df_dchs_agg.style.highlight_max(color='#FFB02E', subset='change', axis=0)

    def __str__(self):
        """Описание осциллограммы"""
        if self.__comtrade is not None:
            desc = f'Наименование осциллограммы: {self.__comtrade.cfg.file_path}\n' \
                   f'Ревизия стандарта: {self.__comtrade.rev_year}\n' \
                   f'Объект установки устройства: {self.__comtrade.station_name}\n' \
                   f'Наименование устройства: {self.__comtrade.rec_dev_id}\n' \
                   f'Начало записи: {self.__comtrade.start_timestamp}\n' \
                   f'Длительность записи: {self.__comtrade.time[-1]:.2f}, c\n' \
                   f'Количество аналоговых каналов: {self.__comtrade.analog_count}\n' \
                   f'Количество дискретных каналов: {self.__comtrade.status_count}'
        else:
            desc = f'Длительность записи: {self.tmax:.2f}, c\n' \
                   f'Количество аналоговых каналов: {len(self.__get_analog_channels())}\n' \
                   f'Количество дискретных каналов: {len(self.__get_discrete_channels())}'
        return desc


class _Examples:
    def __init__(self, directory='records'):
        cwd = _Path(_lb.__file__).parent / directory
        for file in cwd.iterdir():
            if file.suffix == '.cfg':
                self.__setattr__(_translate(file.stem).strip('_'), Record(str(file)))


examples = _Examples(directory='records')

if __name__ == '__main__':
    print(examples.__dict__)
