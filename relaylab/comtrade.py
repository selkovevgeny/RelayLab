# -*- coding: utf-8 -*-
# MIT License

# Copyright (c) 2018 David Rodrigues Parrini

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import array
import datetime as dt
import errno
import io
import math
import os
import re
import struct
import sys
from typing import Union
import warnings

try:
    import numpy

    _HAS_NUMPY = True
except ModuleNotFoundError:
    _HAS_NUMPY = False

try:
    import pandas as pd

    _HAS_PANDAS = True
except ModuleNotFoundError:
    _HAS_PANDAS = False


# COMTRADE standard revisions
REV_1991 = "1991"
REV_1999 = "1999"
REV_2000 = "2000"
REV_2001 = "2001"
REV_2013 = "2013"

# DAT file format types
_TYPE_ASCII = "ASCII"
_TYPE_BINARY = "BINARY"
_TYPE_BINARY32 = "BINARY32"
_TYPE_FLOAT32 = "FLOAT32"

# Special values
_TIMESTAMP_MISSING = 0xFFFFFFFF

# CFF headers
_CFF_HEADER_REXP = r"(?i)--- file type: ([a-z]+)(?:\s+([a-z0-9]+)(?:\s*\:\s*([0-9]+))?)? ---$"

# common separator character of data fields of CFG and ASCII DAT files
_SEPARATOR = ","

# timestamp regular expression
_re_date = re.compile(r"([0-9]{1,2})/([0-9]{1,2})/([0-9]{2,4})")
_re_time = re.compile(r"([0-9]{1,2}):([0-9]{2}):([0-9]{1,2})(.([0-9]{1,12}))?")

# Non-standard revision warning
_WARNING_UNKNOWN_REVISION = "Unknown standard revision \"{}\""
# Date time with nanoseconds resolution warning
_WARNING_DATETIME_NANO = "Unsupported datetime objects with nanoseconds \
resolution. Using truncated values."
# Date time with year 0, month 0 and/or day 0.
_WARNING_MIN_DATE = "Missing date values. Using minimum values: {}."


class ComtradeError(Exception):
    pass


def _read_sep_values(line, expected: int = -1, default: str = ''):
    values = tuple(map(lambda cell: cell.strip(), line.split(_SEPARATOR)))
    if expected == -1 or len(values) == expected:
        return values
    return [values[i] if i < len(values) else default
            for i in range(expected)]


def _preallocate_values(array_type, size, use_numpy_arrays):
    type_mapping_numpy = {"f": "float32", "d": "float64", "i": "int32", "l": "int64"}
    if _HAS_NUMPY and use_numpy_arrays:
        return numpy.zeros(size, dtype=type_mapping_numpy[array_type])
    return array.array(array_type, [0]) * size


def _prevent_null(str_value: str, value_type: type, default_value):
    if len(str_value.strip()) == 0:
        return default_value
    else:
        return value_type(str_value)


def _get_date(date_str: str) -> tuple:
    m = _re_date.match(date_str)
    if m is not None:
        day = int(m.group(1))
        month = int(m.group(2))
        year = int(m.group(3))
        return day, month, year
    return 0, 0, 0


def _get_time(time_str: str, ignore_warnings: bool = False) -> tuple:
    m = _re_time.match(time_str)
    if m is not None:
        hour = int(m.group(1))
        minute = int(m.group(2))
        second = int(m.group(3))
        frac_sec_str = m.group(5)
        # Pad fraction of seconds with 0s to the right
        if len(frac_sec_str) <= 6:
            frac_sec_str = fill_with_zeros_to_the_right(frac_sec_str, 6)
        else:
            frac_sec_str = fill_with_zeros_to_the_right(frac_sec_str, 9)

        frac_second = int(frac_sec_str)
        in_nanoseconds = len(frac_sec_str) > 6
        microsecond = frac_second

        if in_nanoseconds:
            # Nanoseconds resolution is not supported by datetime module, so it's
            # converted to integer below.
            if not ignore_warnings:
                warnings.warn(Warning(_WARNING_DATETIME_NANO))
            microsecond = int(microsecond * 1E-3)
        return hour, minute, second, microsecond, in_nanoseconds


def fill_with_zeros_to_the_right(number_str: str, width: int):
    actual_len = len(number_str)
    if actual_len < width:
        difference = width - actual_len
        fill_chars = "0" * difference
        return number_str + fill_chars
    return number_str


def _get_same_case(original_ext: str, other_ext: str) -> str:
    """Returns each other_ext character with the same case as original_ext's."""
    same_case = ""
    for i in range(len(original_ext)):
        if i < len(other_ext):
            if original_ext[i].isupper():
                same_case += other_ext[i].upper()
            else:
                same_case += other_ext[i].lower()
        else:
            break
    return same_case


def _read_timestamp(timestamp_line: str, rev_year: str, ignore_warnings: bool = False) -> tuple:
    """Process comma separated fields and returns a tuple containing the timestamp
    and a boolean value indicating whether nanoseconds are used.
    Can possibly return the timestamp 00/00/0000 00:00:00.000 for empty strings
    or empty pairs."""
    day, month, year, hour, minute, second, microsecond = (0,) * 7
    nano_sec = False
    if len(timestamp_line.strip()) > 0:
        values = _read_sep_values(timestamp_line, 2)
        if len(values) >= 2:
            date_str, time_str = values[0:2]
            if len(date_str.strip()) > 0:
                # 1991 Format Uses mm/dd/yyyy format
                if rev_year == REV_1991:
                    month, day, year = _get_date(date_str)
                # Modern Formats Use dd/mm/yyyy format
                else:
                    day, month, year = _get_date(date_str)
            if len(time_str.strip()) > 0:
                hour, minute, second, microsecond, \
                    nano_sec = _get_time(time_str, ignore_warnings)

    using_min_data = False
    if year <= 0:
        year = dt.MINYEAR
        using_min_data = True
    if month <= 0:
        month = 1
        using_min_data = True
    if day <= 0:
        day = 1
        using_min_data = True
    # Timezone info unsupported
    tzinfo = None
    try:
        timestamp = dt.datetime(year, month, day, hour, minute, second,
                                microsecond, tzinfo)
    except ValueError as e:
        warnings.warn(Warning(f'{e}. Неверный формат даты: {timestamp_line}'))
        timestamp = dt.datetime(1970, 1, 1, 0, 0, 0,
                                0, tzinfo)

    if not ignore_warnings and using_min_data:
        warnings.warn(Warning(_WARNING_MIN_DATE.format(str(timestamp))))
    return timestamp, nano_sec


def _eof_strip(line: str) -> str:
    """Strip line of whitespace and <sub> (0x1A) control character that may appear
     appended to the end of text files on some platforms."""
    return _replace_sub(line).strip()


def _replace_sub(line: str) -> str:
    """Replace <sub> (0x1A) control character that may appear appended to the end
    of text files on some platforms."""
    return line.replace(chr(0x1A), "")


class Cfg:
    """Parses and stores Comtrade's CFG data."""
    # time base units
    TIME_BASE_NANO_SEC = 1E-9
    TIME_BASE_MICRO_SEC = 1E-6

    def __init__(self, **kwargs):
        """
        Cfg object constructor.

        Keyword arguments:
        ignore_warnings -- whether warnings are displayed in stdout 
            (default: False)
        """
        self._file_path = None
        # implicit data
        self._time_base = self.TIME_BASE_MICRO_SEC

        # Default CFG data
        self._station_name = ""
        self._rec_dev_id = ""
        self._rev_year = REV_2013
        self._channels_count = 0
        self._analog_channels = []
        self._status_channels = []
        self._analog_count = 0
        self._status_count = 0
        self._frequency = 0.0
        self._nrates = 1
        self._sample_rates = []
        self._timestamp_critical = False
        self._start_timestamp = dt.datetime(1900, 1, 1)
        self._trigger_timestamp = dt.datetime(1900, 1, 1)
        self._ft = _TYPE_ASCII
        self._time_multiplier = 1.0
        # 2013 standard revision information
        # time_code,local_code = 0,0 means local time is UTC
        self._time_code = 0
        self._local_code = 0
        # tmq_code,leapsec
        self._tmq_code = 0
        self._leap_second = 0

        self._ignore_warnings = kwargs.get("ignore_warnings", False)

    @property
    def file_path(self) -> Union[str, None]:
        """Return the CFG file path."""
        return self._file_path

    @property
    def station_name(self) -> str:
        """Return the recording device's station name."""
        return self._station_name

    @property
    def rec_dev_id(self) -> str:
        """Return the recording device id."""
        return self._rec_dev_id

    @property
    def rev_year(self) -> str:
        """Return the COMTRADE revision year."""
        return self._rev_year

    @property
    def channels_count(self) -> int:
        """Return the number of channels, total."""
        return self._channels_count

    @property
    def analog_channels(self) -> list:
        """Return the analog channels list with complete channel description."""
        return self._analog_channels

    @property
    def status_channels(self) -> list:
        """Return the status channels list with complete channel description."""
        return self._status_channels

    @property
    def analog_count(self) -> int:
        """Return the number of analog channels."""
        return self._analog_count

    @property
    def status_count(self) -> int:
        """Return the number of status channels."""
        return self._status_count

    @property
    def time_base(self) -> float:
        """Return the time base."""
        return self._time_base

    @property
    def frequency(self) -> float:
        """Return the measured line frequency in Hertz."""
        return self._frequency

    @property
    def ft(self) -> str:
        """Return the expected DAT file format."""
        return self._ft

    @property
    def timemult(self) -> float:
        """Return the DAT time multiplier (Default = 1)."""
        return self._time_multiplier

    @property
    def timestamp_critical(self) -> bool:
        """Returns whether the DAT file must contain non-zero
         timestamp values."""
        return self._timestamp_critical

    @property
    def start_timestamp(self) -> dt.datetime:
        """Return the recording start time stamp as a datetime object."""
        return self._start_timestamp

    @property
    def trigger_timestamp(self) -> dt.datetime:
        """Return the trigger time stamp as a datetime object."""
        return self._trigger_timestamp

    @property
    def nrates(self) -> int:
        """Return the number of different sample rates within the DAT file."""
        return self._nrates

    @property
    def sample_rates(self) -> list:
        """
        Return a list with pairs describing the number of samples for a given
        sample rate.
        """
        return self._sample_rates

    # Deprecated properties - Changed "Digital" for "Status"
    @property
    def digital_channels(self) -> list:
        """Returns the status channels bidimensional values list."""
        if not self._ignore_warnings:
            warnings.warn(FutureWarning("digital_channels is deprecated, "
                                        "use status_channels instead."))
        return self._status_channels

    @property
    def digital_count(self) -> int:
        """Returns the number of status channels."""
        if not self._ignore_warnings:
            warnings.warn(FutureWarning("digital_count is deprecated, "
                                        "use status_count instead."))
        return self._status_count

    def load(self, filepath, **kwargs):
        """Load and read a CFG file contents."""
        self._file_path = filepath

        if os.path.isfile(self._file_path):
            filtered_kwargs = {
                "encoding": kwargs.get("encoding", "utf-8")
            }
            with open(self._file_path, "r", **filtered_kwargs) as cfg:
                self._read_io(cfg)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    self._file_path)

    def read(self, cfg_lines):
        """Read CFG-format data of a FileIO or StringIO object."""
        if type(cfg_lines) is str:
            self._read_io(io.StringIO(cfg_lines))
        else:
            self._read_io(cfg_lines)

    def _read_io(self, cfg):
        """Read CFG-format lines and stores its data."""
        line_count = 0
        self._nrates = 1
        self._sample_rates = []
        self._analog_channels = []
        self._status_channels = []

        # First line
        line = cfg.readline()
        # station, device, and comtrade standard revision information
        packed = _read_sep_values(line)
        if 3 == len(packed):
            # only 1999 revision and above has the standard revision year
            self._station_name, self._rec_dev_id, self._rev_year = packed
            self._rev_year = self._rev_year.strip()

            if self._rev_year not in (REV_1991, REV_1999, REV_2000, REV_2001, REV_2013):
                if not self._ignore_warnings:
                    msg = _WARNING_UNKNOWN_REVISION.format(self._rev_year)
                    warnings.warn(Warning(msg))
        else:
            self._station_name, self._rec_dev_id = packed
            self._rev_year = REV_1991
        line_count = line_count + 1

        # Second line
        line = cfg.readline()
        # number of channels and its type
        totchn, achn, schn = _read_sep_values(line, 3, '0')
        self._channels_count = int(totchn)
        self._analog_count = int(achn[:-1])
        self._status_count = int(schn[:-1])
        self._analog_channels = [None] * self._analog_count
        self._status_channels = [None] * self._status_count
        line_count = line_count + 1

        # Analog channel description lines
        for ichn in range(self._analog_count):
            line = cfg.readline()
            packed = _read_sep_values(line, 13, '0')
            # unpack values
            n, name, ph, ccbm, uu, a, b, skew, cmin, cmax, \
                primary, secondary, pors = packed
            # type conversion
            n = int(n)
            a = float(a)
            b = _prevent_null(b, float, 0.0)
            skew = _prevent_null(skew, float, 0.0)
            cmin = float(cmin)
            cmax = float(cmax)
            primary = float(primary)
            secondary = float(secondary)
            self.analog_channels[ichn] = AnalogChannel(n, a, b, skew,
                                                       cmin, cmax, name, uu, ph, ccbm, primary, secondary, pors)
            line_count = line_count + 1

        # Status channel description lines
        for ichn in range(self._status_count):
            line = cfg.readline()
            # unpack values
            packed = _read_sep_values(line, 5, '0')
            n, name, ph, ccbm, y = packed
            # type conversion
            n = int(n)
            y = _prevent_null(y, int, 0)  # TODO: actually a critical data. In the future add a warning.
            self.status_channels[ichn] = StatusChannel(n, name, ph, ccbm, y)
            line_count = line_count + 1

        # Frequency line
        line = cfg.readline()
        if len(line.strip()) > 0:
            self._frequency = float(line.strip())
        line_count = line_count + 1

        # Nrates line
        # number of different sample rates
        line = cfg.readline()
        self._nrates = int(line.strip())
        if self._nrates == 0:
            self._nrates = 1
            self._timestamp_critical = True
        else:
            self._timestamp_critical = False
        line_count = line_count + 1

        for inrate in range(self._nrates):
            line = cfg.readline()
            # each sample rate
            samp, endsamp = _read_sep_values(line)
            samp = float(samp)
            endsamp = int(endsamp)
            self.sample_rates.append([samp, endsamp])
            line_count = line_count + 1

        # First data point time and time base
        line = cfg.readline()
        ts_str = line.strip()
        self._start_timestamp, nanosec = _read_timestamp(
            ts_str,
            self.rev_year,
            self._ignore_warnings
        )
        self._time_base = self._get_time_base(nanosec)
        line_count = line_count + 1

        # Event data point and time base
        line = cfg.readline()
        ts_str = line.strip()
        self._trigger_timestamp, nanosec = _read_timestamp(
            ts_str,
            self.rev_year,
            self._ignore_warnings
        )
        self._time_base = min([self.time_base, self._get_time_base(nanosec)])
        line_count = line_count + 1

        # DAT file type
        line = cfg.readline()
        self._ft = line.strip()
        line_count = line_count + 1

        # Timestamp multiplication factor
        if self._rev_year in (REV_1999, REV_1999, REV_2001, REV_2013):
            line = _eof_strip(cfg.readline())
            if len(line) > 0:
                self._time_multiplier = float(line)
            else:
                self._time_multiplier = 1.0
            line_count = line_count + 1

        # time_code and local_code
        if self._rev_year == REV_2013:
            line = _eof_strip(cfg.readline())

            if line:
                self._time_code, self._local_code = _read_sep_values(line)
                line_count = line_count + 1

                line = _eof_strip(cfg.readline())
                # time_code and local_code
                self._tmq_code, self._leap_second = _read_sep_values(line)

    def _get_time_base(self, using_nanoseconds: bool):
        """
        Return the time base, which is based on the fractionary part of the 
        seconds in a timestamp (00.XXXXX).
        """
        if using_nanoseconds:
            return self.TIME_BASE_NANO_SEC
        else:
            return self.TIME_BASE_MICRO_SEC


class Comtrade:
    """Parses and stores Comtrade data."""
    # extensions
    EXT_CFG = "cfg"
    EXT_DAT = "dat"
    EXT_INF = "inf"
    EXT_HDR = "hdr"
    # format specific
    ASCII_SEPARATOR = ","

    def __init__(self, **kwargs):
        """
        Comtrade object constructor.

        Keyword arguments:
        ignore_warnings -- whether warnings are displayed in stdout 
            (default: False).
        """
        self.file_path = ""

        self._cfg = Cfg(**kwargs)

        # Default CFG data
        self._analog_channel_ids = []
        self._analog_phases = []
        self._status_channel_ids = []
        self._status_phases = []
        self._timestamp_critical = False

        # Data types
        self._use_numpy_arrays = kwargs.get("use_numpy_arrays", False)
        self._use_double_precision = kwargs.get("use_double_precision", False)

        # DAT file data
        self._time_values = _preallocate_values(
            "d" if self._use_double_precision else "f",
            0,
            self._use_numpy_arrays,
        )
        self._analog_values = []
        self._status_values = []

        # Additional CFF data (or additional comtrade files)
        self._hdr = None
        self._inf = None

        if "ignore_warnings" in kwargs:
            self.ignore_warnings = kwargs["ignore_warnings"]
        else:
            self.ignore_warnings = False

    @property
    def station_name(self) -> str:
        """Return the recording device's station name."""
        return self._cfg.station_name

    @property
    def rec_dev_id(self) -> str:
        """Return the recording device id."""
        return self._cfg.rec_dev_id

    @property
    def rev_year(self) -> str:
        """Return the COMTRADE revision year."""
        return self._cfg.rev_year

    @property
    def cfg(self) -> Cfg:
        """Return the underlying CFG class instance."""
        return self._cfg

    @property
    def hdr(self):
        """Return the HDR file contents."""
        return self._hdr

    @property
    def inf(self):
        """Return the INF file contents."""
        return self._inf

    @property
    def analog_channel_ids(self) -> list:
        """Returns the analog channels name list."""
        return self._analog_channel_ids

    @property
    def analog_phases(self) -> list:
        """Returns the analog phase name list."""
        return self._analog_phases

    @property
    def status_channel_ids(self) -> list:
        """Returns the status channels name list."""
        return self._status_channel_ids

    @property
    def status_phases(self) -> list:
        """Returns the status phase name list."""
        return self._status_phases

    @property
    def time(self) -> list:
        """Return the time values list."""
        return self._time_values

    @property
    def analog(self) -> list:
        """Return the analog channel values bidimensional list."""
        return self._analog_values

    @property
    def status(self) -> list:
        """Return the status channel values bidimensional list."""
        return self._status_values

    @property
    def total_samples(self) -> int:
        """Return the total number of samples (per channel)."""
        return self._total_samples

    @property
    def frequency(self) -> float:
        """Return the measured line frequency in Hertz."""
        return self._cfg.frequency

    @property
    def start_timestamp(self):
        """Return the recording start time stamp as a datetime object."""
        return self._cfg.start_timestamp

    @property
    def trigger_timestamp(self):
        """Return the trigger time stamp as a datetime object."""
        return self._cfg.trigger_timestamp

    @property
    def channels_count(self) -> int:
        """Return the number of channels, total."""
        return self._cfg.channels_count

    @property
    def analog_count(self) -> int:
        """Return the number of analog channels."""
        return self._cfg.analog_count

    @property
    def status_count(self) -> int:
        """Return the number of status channels."""
        return self._cfg.status_count

    @property
    def trigger_time(self) -> float:
        """Return relative trigger time in seconds."""
        stt = self._cfg.start_timestamp
        trg = self._cfg.trigger_timestamp
        tdiff = trg - stt
        tsec = (tdiff.days * 60 * 60 * 24) + tdiff.seconds + (tdiff.microseconds * 1E-6)
        return tsec

    @property
    def time_base(self) -> float:
        """Return the time base."""
        return self._cfg.time_base

    @property
    def ft(self) -> str:
        """Return the expected DAT file format."""
        return self._cfg.ft

    # Deprecated properties - Changed "Digital" for "Status"
    @property
    def digital_channel_ids(self) -> list:
        """Returns the status channels name list."""
        if not self.ignore_warnings:
            warnings.warn(FutureWarning("digital_channel_ids is deprecated, use status_channel_ids instead."))
        return self._status_channel_ids

    @property
    def digital(self) -> list:
        """Returns the status channels bidimensional values list."""
        if not self.ignore_warnings:
            warnings.warn(FutureWarning("digital is deprecated, use status instead."))
        return self._status_values

    @property
    def digital_count(self) -> int:
        """Returns the number of status channels."""
        if not self.ignore_warnings:
            warnings.warn(FutureWarning("digital_count is deprecated, use status_count instead."))
        return self._cfg.status_count

    def _get_dat_reader(self):
        # case-insensitive comparison of file format
        ft_upper = self.ft.upper()
        dat_kwargs = {"use_numpy_arrays": self._use_numpy_arrays,
                      "use_double_precision": self._use_double_precision,
                      "rev_year": self.rev_year}
        if ft_upper == _TYPE_ASCII:
            dat = _AsciiDatReader(**dat_kwargs)
        elif ft_upper == _TYPE_BINARY:
            dat = _BinaryDatReader(**dat_kwargs)
        elif ft_upper == _TYPE_BINARY32:
            dat = _Binary32DatReader(**dat_kwargs)
        elif ft_upper == _TYPE_FLOAT32:
            dat = _Float32DatReader(**dat_kwargs)
        else:
            raise ComtradeError("Not supported data file format: {}".format(self.ft))
        return dat

    def read(self, cfg_lines, dat_lines_or_bytes) -> None:
        """
        Read CFG and DAT files contents. Expects FileIO or StringIO objects.
        """
        self._cfg.read(cfg_lines)

        # channel ids
        self._cfg_extract_channels_ids(self._cfg)

        # channel phases
        self._cfg_extract_phases(self._cfg)

        dat = self._get_dat_reader()
        dat.read(dat_lines_or_bytes, self._cfg)

        # copy .dat object information
        self._dat_extract_data(dat)

    def _cfg_extract_channels_ids(self, cfg) -> None:
        self._analog_channel_ids = [channel.name for channel in cfg.analog_channels]
        self._status_channel_ids = [channel.name for channel in cfg.status_channels]

    def _cfg_extract_phases(self, cfg) -> None:
        self._analog_phases = [channel.ph for channel in cfg.analog_channels]
        self._status_phases = [channel.ph for channel in cfg.status_channels]

    def _dat_extract_data(self, dat) -> None:
        self._time_values = dat.time
        self._analog_values = dat.analog
        self._status_values = dat.status
        self._total_samples = dat.total_samples

    def to_dataframe(self, **kwargs) -> "pd.DataFrame":
        """Return a pandas DataFrame object with comtrade data."""
        if _HAS_PANDAS:
            index_type = kwargs.get("index_type", "time")
            data = {
                "time": self.time,
            }
            data.update({self.analog_channel_ids[i]: self.analog[i] for i in range(self.analog_count)})
            data.update({self.status_channel_ids[i]: self.status[i] for i in range(self.status_count)})
            df = pd.DataFrame(data)

            if index_type == "time":
                df.set_index("time", inplace=True)
            elif index_type == "sample":
                pass
            return df
        raise ComtradeError("pandas package is not installed.")

    def load(self, cfg_file, dat_file=None, **kwargs) -> "Comtrade":
        """
        Load CFG, DAT, INF, and HDR files. Each must be a FileIO or StringIO
        object. dat_file, inf_file, and hdr_file are optional (Default: None).

        cfg_file is the cfg file path, including its extension.
        dat_file is optional, and may be set if the DAT file name differs from 
            the CFG file name.

        Keyword arguments:
        inf_file -- optional INF file path (Default = None)
        hdr_file -- optional HDR file path (Default = None)
        """
        inf_file = kwargs.get("inf_file", None)
        hdr_file = kwargs.get("hdr_file", None)

        # which extension: CFG or CFF?
        file_ext = cfg_file[-3:]
        file_ext_upper = file_ext.upper()
        if file_ext_upper == "CFF":
            # check if the CFF file exists
            self._load_cff(cfg_file)

        elif file_ext_upper == "CFG":
            basename = cfg_file[:-3]
            # if not informed, infer dat_file with cfg_file
            if dat_file is None:
                dat_file = cfg_file[:-3] + _get_same_case(file_ext, self.EXT_DAT)

            if inf_file is None:
                inf_file = basename + _get_same_case(file_ext, self.EXT_INF)

            if hdr_file is None:
                hdr_file = basename + _get_same_case(file_ext, self.EXT_HDR)

            # load both cfg and dat
            file_kwargs = {}
            if "encoding" in kwargs:
                file_kwargs["encoding"] = kwargs["encoding"]
            self._load_cfg(cfg_file, **file_kwargs)
            self._load_dat(dat_file)

            # Load additional inf and hdr files, if they exist.
            self._load_inf(inf_file, **file_kwargs)
            self._load_hdr(hdr_file, **file_kwargs)

        else:
            raise ComtradeError(r"Expected CFG file path, instead got \"{}\".".format(cfg_file))

        return self

    def _load_cfg(self, cfg_filepath, **kwargs):
        self._cfg.load(cfg_filepath, **kwargs)

        # channel ids
        self._cfg_extract_channels_ids(self._cfg)

        # channel phases
        self._cfg_extract_phases(self._cfg)

    def _load_dat(self, dat_filepath):
        dat = self._get_dat_reader()
        dat.load(dat_filepath, self._cfg)

        # copy .dat object information
        self._dat_extract_data(dat)

    def _load_inf(self, inf_file, **kwargs):
        if os.path.exists(inf_file):
            kwargs["encoding"] = kwargs.get("encoding", "utf-8")
            with open(inf_file, 'r', **kwargs) as file:
                self._inf = _replace_sub(file.read())
                if len(self._inf) == 0:
                    self._inf = None
        else:
            self._inf = None

    def _load_hdr(self, hdr_file, **kwargs):
        if os.path.exists(hdr_file):
            kwargs["encoding"] = kwargs.get("encoding", "utf-8")
            with open(hdr_file, 'r', **kwargs) as file:
                self._hdr = _replace_sub(file.read())
                if len(self._hdr) == 0:
                    self._hdr = None
        else:
            self._hdr = None

    @staticmethod
    def _read_mixed_text_bin_data_as_text(cff_file) -> str:
        chunk_delimiter = b"\n"
        current_chunk = []
        while True:
            current_value = cff_file.read(1)
            if not current_value:
                break
            if current_value != chunk_delimiter:
                current_chunk.append(current_value)
            else:
                yield b"".join(current_chunk).decode("utf-8", errors="ignore").strip()
                current_chunk = []

    def _load_cff(self, cff_file_path: str, **kwargs):
        # stores each file type lines
        cfg_lines = []
        dat_lines = []
        hdr_lines = []
        inf_lines = []
        header_re = re.compile(_CFF_HEADER_REXP)

        with open(cff_file_path, "rb") as cff_file:
            # file type: CFG, HDR, INF, DAT
            ftype = None
            # file format: ASCII, BINARY, BINARY32, FLOAT32
            fformat = None

            line_number = 0
            last_match = None
            for current_line in self._read_mixed_text_bin_data_as_text(cff_file):
                line_number += 1
                mobj = header_re.match(current_line.strip().upper())
                if mobj is not None:
                    last_match = mobj
                    groups = last_match.groups()
                    ftype = groups[0]
                    if groups[1] is not None:
                        fformat = groups[1]
                        fbytes_obj = groups[2]
                        if ftype == "DAT" and fformat != _TYPE_ASCII:
                            break

                elif last_match is not None and ftype == "CFG":
                    cfg_lines.append(current_line.strip())

                elif last_match is not None and ftype == "HDR":
                    hdr_lines.append(current_line.strip())

                elif last_match is not None and ftype == "INF":
                    inf_lines.append(current_line.strip())

                elif last_match is not None and ftype == "DAT" and fformat == _TYPE_ASCII:
                    dat_lines.append(current_line.strip())

            if fformat == _TYPE_ASCII:
                # process ASCII CFF data
                self.read("\n".join(cfg_lines), "\n".join(dat_lines))
            else:
                # read remaining data as .dat values.
                dat_bytes = cff_file.read()
                self.read("\n".join(cfg_lines), dat_bytes)

        # stores additional data
        self._hdr = "\n".join(hdr_lines)

        self._inf = "\n".join(inf_lines)

    def cfg_summary(self):
        """Returns the CFG attributes summary string."""
        header_line = "Channels (Analog,Digital,Total): {}A + {}D = {}"
        sample_line = "Sample rate of {} Hz to the sample #{}"
        interval_line = "From {} to {} with time mult. = {}"
        format_line = "{} format"

        lines = [header_line.format(self.analog_count, self.status_count,
                                    self.channels_count),
                 "Line frequency: {} Hz".format(self.frequency)]
        for i in range(self._cfg.nrates):
            rate, points = self._cfg.sample_rates[i]
            lines.append(sample_line.format(rate, points))
        lines.append(interval_line.format(self.start_timestamp,
                                          self.trigger_timestamp,
                                          self._cfg.timemult))
        lines.append(format_line.format(self.ft))
        return "\n".join(lines)


class Channel:
    """Holds common channel description data."""

    def __init__(self, n=1, name='', ph='', ccbm=''):
        """Channel abstract class constructor."""
        self.n = n
        self.name = name
        self.ph = ph
        self.ccbm = ccbm

    def __str__(self):
        return ','.join([str(self.n), self.name, self.ph, self.ccbm])


class StatusChannel(Channel):
    """Holds status channel description data."""

    def __init__(self, n: int, name='', ph='', ccbm='', y=0):
        """StatusChannel class constructor."""
        super().__init__(n, name, ph, ccbm)
        self.name = name
        self.n = n
        self.name = name
        self.ph = ph
        self.ccbm = ccbm
        self.y = y

    def __str__(self):
        fields = [str(self.n), self.name, self.ph, self.ccbm, str(self.y)]
        return ','.join(fields)


class AnalogChannel(Channel):
    """Holds analog channel description data."""

    def __init__(self, n: int, a: float, b=0.0, skew=0.0, cmin=-32767.0,
                 cmax=32767.0, name='', uu='', ph='', ccbm='', primary=1.0,
                 secondary=1.0, pors='P'):
        """AnalogChannel class constructor."""
        super().__init__(n, name, ph, ccbm)
        self.name = name
        self.uu = uu
        self.n = n
        self.a = a
        self.b = b
        self.skew = skew
        self.cmin = cmin
        self.cmax = cmax
        # misc
        self.uu = uu
        self.ph = ph
        self.ccbm = ccbm
        self.primary = primary
        self.secondary = secondary
        self.pors = pors

    def __str__(self):
        fields = [str(self.n), self.name, self.ph, self.ccbm, self.uu,
                  str(self.a), str(self.b), str(self.skew), str(self.cmin),
                  str(self.cmax), str(self.primary), str(self.secondary), self.pors]
        return ','.join(fields)


class _DatReader:
    """Abstract DatReader class. Used to parse DAT file contents."""
    read_mode = "r"

    def __init__(self, **kwargs):
        """DatReader class constructor."""
        self._use_numpy_arrays = kwargs.get("use_numpy_arrays", False)
        self._use_double_precision = kwargs.get("use_double_precision", False)
        self._rev_year = kwargs.get("rev_year", REV_2013)

        self.file_path = ""
        self._content = None
        self._cfg = None
        self.time = _preallocate_values(
            "d" if self._use_double_precision else "f",
            0,
            self._use_numpy_arrays,
        )
        self.analog = []
        self.status = []
        self._total_samples = 0

        # To be replaced by subclasses.
        self.DATA_MISSING = None

    @property
    def total_samples(self):
        """Return the total samples (per channel)."""
        return self._total_samples

    def load(self, dat_filepath, cfg, **kwargs):
        """Load a DAT file and parse its contents."""
        self.file_path = dat_filepath
        self._content = None
        if os.path.isfile(self.file_path):
            # extract CFG file information regarding data dimensions
            self._cfg = cfg
            self._preallocate()
            if "encoding" not in kwargs and self.read_mode != "rb":
                kwargs["encoding"] = "utf-8"
            with open(self.file_path, self.read_mode, **kwargs) as contents:
                self.parse(contents)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    self.file_path)

    def read(self, dat_lines, cfg):
        """
        Read a DAT file contents, expecting a list of string or FileIO object.
        """
        self.file_path = None
        self._content = dat_lines
        self._cfg = cfg
        self._preallocate()
        self.parse(dat_lines)

    def _preallocate(self):
        # read from the cfg file the number of samples in the dat file
        steps = self._cfg.sample_rates[-1][1]  # last samp field
        self._total_samples = steps

        # analog and status count
        analog_count = self._cfg.analog_count
        status_count = self._cfg.status_count

        # preallocate analog and status values
        self.time = _preallocate_values(
            "d" if self._use_double_precision else "f",
            steps,
            self._use_numpy_arrays,
        )
        self.analog = [None] * analog_count
        self.status = [None] * status_count
        # preallocate each channel values with zeros
        for i in range(analog_count):
            self.analog[i] = _preallocate_values(
                "d" if self._use_double_precision else "f",
                steps,
                self._use_numpy_arrays,
            )
        for i in range(status_count):
            self.status[i] = _preallocate_values("i", steps,
                                                 self._use_numpy_arrays)

    def _get_samp(self, n) -> float:
        """Get the sampling rate for a sample n (1-based index)."""
        # TODO: make tests.
        last_sample_rate = 1.0
        for samp, endsamp in self._cfg.sample_rates:
            if n <= endsamp:
                return samp
        return last_sample_rate

    def _get_time(self, n: int, ts_value: float, time_base: float,
                  time_multiplier: float):
        # TODO: add option to enforce dat file timestamp, when available.
        # TODO: make tests.
        sample_rate = self._get_samp(n)
        if not self._cfg.timestamp_critical or ts_value == _TIMESTAMP_MISSING:
            # if the timestamp is missing, use calculated.
            if sample_rate != 0.0:
                return (n - 1) / sample_rate
            else:
                raise ComtradeError("Missing timestamp and no sample rate "
                                    "provided.")
        else:
            # Use provided timestamp if it's not missing
            return ts_value * time_base * time_multiplier

    def filter_missing(self, value) -> float:
        return float(value) if value != self.DATA_MISSING else float('nan')

    def parse(self, contents):
        """Virtual method, parse DAT file contents."""
        pass


class _AsciiDatReader(_DatReader):
    """ASCII format DatReader subclass."""

    def __init__(self, **kwargs):
        # Call the initialization for the inherited class
        super().__init__(**kwargs)
        self.ASCII_SEPARATOR = _SEPARATOR

        if self._rev_year == REV_1991:
            self.DATA_MISSING = ""
        else:
            self.DATA_MISSING = "99999"

    def parse(self, contents):
        """Parse a ASCII file contents."""
        # Check if contents has been read as an io.BytesIO object, if so, decode into a string.
        contents = contents.decode() if (type(contents) == bytes) else contents

        analog_count = self._cfg.analog_count
        status_count = self._cfg.status_count
        time_mult = self._cfg.timemult
        time_base = self._cfg.time_base

        # auxiliary vectors (channels gains and offsets)
        a = [x.a for x in self._cfg.analog_channels]
        b = [x.b for x in self._cfg.analog_channels]

        # extract lines
        if type(contents) is str:
            lines = contents.splitlines()
        else:
            lines = contents

        line_number = 0
        for line in lines:
            if len(line) > 2:
                line_number = line_number + 1
                if line_number > self._total_samples:
                    break
                values = line.strip().split(self.ASCII_SEPARATOR)

                n = int(float(values[0]))
                # Read time
                ts_val = float(values[1])
                ts = self._get_time(n, ts_val, time_base, time_mult)

                avalues = [value * a[i] + b[i] if not math.isnan(value) else float('nan')
                           for i, value in enumerate(map(self.filter_missing, values[2:analog_count + 2]))]
                svalues = [int(x) for x in values[len(values) - status_count:]]

                # store
                self.time[line_number - 1] = ts
                for i in range(analog_count):
                    self.analog[i][line_number - 1] = avalues[i]
                for i in range(status_count):
                    self.status[i][line_number - 1] = svalues[i]


class _BinaryDatReader(_DatReader):
    """16-bit binary format DatReader subclass."""

    def __init__(self, **kwargs):
        # Call the initialization for the inherited class
        super().__init__(**kwargs)
        self.ANALOG_BYTES = 2
        self.STATUS_BYTES = 2
        self.TIME_BYTES = 4
        self.SAMPLE_NUMBER_BYTES = 4

        if self._rev_year == REV_1991:
            self.DATA_MISSING = -1  # 0xFFFF
        else:
            self.DATA_MISSING = -32768  # 0x8000

        self.read_mode = "rb"

        if struct.calcsize("L") == 4:
            self.STRUCT_FORMAT = "LL {acount:d}h {dcount:d}H"
            self.STRUCT_FORMAT_ANALOG_ONLY = "LL {acount:d}h"
            self.STRUCT_FORMAT_STATUS_ONLY = "LL {dcount:d}H"
        else:
            self.STRUCT_FORMAT = "II {acount:d}h {dcount:d}H"
            self.STRUCT_FORMAT_ANALOG_ONLY = "II {acount:d}h"
            self.STRUCT_FORMAT_STATUS_ONLY = "II {dcount:d}H"

    def get_reader_format(self, analog_channels, status_bytes):
        # Number of status fields of 2 bytes based on the total number of 
        # bytes.
        dcount = math.floor(status_bytes / 2)

        # Check the file configuration
        if int(status_bytes) > 0 and int(analog_channels) > 0:
            return self.STRUCT_FORMAT.format(acount=analog_channels,
                                             dcount=dcount)
        elif int(analog_channels) > 0:
            # Analog channels only.
            return self.STRUCT_FORMAT_ANALOG_ONLY.format(acount=analog_channels)
        else:
            # Status channels only.
            return self.STRUCT_FORMAT_STATUS_ONLY.format(acount=dcount)

    def parse(self, contents):
        """Parse DAT binary file contents."""
        time_mult = self._cfg.timemult
        time_base = self._cfg.time_base
        achannels = self._cfg.analog_count
        schannel = self._cfg.status_count

        # auxillary vectors (channels gains and offsets)
        a = [x.a for x in self._cfg.analog_channels]
        b = [x.b for x in self._cfg.analog_channels]

        dbytes = self.STATUS_BYTES * math.ceil(schannel / 16.0)
        groups_of_16bits = math.floor(dbytes / self.STATUS_BYTES)

        # Struct format.
        row_reader = struct.Struct(self.get_reader_format(achannels, dbytes))

        # Row reading function.
        if isinstance(contents, io.TextIOBase) or \
                isinstance(contents, io.BufferedIOBase):
            # Read all buffer contents
            contents = contents.read()

        for irow, values in enumerate(row_reader.iter_unpack(contents)):
            # Sample number
            n = values[0]
            # Time stamp
            ts_val = values[1]
            ts = self._get_time(n, ts_val, time_base, time_mult)

            if irow >= self.total_samples:
                break
            self.time[irow] = ts

            # Extract analog channel values.
            for ichannel in range(achannels):
                yint = self.filter_missing(values[ichannel + 2])
                y = a[ichannel] * yint + b[ichannel] if not math.isnan(yint) else float('nan')
                self.analog[ichannel][irow] = y

            # Extract status channel values.
            for igroup in range(groups_of_16bits):
                group = values[achannels + 2 + igroup]

                # for each group of 16 bits, extract the status channels
                maxchn = min([(igroup + 1) * 16, schannel])
                for ichannel in range(igroup * 16, maxchn):
                    chnindex = ichannel - igroup * 16
                    mask = int('0b01', 2) << chnindex
                    extract = (group & mask) >> chnindex

                    self.status[ichannel][irow] = extract

            # Get the next row
            irow += 1


class _Binary32DatReader(_BinaryDatReader):
    """32-bit binary format DatReader subclass."""

    def __init__(self, **kwargs):
        # Call the initialization for the inherited class
        super().__init__(**kwargs)
        self.ANALOG_BYTES = 4

        if struct.calcsize("L") == 4:
            self.STRUCT_FORMAT = "LL {acount:d}l {dcount:d}H"
            self.STRUCT_FORMAT_ANALOG_ONLY = "LL {acount:d}l"
        else:
            self.STRUCT_FORMAT = "II {acount:d}i {dcount:d}H"
            self.STRUCT_FORMAT_ANALOG_ONLY = "II {acount:d}i"

        # maximum negative value
        self.DATA_MISSING = -2147483648  # 0x80000000


class _Float32DatReader(_BinaryDatReader):
    """Single precision (float) binary format DatReader subclass."""

    def __init__(self, **kwargs):
        # Call the initialization for the inherited class
        super().__init__(**kwargs)
        self.ANALOG_BYTES = 4

        if struct.calcsize("L") == 4:
            self.STRUCT_FORMAT = "LL {acount:d}f {dcount:d}H"
            self.STRUCT_FORMAT_ANALOG_ONLY = "LL {acount:d}f"
        else:
            self.STRUCT_FORMAT = "II {acount:d}f {dcount:d}H"
            self.STRUCT_FORMAT_ANALOG_ONLY = "II {acount:d}f"

        # Maximum negative value
        self.DATA_MISSING = sys.float_info.min


def load(cfg_or_cff_file_path, dat_file_path=None, **kwargs) -> Comtrade:
    """Load and read comtrade files contents."""
    return Comtrade(**kwargs).load(cfg_or_cff_file_path, dat_file_path, **kwargs)


def load_as_dataframe(cfg_or_cff_file_path, dat_file_path=None, **kwargs) -> "pd.DataFrame":
    """Load and read comtrade files contents and returns a dataframe."""
    return Comtrade(**kwargs).load(cfg_or_cff_file_path,
                                   dat_file_path,
                                   use_numpy_arrays=True, **kwargs).to_dataframe(**kwargs)

