# Relay Lab

__Relay Lab__ - это библиотека, предназначенная для простой работы с аварийными осциллограммами в формате COMTRADE в jupiter notebook. Предназначана для исследовательских и учебных целей, когда программы просмотра осциллограмм не обеспечивают необходимую гибкость в работе с осциллограммами.

Библиотека позволяет:
- загружать осциллограммы в формате comtrade;
- генерировать сигналы тока и напряжения;
- выполнять манипуляции с аналоговыми и логическими сигналами: сложение, вычитание, расчет диф. токов, сопротивлений, сравнение с уставками и пр.
- строить интерактивные графики сигналов и характеристики срабатывания защит.

В составе библиотеки находится набор осциллограмм, типичных для энергосистемы.

## Предварительный просмотр примеров
Ниже даны ссылки для предварительного просмотра примеров on-line:
1. Работа с осциллограммами, анализ максимальной токовой защиты [-->](https://nbviewer.org/github/selkovevgeny/RelayLab/blob/master/notebooks/1.%20%D0%A0%D0%B0%D0%B1%D0%BE%D1%82%D0%B0%20%D1%81%20%D0%BE%D1%81%D1%86%D0%B8%D0%BB%D0%BB%D0%BE%D0%B3%D1%80%D0%B0%D0%BC%D0%BC%D0%B0%D0%BC%D0%B8%2C%20%D0%B0%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7%20%D0%BC%D0%B0%D0%BA%D1%81%D0%B8%D0%BC%D0%B0%D0%BB%D1%8C%D0%BD%D0%BE%D0%B9%20%D1%82%D0%BE%D0%BA%D0%BE%D0%B2%D0%BE%D0%B9%20%D0%B7%D0%B0%D1%89%D0%B8%D1%82%D1%8B.ipynb)
2. Создание сигналов [-->](https://nbviewer.org/github/selkovevgeny/RelayLab/blob/master/notebooks/2.%20%D0%A1%D0%BE%D0%B7%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5%20%D1%81%D0%B8%D0%B3%D0%BD%D0%B0%D0%BB%D0%BE%D0%B2.ipynb)
3. Действующее значение [-->](https://github.com/selkovevgeny/RelayLab/blob/master/notebooks/3.%20%D0%94%D0%B5%D0%B9%D1%81%D1%82%D0%B2%D1%83%D1%8E%D1%89%D0%B5%D0%B5%20%D0%B7%D0%BD%D0%B0%D1%87%D0%B5%D0%BD%D0%B8%D0%B5.ipynb)
4. Векторная диаграмма [-->](https://nbviewer.org/github/selkovevgeny/RelayLab/blob/master/notebooks/4.%20%D0%92%D0%B5%D0%BA%D1%82%D0%BE%D1%80%D0%BD%D0%B0%D1%8F%20%D0%B4%D0%B8%D0%B0%D0%B3%D1%80%D0%B0%D0%BC%D0%BC%D0%B0.ipynb)
5. Насыщение ТТ [-->](https://nbviewer.org/github/selkovevgeny/RelayLab/blob/master/notebooks/5.%20%D0%9D%D0%B0%D1%81%D1%8B%D1%89%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%A2%D0%A2.ipynb)
6. Анализ работы диф. защиты [-->](https://nbviewer.org/github/selkovevgeny/RelayLab/blob/master/notebooks/6.%20%D0%90%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%8B%20%D0%B4%D0%B8%D1%84.%20%D0%B7%D0%B0%D1%89%D0%B8%D1%82%D1%8B.ipynb)

В примерах, в которых применяется элемент widget, обеспечивающий интерактивность, графики при предварительном просмотре не отображаются. Для полноценного взаимодействия необходимо установить программу Anaconda c Jupiter Notebook, согласно инструкции ниже, или открыть jupiter notebook online в предварительном просмотре (кнопка Execute on Binder в правом верхнем углу).

## Требования к программному обеспечению

- [Anaconda](https://www.anaconda.com/download) преставляет собой дистрибутив, включающий набор популярных библиотек для анализа данных. Мы будем пользоваться программой jupiter notebook.
- plotly 5.18.0 или более новая.

## Установка

1. Загружаем дистрибутив Anaconda
2. Создаем рабочую папку. В нее помещаем папку 'notebooks', которую загружаем из репозитория. В данной папке находятся примеры файлов jupiter notebook в которых выполнен анализ типичных осциллограмм.
3. Запускаем jupiter notebook, который установился при установке Anaconda. В интерфейсе jupiter notebook выбираем рабочую папку. Открываем или создаем новый файл.
4. Загружаем библиотеку relaylab:
```
pip install https://github.com/selkovevgeny/RelayLab/archive/master.zip
```
Допустимо копировать папку `relaylab` из  репозитория в рабочую папку.  
5. Подключаем функцию подсказки и дополнения кода. Ссылка на инструкцию: https://russianblogs.com/article/40391171020/

## Как использовать

Пример загрузки осциллограммы:
```
from relay_lab import equipment as eqp, signals as sg, record, characteristics, functions as func
from relay_lab.signals import color
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

rec = record.examples.transformer_short_circuit_out_zone
print(rec)
```
Больше примеров смотрите в папке 'notebooks', которую можно загрузить из репозитория.

## Поддержка

При обнаружении ошибок или наличии предложений по развитию библиотеки обращайтесь в профиль ВК или пишите на email.


## Лицензия

Библиотека доступна на [GitHub](https://github.com/selkovevgeny/RelayLab) под лицензией MIT.

