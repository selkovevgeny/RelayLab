from setuptools import setup

setup(
   name='relaylab',
   version='0.1.0',
   description='Библиотека предназначенная для простой работы с аварийными осциллограммами в формате COMTRADE в jupiter notebook',
   author='selkovevgeny',
   author_email='selkov.evgeny@yandex.ru',
   packages=['relaylab', 'relaylab.records'],  #same as name
   install_requires=['plotly', 'numpy', 'pandas'], #external packages as dependencies
)