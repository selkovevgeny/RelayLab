from setuptools import setup

setup(
   name='relaylab',
   version='0.1.2',
   description='Библиотека предназначенная для простой работы с аварийными осциллограммами в формате COMTRADE в jupiter notebook',
   author='selkovevgeny',
   author_email='selkov.evgeny@yandex.ru',
   packages=['relaylab'],  #same as name
   include_package_data=True,
   install_requires=['plotly', 'numpy', 'pandas'], #external packages as dependencies
)