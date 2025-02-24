# Import the reactivity coefficients from config file
import configparser
import pathlib

path = pathlib.Path(__file__).parent / "config" / "reactions.ini"
config = configparser.ConfigParser()
config.read(path)
