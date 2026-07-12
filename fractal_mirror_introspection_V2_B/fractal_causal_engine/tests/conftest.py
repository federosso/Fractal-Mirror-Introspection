# I test del motore importano `fractal_causal_engine.*` in assoluto: il package
# vive nella radice del progetto come sorgente semplice (nessuna installazione),
# quindi la radice va messa sul sys.path.
import sys
import pathlib
RADICE_PROGETTO = pathlib.Path(__file__).resolve().parents[2]
if str(RADICE_PROGETTO) not in sys.path:
    sys.path.insert(0, str(RADICE_PROGETTO))
