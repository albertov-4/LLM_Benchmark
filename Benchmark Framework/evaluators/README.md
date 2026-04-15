# Evaluators

Questa cartella contiene i componenti comuni di valutazione.

Componenti:
- `parser.py`: trasforma output grezzo in piano strutturato
- `validator.py`: interfaccia comune al validator esterno
- `metrics.py`: metriche aggregate e per-run
- `error_taxonomy.py`: insieme controllato di tipi di errore

Principio guida:
- tutti i modelli devono passare dallo stesso parser e dallo stesso validator
- le differenze osservate devono derivare dal modello, non da pipeline diverse
