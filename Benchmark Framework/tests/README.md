# Tests

Questa cartella contiene il supporto minimo per verificare il framework senza
dipendere da modelli o validator reali.

Struttura:
- `mocks/`: adapter e validator finti ma controllabili
- `fixtures/`: piccoli task e file di configurazione usati nei test
- `test_hf_local.py`: test dell'adapter locale HF senza dipendere da modelli reali
- `test_parser.py`: test mirati del parser condiviso
- `test_run_case.py`: smoke test della pipeline di un singolo run
- `test_run_suite.py`: smoke test dell'orchestrazione della suite
- `test_real_validator.py`: integrazione del validator reale `VAL` sul task toy
- `run_mock_suite.py`: entry point manuale per lanciare la suite con i mock

Obiettivo:
- provare il flusso `generate -> parse -> validate -> metrics`
- verificare che `run_suite.py` riesca a orchestrare piu componenti insieme
- mantenere i test indipendenti da GPU, API esterne e tool di validazione reali

Comandi utili:
- `python "Benchmark Framework/tests/test_hf_local.py"`
- `python "Benchmark Framework/tests/test_parser.py"`
- `python "Benchmark Framework/tests/test_run_case.py"`
- `python "Benchmark Framework/tests/test_run_suite.py"`
- `python "Benchmark Framework/tests/test_real_validator.py"`
- `python "Benchmark Framework/tests/run_mock_suite.py"`
- `test_run_case.py` verifica anche la persistenza di `raw/parsed/scored` su una cartella temporanea
- `python -m unittest discover -s "Benchmark Framework/tests" -p "test_*.py"`
