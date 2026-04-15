# Runner

Questa cartella contiene gli entry point logici per eseguire il benchmark.

File previsti:
- `run_case.py`: esecuzione di un singolo task con un singolo modello e protocollo
- `run_suite.py`: esecuzione di una suite completa
- `batch_matrix.py`: costruzione della matrice modello x protocollo x difficolta

Responsabilita del runner:
- leggere il task index
- caricare protocollo e modello
- invocare adapter, parser e validator
- salvare raw output, parsed output e score
