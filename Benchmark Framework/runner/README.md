# Runner

Questa cartella contiene gli entry point logici per eseguire il benchmark.

File previsti:
- `run_case.py`: esecuzione di un singolo task con un singolo modello e protocollo
- `run_suite.py`: esecuzione di una suite completa
- `batch_matrix.py`: costruzione della matrice modello x protocollo x difficolta

Responsabilita del runner:
- scoprire i task dalla struttura delle cartelle
- caricare protocollo e modello
- caricare prompt di sistema, dominio, esempi e feedback in base al protocollo
- invocare adapter, parser e validator
- salvare raw output, parsed output e score

Struttura generale:
- `run_case.py` rappresenta l'unita minima del benchmark: un modello, un protocollo, un task
- `run_suite.py` costruisce e orchestra una campagna completa a partire da task, modelli e protocolli disponibili
- `batch_matrix.py` e il punto naturale in cui esplicitare o filtrare la matrice dei job prima dell'esecuzione

Struttura di `run_suite.py`:
- scopre task, protocolli e modelli a partire dalla struttura del framework
- trasforma i task scoperti in `TaskSpec` e i protocolli caricati in `ProtocolSpec`
- carica il bundle di prompt da `prompts/` in base alla famiglia del task e ai flag del protocollo
- costruisce un job per ogni combinazione modello x protocollo x task
- per ogni job crea adapter e validator e delega l'esecuzione concreta a `run_case.py`
- raccoglie i `ResultRecord` normalizzati e produce un'aggregazione finale della suite
- puo usare sia validator mock sia un validator reale `VAL`, senza cambiare la logica di orchestrazione

Scoperta dei task:
- i task non vengono letti da un indice centrale
- la fonte di verita e la gerarchia del benchmark:
  - `tasks/<task_family>/domain/domain.pddl`
  - `tasks/<task_family>/easy/*.pddl`
  - `tasks/<task_family>/medium/*.pddl`
  - `tasks/<task_family>/hard/*.pddl`

Struttura di `ResultRecord`:
- `ResultRecord` descrive l'esito completo di un singolo run
- contiene l'identita del run, l'esito globale, l'output prodotto dal modello e i riferimenti agli artefatti salvati

Campi principali di `ResultRecord`:
- `model_id`: modello usato nel run
- `task_id`: identificatore stabile del task concreto, costruito da famiglia, difficolta e istanza
- `protocol_id`: protocollo usato per interrogare il modello
- `task_family`, `tier`, `instance_id`: componenti del task utili per filtrare, raggruppare e analizzare i risultati
- `solved`: indica se il task e stato risolto con almeno un piano valido
- `iterations_used`: numero di tentativi effettivamente eseguiti
- `max_iterations`: massimo numero di tentativi consentiti dal protocollo
- `stopped_by_iteration_limit`: indica se il run si e fermato per esaurimento del budget di iterazioni
- `raw_output`: ultimo output grezzo prodotto dal modello
- `parsed_plan`: versione strutturata del piano estratta dall'output, se disponibile
- `validation_result`: esito finale della validazione del piano
- `metrics`: metriche derivate dal run, pronte per confronto e analisi
- `raw_output_path`, `parsed_output_path`, `scored_output_path`: percorsi degli artefatti eventualmente salvati su disco

Separazione delle responsabilita:
- `TaskSpec` descrive cosa bisogna risolvere
- `ProtocolSpec` descrive come il modello deve essere interrogato
- `ResultRecord` descrive cosa e successo alla fine del run

Flusso di un run:
- `run_case.py` carica dominio e problema dal task selezionato
- costruisce i messaggi da inviare al modello in base al protocollo, al prompt di sistema e al prompt della famiglia di task
- passa l'output grezzo del modello al parser condiviso
- se il parser non trova un piano valido, il run produce un errore di parsing e genera feedback per il tentativo successivo
- se il parser estrae un piano, il validator controlla il piano sul dominio e sull'istanza
- se la validazione fallisce, il runner costruisce feedback di repair e avvia una nuova iterazione fino al limite previsto dal protocollo
- al termine del loop, `metrics.py` trasforma il risultato normalizzato del run in metriche confrontabili
- se riceve un `output_root`, salva tre artefatti per-job:
  - raw output del modello
  - piano parsato e validazione
  - risultato scored finale con metriche e path degli artefatti

Flusso di una suite:
- `run_suite.py` parte dalla discovery dei task e costruisce la matrice completa dei job
- durante l'esecuzione stampa a terminale una riga `START`, `DONE` o `ERROR` per ogni job
- per ogni job carica il protocollo richiesto e recupera la configurazione del modello dal registry
- nella CLI `run_benchmark.py`, `--adapter` seleziona automaticamente il registry coerente
- internamente `run_suite.py` mantiene anche `adapter_override` per test e usi avanzati
- se passi `--model-id`, viene eseguito solo quel modello usando l'adapter dichiarato nel YAML
- se passi `--protocol-id`, viene eseguito solo quel protocollo
- se il protocollo richiede esempi o feedback esterno, li carica dai file nella cartella `prompts/`
- se non riceve un adapter factory reale, usa uno scaffold minimo per mantenere il flusso eseguibile
- se riceve `use_real_validator=True`, costruisce automaticamente un `VALValidatorAdapter`
- se non riceve un validator reale, usa un validator di fallback che segnala l'assenza del componente
- dopo l'esecuzione dei singoli run, aggrega i risultati per modello, protocollo e livello di difficolta

Ordine di selezione del validator:
- se passi `validator_factory`, `run_suite.py` usa quella
- altrimenti, se passi `validator`, usa l'istanza fornita
- altrimenti, se passi `use_real_validator=True`, prova a costruire un validator reale `VAL`
- se nessuna di queste opzioni e disponibile, usa `_UnavailableValidator`
