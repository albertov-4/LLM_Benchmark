# Evaluators

Questa cartella contiene i componenti comuni di valutazione.

Componenti:
- `parser.py`: trasforma output grezzo in piano strutturato
- `validator.py`: interfaccia comune al validator esterno e struttura di `ValidationResult`
- `metrics.py`: metriche aggregate e per-run
- `error_taxonomy.py`: insieme controllato di tipi di errore condivisi tra validator, parser e scoring

Principio guida:
- tutti i modelli devono passare dallo stesso parser e dallo stesso validator
- le differenze osservate devono derivare dal modello, non da pipeline diverse

Struttura del parser:
- `parser.py` converte il testo grezzo del modello in un `ParsedPlan` con `actions`, `reasoning` e `format_issues`
- il parser non decide se un piano e corretto: estrae solo la sequenza candidata da dare al validator
- il parser e pensato per output reali di LLM, quindi gestisce in modo comune:
  - reasoning prima del piano
  - markdown fences
  - liste numerate o puntate
  - azioni PDDL inserite dentro righe di testo piu verbose

Flusso del parser:
- se l'output e vuoto, restituisce `empty_output`
- se trova un marcatore come `Plan:` o `Final plan:`, separa il reasoning dalla parte finale orientata al piano
- rimuove eventuali code fence markdown senza perdere il contenuto interno
- estrae azioni in forma `(...)` anche se compaiono dentro liste o testo misto
- normalizza gli spazi interni delle azioni
- registra in `format_issues` le anomalie incontrate, per esempio `markdown_fences_removed`, `reasoning_before_plan_removed`, `actions_embedded_in_text` o `no_parenthesized_actions_found`

Struttura del validator:
- `validator.py` restituisce un `ValidationResult`, cioe un oggetto normalizzato che descrive l'esito della validazione di un singolo piano
- il validator valuta un singolo tentativo, non l'intero loop iterativo di repair
- le informazioni sul numero di iterazioni usate da un modello appartengono al risultato complessivo del run, non al validator
- il risultato della validazione separa sempre:
  - lo stato generale dell'esito, tramite `status`
  - il motivo specifico del fallimento, tramite `error_type`

Componenti del validator:
- `ValidatorAdapter` definisce l'interfaccia comune che il runner si aspetta
- `VALValidatorConfig` descrive come invocare un validator esterno reale
- `VALValidatorAdapter` incapsula il flusso reale di validazione: scrittura del piano su file temporaneo, chiamata a subprocess, gestione di timeout e crash, normalizzazione dell'output in `ValidationResult`
- `build_feedback_from_validation(...)` trasforma il risultato della validazione in un messaggio sintetico riusabile nel repair loop

Campi principali di `ValidationResult`:
- `valid`: esito booleano della validazione
- `status`: uno tra `valid`, `invalid`, `parse_error`, `timeout`, `validator_error`
- `error_type`: categoria piu specifica del fallimento, se presente
- `feedback_text`: messaggio breve riusabile in un eventuale loop di repair
- `failed_step` e `failed_action`: punto del piano in cui la validazione fallisce, se disponibile
- `goal_satisfied`: distingue tra piano eseguibile ma goal non raggiunto e piano che fallisce prima
- `plan_length`, `validation_time_ms`, `raw_validator_output`, `details`: supporto ad analisi, debugging e audit

Flusso del validator reale:
- il piano estratto dal parser viene scritto in un file temporaneo
- il validator esterno viene invocato con dominio, problema e piano
- stdout, stderr e return code vengono raccolti e interpretati
- errori tecnici come timeout, binario mancante o crash vengono trasformati in `ValidationResult` coerenti con la tassonomia del benchmark
- l'output del validator viene tradotto in una prima mappatura euristica verso `valid`, `invalid_precondition`, `unsatisfied_goal`, `unknown_action` o altri errori condivisi

Tassonomia degli errori:
- `error_taxonomy.py` definisce un vocabolario controllato di errori, usato per mantenere confrontabili i risultati tra modelli e task diversi
- la tassonomia distingue tra errori logici del piano ed errori tecnici della pipeline
- errori logici del piano: `empty_plan`, `syntax_error`, `unknown_action`, `invalid_precondition`, `unsatisfied_goal`
- errori tecnici della pipeline: `parse_error`, `timeout`, `validator_crash`, `validator_unavailable`
- fallback: `unknown`

Struttura delle metriche:
- `metrics.py` trasforma il risultato completo di un run in un insieme di misure confrontabili tra modelli
- le metriche non devono dipendere direttamente dal testo grezzo prodotto dal modello, ma dal risultato normalizzato del run
- questo livello serve a separare i dati grezzi dalle misure gia pronte per analisi, tabelle e confronti

Metriche core:
- `validity_at_1`: il task e stato risolto al primo tentativo
- `validity_at_k`: il task e stato risolto entro il massimo numero di iterazioni consentito
- `repair_success`: il task e stato risolto, ma non al primo tentativo
- `iterations_to_valid`: numero di iterazioni necessarie per arrivare a un piano valido
- `plan_length`: lunghezza del piano finale, ricavata dal parser o dal validator
- `error_type`: tipo di errore finale, se il run non termina con successo
- `hit_iteration_limit`: indica se il run ha consumato o esaurito il budget di iterazioni disponibile
