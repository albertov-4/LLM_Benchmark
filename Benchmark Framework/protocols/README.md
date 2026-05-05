# Protocols

Questa cartella definisce i protocolli sperimentali usati dal benchmark.

Un protocollo descrive **come** interrogare il modello, non **quale** task
risolvere e non **quale** modello usare. La matrice finale viene costruita da:

```text
modelli selezionati x protocolli selezionati x task scoperti
```

Per lanciare un solo protocollo:

```powershell
python "Benchmark Framework/run_benchmark.py" --protocol-id direct_plan --use-real-validator
```

Per combinare protocollo e filtro task:

```powershell
python "Benchmark Framework/run_benchmark.py" --protocol-id iterative_repair --task-family <task_family> --tier <tier> --use-real-validator
```

## Protocolli Disponibili

`direct_plan.yaml`

Il modello riceve dominio, problema e istruzioni di formato. Deve produrre
direttamente il piano finale, senza reasoning esplicito e senza repair.

Serve a misurare:
- validita al primo tentativo
- rispetto del formato richiesto
- capacita di planning senza aiuti esterni

`direct_plan_with_rationale.yaml`

Il modello puo produrre reasoning testuale, ma deve comunque rendere estraibile
un piano finale. Il parser cerca le azioni PDDL dentro l'output del modello.

Serve a misurare:
- se il reasoning migliora la qualita del piano
- se il reasoning resta coerente con le azioni finali
- quanto rumore testuale introduce rispetto al formato plan-only

`iterative_repair.yaml`

Il modello genera un piano, il validator lo controlla e, se fallisce, il runner
costruisce un feedback da reinserire nel tentativo successivo.

Serve a misurare:
- se il modello sa correggersi dopo feedback esterno
- quante iterazioni servono per arrivare a un piano valido
- quali errori persistono anche dopo repair

## Campi YAML

`protocol_id`

Identificatore stabile del protocollo. Deve corrispondere al nome logico usato
nel runner e nei risultati.

Esempio:

```yaml
protocol_id: iterative_repair
```

`description`

Descrizione leggibile del protocollo. Non influenza direttamente l'esecuzione,
ma serve a documentare l'obiettivo sperimentale.

`prompting`

Controlla quali parti vengono incluse nel prompt.

Campi principali:
- `use_system_prompt`: include `prompts/system.txt`
- `include_domain_prompt`: include il prompt specifico della famiglia di task, per esempio `prompts/farmland.txt`
- se `include_domain_prompt: true`, il file `prompts/<task_family>.txt` e obbligatorio; il framework non usa `prompts/default.txt` come fallback automatico
- `include_examples`: include esempi se disponibili
- `include_chain_of_thought`: aggiunge istruzioni di rationale quando il protocollo non e plan-only; se il protocollo e plan-only, il modello puo ragionare internamente ma deve restituire solo azioni
- `include_external_feedback`: abilita feedback del validator nei tentativi successivi

`generation`

Descrive le impostazioni di generazione da passare agli adapter quando supportate.

Campi principali:
- `mode`: etichetta descrittiva, per esempio `deterministic`, `semi_deterministic`, `repair_loop`
- `temperature`: controlla quanto il modello genera in modo variabile
- `top_k`: limita il sampling ai token candidati principali, se supportato
- `max_tokens`: massimo numero di token generabili

Nota: non tutti gli adapter usano tutti i campi nello stesso modo. Il runner
passa i parametri comuni agli adapter quando sono supportati.

`evaluation`

Controlla il loop di valutazione.

Campi principali:
- `max_iterations`: massimo numero di tentativi per task
- `require_final_plan_only`: se `true`, il prompt richiede solo azioni PDDL nel formato finale

In `direct_plan` e `direct_plan_with_rationale`, `max_iterations` e normalmente
`1`. In `iterative_repair`, `max_iterations` puo essere maggiore di `1`.

`primary_questions`

Lista di domande sperimentali associate al protocollo. Serve per ricordare cosa
si vuole misurare quando si analizzano i risultati.

## Iterative Repair In Pratica

Il loop di repair funziona cosi:

1. Il runner costruisce il prompt iniziale con dominio, problema e istruzioni.
2. Il modello genera un piano candidato.
3. Il parser estrae azioni PDDL dal testo generato.
4. Se il parser non trova azioni, viene creato un errore di parsing.
5. Se ci sono azioni, `VAL` valida il piano sul dominio e sul problema.
6. Se il piano e valido, il run termina con `solved: true`.
7. Se il piano non e valido, il runner crea un feedback sintetico.
8. Il feedback viene aggiunto al prompt successivo.
9. Il ciclo continua fino a piano valido o `max_iterations`.

Il feedback testuale di base vive in:

```text
Benchmark Framework/prompts/feedback.txt
```

## Output E Analisi

Nei risultati finali trovi:
- `solved`: se il task e stato risolto
- `iterations_used`: quanti tentativi sono stati usati
- `max_iterations`: budget massimo del protocollo
- `stopped_by_iteration_limit`: se il run si e fermato per limite di iterazioni
- `validation_result`: esito finale della validazione
- `metrics`: metriche derivate, per esempio `repair_success` e `iterations_to_valid`

Gli output per-job sono separati per livello:
- `raw`: contiene `messages`, payload `generation`, `raw_output` e `raw_generations`
- `parsed`: contiene `parsed_plan` per ogni tentativo
- `scored`: contiene `validation_result`, `feedback_to_next_iteration`, metriche finali e path degli artefatti

Il campo `attempts` esiste in tutti e tre i livelli, ma con contenuto diverso.
Questo evita di duplicare tutte le informazioni in ogni file.
