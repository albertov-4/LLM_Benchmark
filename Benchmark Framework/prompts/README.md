# Prompts

Questa cartella contiene i testi che il benchmark usa per costruire i messaggi
inviati ai modelli.

I prompt non scelgono il modello, il protocollo o il task. Servono solo a
definire **come** il task viene presentato alla LLM.

## Struttura

```text
prompts/
|-- system.txt
|-- farmland.txt
|-- feedback.txt
|-- examples/
    |-- farmland.txt
```

## `system.txt`

Prompt globale del benchmark.

Viene inserito come messaggio `system` quando il protocollo ha:

```yaml
prompting:
  use_system_prompt: true
```

Serve a fissare regole comuni per tutti i domini:

- usare solo azioni definite nel dominio
- usare solo oggetti presenti nel problema
- non inventare predicati, oggetti o fluenti numerici
- produrre azioni PDDL groundate
- rispettare precondizioni, effetti e goal finale

Questo file deve restare generale. Non deve contenere dettagli specifici di un
singolo dominio come `farmland`.

## `<task_family>.txt`

Prompt specifico del dominio.

Esempio:

```text
prompts/farmland.txt
```

Viene caricato quando il protocollo ha:

```yaml
prompting:
  include_domain_prompt: true
```

Il file e obbligatorio. Se manca `prompts/<task_family>.txt`, il benchmark si
ferma con un errore esplicito.

Non esiste fallback automatico a `default.txt`. Questa scelta evita benchmark
ambigui in cui un dominio viene testato con un prompt generico invece che con
istruzioni dominio-specifiche.

Il prompt del dominio dovrebbe spiegare:

- significato generale del dominio
- vincoli importanti
- azioni disponibili
- errori tipici da evitare
- formato atteso, se ci sono particolarita del dominio

## `examples/<task_family>.txt`

Esempi opzionali per una specifica famiglia di task.

Esempio:

```text
prompts/examples/farmland.txt
```

Viene caricato solo se il protocollo ha:

```yaml
prompting:
  include_examples: true
```

Gli esempi servono a chiarire il formato delle azioni e gli errori da evitare.
Non devono essere soluzioni delle istanze del benchmark.

## `feedback.txt`

Prompt usato nel protocollo di repair.

Viene inserito nel messaggio di feedback quando il protocollo ha:

```yaml
prompting:
  include_external_feedback: true
```

Il runner combina questo testo con l'errore prodotto dal validator. Il modello
riceve quindi:

- regole generali di repair
- stato della validazione
- tipo di errore
- eventuale step fallito
- eventuale azione fallita
- messaggio del validator

Il punto importante e che il modello deve restituire un piano completo corretto,
non solo una patch o la singola azione fallita.

## Flow Dei Protocolli

`direct_plan`

```text
system.txt
+ <task_family>.txt
+ dominio PDDL
+ problema PDDL
```

Il modello deve produrre direttamente il piano finale.

`direct_plan_with_rationale`

```text
system.txt
+ <task_family>.txt
+ examples/<task_family>.txt
+ dominio PDDL
+ problema PDDL
```

Il modello puo produrre un breve rationale se il protocollo non richiede output
plan-only. Il piano deve comunque essere estraibile come sequenza di azioni
PDDL.

`iterative_repair`

Primo tentativo:

```text
system.txt
+ <task_family>.txt
+ examples/<task_family>.txt
+ dominio PDDL
+ problema PDDL
```

Tentativi successivi:

```text
messaggi precedenti
+ feedback.txt
+ feedback del validator
```

Ogni iterazione viene salvata negli output dentro il campo `attempts`, inclusi
i messaggi passati al modello.

## Quando Aggiungi Un Nuovo Dominio

Per aggiungere un nuovo task family, per esempio `logistics`, crea almeno:

```text
tasks/logistics/domain/domain.pddl
tasks/logistics/easy/<istanza>.pddl
prompts/logistics.txt
```

Se vuoi usare esempi nel protocollo, aggiungi anche:

```text
prompts/examples/logistics.txt
```

Senza `prompts/logistics.txt`, ogni protocollo con `include_domain_prompt: true`
fallira prima di interrogare il modello.
