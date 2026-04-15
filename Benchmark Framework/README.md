# Benchmark Framework

Questa cartella e una proposta di struttura comune per confrontare piu LLM sugli
stessi task di planning, con protocolli e metriche coerenti.

Obiettivi:
- confrontare modelli diversi sulla stessa base
- separare task, protocollo, modello ed evaluation
- definire tre livelli di difficolta: `easy`, `medium`, `hard`
- salvare output e metriche in un formato comparabile
- partire da zero, senza dipendere dalle repo dentro `References/`

Struttura:

```text
Benchmark Framework/
|-- tasks/
|-- protocols/
|-- models/
|-- prompts/
|-- runner/
|-- evaluators/
|-- outputs/
|-- analysis/
`-- config/
```

Principi di progettazione:
- i task sono indipendenti dal modello
- il protocollo di prompting e indipendente dal task
- ogni modello usa un adapter con interfaccia comune
- parsing, validation e metriche sono unici per tutti
- i risultati grezzi e quelli valutati sono separati
- la gerarchia delle cartelle e la convenzione di naming sono la fonte di verita
- i manifest non sono obbligatori

Struttura consigliata per ogni famiglia di task:

```text
tasks/
`-- <task_family>/
    |-- README.md
    |-- domain/
    |   `-- domain.pddl
    |-- easy/
    |   |-- instance-01.pddl
    |   `-- instance-02.pddl
    |-- medium/
    |   `-- ...
    `-- hard/
        `-- ...
```

Come usare questa struttura:
1. creare da zero una famiglia di task dentro `tasks/`
2. aggiungere `domain/domain.pddl`
3. mettere le istanze `.pddl` nelle cartelle `easy`, `medium`, `hard`
4. registrare i modelli in `models/model_registry.yaml`
5. scegliere un protocollo in `protocols/`
6. lanciare la batteria di test con i runner
7. salvare gli output grezzi in `outputs/raw/`
8. salvare output parsati e metriche in `outputs/parsed/` e `outputs/scored/`

Cartelle chiave:
- `tasks/`: benchmark vero e proprio
- `protocols/`: definizione dei test
- `models/`: registry e adapter dei modelli
- `prompts/`: base comune dei prompt
- `evaluators/`: parser, validator, metriche, tassonomia errori
- `runner/`: orchestrazione delle campagne di test
- `analysis/`: notebook e report finali

Nota:
- `citycar` e `tetris` sono lasciati come esempi di famiglie di task, ma ora
  la struttura e pensata per essere popolata ex novo
- eventuali manifest o file indice possono essere aggiunti in futuro, ma non
  sono necessari per far funzionare il benchmark
