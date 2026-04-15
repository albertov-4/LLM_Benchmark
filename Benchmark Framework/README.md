# Benchmark Framework

Questa cartella e una proposta di struttura comune per confrontare piu LLM sugli
stessi task di planning, con protocolli e metriche coerenti.

Obiettivi:
- confrontare modelli diversi sulla stessa base
- separare task, protocollo, modello ed evaluation
- definire tre livelli di difficolta: `easy`, `medium`, `hard`
- salvare output e metriche in un formato comparabile

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

Come usare questa struttura:
1. definire o importare i task in `tasks/`
2. classificare le istanze in `easy`, `medium`, `hard`
3. registrare i modelli in `models/model_registry.yaml`
4. scegliere un protocollo in `protocols/`
5. lanciare la batteria di test con i runner
6. salvare gli output grezzi in `outputs/raw/`
7. salvare output parsati e metriche in `outputs/parsed/` e `outputs/scored/`

Cartelle chiave:
- `tasks/`: benchmark vero e proprio
- `protocols/`: definizione dei test
- `models/`: registry e adapter dei modelli
- `prompts/`: base comune dei prompt
- `evaluators/`: parser, validator, metriche, tassonomia errori
- `runner/`: orchestrazione delle campagne di test
- `analysis/`: notebook e report finali

Nota:
- in questa prima versione i manifest di `citycar` e `tetris` puntano ai dati gia
  presenti nella repo sotto `References/LLM-Needs-a-Plan-main/`
- la struttura e pensata per essere estesa anche ai domini presenti in
  `Planning Domains/`
