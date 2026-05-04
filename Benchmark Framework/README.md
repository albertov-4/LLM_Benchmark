# Benchmark Framework

Questa cartella e una proposta di struttura comune per confrontare piu LLM sugli
stessi task di planning, con protocolli e metriche coerenti.

Obiettivi:
- confrontare modelli diversi sulla stessa base
- separare task, protocollo, modello ed evaluation
- definire tre livelli di difficolta: `easy`, `medium`, `hard`
- salvare output e metriche in un formato comparabile
- partire da zero, senza dipendere dalle repo dentro `References/`
- poter lanciare test reali con adapter NVIDIA API, Hugging Face locale, Ollama, llama.cpp e validator `VAL`

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
4. registrare i modelli nei file `models/model_registry_*.yaml`
5. scegliere un protocollo in `protocols/`
6. lanciare la batteria di test con `run_benchmark.py`
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
- `citycar` e `tetris` sono ora famiglie starter reali, con dominio e istanze
  `easy/medium/hard` gia pronte per i primi test locali
- eventuali manifest o file indice possono essere aggiunti in futuro, ma non
  sono necessari per far funzionare il benchmark

Entry point consigliato:
- `python "Benchmark Framework/run_benchmark.py" --use-real-validator`
- default: usa `models/model_registry_nvidia.yaml` e tutti i modelli con `enabled: true`
- per scegliere il backend via registry: `python "Benchmark Framework/run_benchmark.py" --adapter hf_local --protocol-id direct_plan --use-real-validator`
- per un solo modello: `python "Benchmark Framework/run_benchmark.py" --model-id nvidia_gemma_4_31b_it --use-real-validator`
- per un solo protocollo: `python "Benchmark Framework/run_benchmark.py" --protocol-id direct_plan --use-real-validator`
- per un registry diverso: `python "Benchmark Framework/run_benchmark.py" --model-registry-path "models/model_registry_ollama.yaml" --model-id ollama_phi_4_mini_instruct`
- il launcher salva un riepilogo JSON in `outputs/scored/suite_result_latest.json`
- durante l'esecuzione stampa una riga `START`, `DONE` o `ERROR` per ogni job
- inoltre salva per ogni job:
  - `outputs/raw/<model_id>/<protocol_id>/<task_family>/<tier>/<instance_id>.json`
  - `outputs/parsed/<model_id>/<protocol_id>/<task_family>/<tier>/<instance_id>.json`
  - `outputs/scored/<model_id>/<protocol_id>/<task_family>/<tier>/<instance_id>.json`

Setup rapido:
- dipendenze Python in [requirements.txt](requirements.txt)
- istruzioni ambiente e validator in [SETUP.md](SETUP.md)
