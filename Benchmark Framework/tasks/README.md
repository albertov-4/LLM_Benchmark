# Tasks

Questa cartella contiene la definizione del benchmark a livello di task.

Ogni famiglia di task dovrebbe avere:
- una cartella dedicata, per esempio `<task_family>/`
- una sottocartella `domain/` con `domain.pddl`
- tre sottocartelle: `easy`, `medium`, `hard`
- istanze `.pddl` direttamente dentro i tier di difficolta

Convenzione consigliata:

```text
tasks/
`-- <task_family>/
    |-- domain/
    |   `-- domain.pddl
    |-- easy/
    |-- medium/
    `-- hard/
```

Regola pratica:
- la difficolta non va decisa solo "a occhio", ma spiegata nel `README.md` della famiglia
- la scoperta dei task puo essere fatta direttamente dalla struttura delle cartelle

Nota:
- una cartella `metadata/` puo esistere per csv o indici futuri, ma e opzionale
- le famiglie concrete del benchmark vengono scoperte direttamente dalla
  struttura delle cartelle
