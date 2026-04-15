# Tasks

Questa cartella contiene la definizione del benchmark a livello di task.

Ogni famiglia di task dovrebbe avere:
- un `manifest.yaml`
- tre sottocartelle: `easy`, `medium`, `hard`
- un riferimento chiaro al file di dominio
- un mapping esplicito delle istanze incluse nel benchmark

File importante:
- `metadata/task_index.csv`: tabella piatta utile per runner e analisi

Regola pratica:
- la difficolta non va decisa solo "a occhio", ma documentata nel manifest
- ogni istanza dovrebbe avere almeno un tag o una motivazione sintetica
