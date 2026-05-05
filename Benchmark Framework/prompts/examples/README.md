# Prompt Examples

Questa cartella contiene esempi opzionali caricati dai protocolli che usano
`include_examples: true`.

Convenzione:
- `examples/<task_family>.txt`
- il contenuto viene aggiunto dopo `prompts/<task_family>.txt`
- gli esempi devono chiarire formato e vincoli, non risolvere le istanze del benchmark
- se un file di esempi manca, il benchmark continua senza esempi

Esempio:

```text
prompts/examples/farmland.txt
```

Nota: il prompt dominio-specifico principale resta obbligatorio quando il
protocollo ha `include_domain_prompt: true`. Gli esempi invece sono opzionali.
