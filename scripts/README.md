# How to rebuild opcode.ml and instr.ml

```sh
cd ..
python3 scripts/spec2ml.py --opcode --replace
python3 scripts/spec2ml.py --instr --replace
dune build
```
