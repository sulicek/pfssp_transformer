# Models

The models were renamed several times, so the same model goes by a different name in
the code, in the saved evaluation results and in the paper.

- **Name in code** – model directory in `src/models/`.
- **Import function** – its loader in `repository.py`.
- **Name in evaluations** – key the result is stored under in the `src/evaluations/results/*.pkl` files.
- **Name in paper** – label used in the plots and tables produced by `src/graphs_repl.py`.

## Neural models

| Name in code | Import function | Name in evaluations | Name in paper | Notes |
| --- | --- | --- | --- | --- |
| `linear` | `linear_model` | `transformer linear model` | PFSNET<sub>m=20, BN</sub> | batch norm model |
| `linear` | `linear_model` | `transformer linear model no eval` | PFSNET<sub>m=20, AT</sub> | the same BN model and weights, run without `model.eval()` (`do_eval=False`) |
| `pfsnet20_ln` | `pfsnet20_ln_model` | `pfsnet20_ln` | PFSNET<sub>m=20, LN</sub> | layer norm model |
| `pfsnet20_ln` | `pfsnet20_ln_model` | `pfsnet20_ln no eval` | – | LN model run without `model.eval()`, ran for brevity |
| `lje` | `lje_model` | `transformer embed model` | PFSNET<sub>gen, BN</sub> | batch norm model |
| `lje` | `lje_model` | `transformer embed model no eval` | PFSNET<sub>gen, AT</sub> | the same BN model and weights, run without `model.eval()` (`do_eval=False`); its tours are the warm start in `warmstart.py` |
| `pfsnetgen_ln` | `pfsnetgen_ln_model` | `pfsnetgen_ln` | PFSNET<sub>gen, LN</sub> | layer norm model |
| `pfsnetgen_ln` | `pfsnetgen_ln_model` | `pfsnetgen_ln no eval` | – | LN model run without `model.eval()`, ran for brevity
