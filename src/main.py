import importlib

import typer
from rich import print
from rich.console import Group
from rich.panel import Panel

app = typer.Typer(pretty_exceptions_enable=False)


@app.command(no_args_is_help=True)
def eval(evaluation_name: str):
    """
    Run the specified evaluation.
    """
    importlib.import_module(f"evaluations.{evaluation_name}")


@app.command()
def list_eval():
    """
    List all available evaluation scripts.
    """
    import os

    eval_res_dir = "evaluations/"
    res = os.listdir(eval_res_dir)
    text = ""
    for file in res:
        if not (os.path.isfile(os.path.join(eval_res_dir, file))):
            continue
        found = file[: file.find(".py")]
        if found == "__init__" or found.startswith("_"):
            continue
        text += f"{found}\n"
    panel = Panel.fit(text, title="Evaluations to run")
    print(panel)


if __name__ == "__main__":
    app()
