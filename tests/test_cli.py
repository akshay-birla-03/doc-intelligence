"""Tests for the command-line interface."""

import json
import subprocess
import sys

from docintel.cli import main
from docintel.data import make_invoice


def test_cli_reads_file(tmp_path, capsys):
    doc = make_invoice(seed=11)
    path = tmp_path / "inv.txt"
    path.write_text(doc.text)
    rc = main([str(path)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["fields"]["invoice_number"] == doc.fields["invoice_number"]


def test_cli_empty_input_errors(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n")
    assert main([str(path)]) == 1


def test_cli_stdin_entrypoint():
    doc = make_invoice(seed=12)
    proc = subprocess.run(
        [sys.executable, "-m", "docintel.cli"],
        input=doc.text,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    out = json.loads(proc.stdout)
    assert out["doc_type"] == "invoice"
