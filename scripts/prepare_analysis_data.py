#!/usr/bin/env python3
"""Extrai somente os dados tabulares necessários para a análise da campanha."""

import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


REQUIRED = {
    "results_raw.csv",
    "results_summary.csv",
    "comparisons_raw.csv",
    "comparisons_summary.csv",
    "accuracy.csv",
    "manifest.csv",
    "environment.txt",
    "input_SHA256SUMS",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepara os CSVs da campanha sem extrair os milhares de arquivos brutos."
    )
    parser.add_argument(
        "archive", nargs="?", default="condor_full_results.tar.gz",
        help="arquivo .tar.gz produzido pela campanha",
    )
    parser.add_argument(
        "--output", default="results/analysis_data",
        help="diretório que receberá os CSVs (padrão: results/analysis_data)",
    )
    args = parser.parse_args()
    archive = Path(args.archive).resolve()
    output = Path(args.output).resolve()
    if not archive.is_file():
        raise SystemExit(f"[ERRO] Arquivo não encontrado: {archive}")

    output.mkdir(parents=True, exist_ok=True)
    found = {}
    campaign_roots = set()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            posix = PurePosixPath(member.name)
            if posix.is_absolute() or ".." in posix.parts:
                raise SystemExit(f"[ERRO] Caminho inseguro no arquivo: {member.name}")
            if member.isfile() and posix.name in REQUIRED:
                if posix.name in found:
                    raise SystemExit(f"[ERRO] Arquivo duplicado no pacote: {posix.name}")
                found[posix.name] = member
                if len(posix.parts) > 1:
                    campaign_roots.add(posix.parts[0])

        missing = sorted(REQUIRED - set(found))
        if missing:
            raise SystemExit("[ERRO] Dados ausentes no pacote: " + ", ".join(missing))

        for name, member in found.items():
            source = bundle.extractfile(member)
            if source is None:
                raise SystemExit(f"[ERRO] Não foi possível ler {member.name}")
            with source, (output / name).open("wb") as destination:
                shutil.copyfileobj(source, destination)

    metadata = {
        "campaign": sorted(campaign_roots)[0] if len(campaign_roots) == 1 else None,
        "source_archive": str(archive),
        "source_sha256": sha256(archive),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {name: sha256(output / name) for name in sorted(REQUIRED)},
    }
    (output / "analysis_source.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Dados preparados em: {output}")
    print(f"Campanha: {metadata['campaign']}")
    print(f"SHA-256 do pacote: {metadata['source_sha256']}")


if __name__ == "__main__":
    main()
