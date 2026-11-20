"""Data-only GMTKN55 template parser and complete CUDA reaction scorer.

No upstream shell script or evaluator is executed. Parsing decimal strings
is input I/O; all energy combinations, weights and statistics use CUDA.
"""
import hashlib
import json
import re
import shlex
from pathlib import Path, PurePosixPath


GROUPS = {
    "basic_properties_and_small_reactions": "W4-11 G21EA G21IP DIPCS10 PA26 SIE4x4 ALKBDE10 YBDE18 AL2X6 HEAVYSB11 NBPRC ALK8 RC21 G2RC BH76RC FH51 TAUT15 DC13".split(),
    "large_reactions_and_isomerizations": "MB16-43 DARC RSE43 BSR36 CDIE20 ISO34 ISOL24 C60ISO PArel".split(),
    "barriers": "BH76 BHPERI BHDIV10 INV24 BHROT27 PX13 WCPT18".split(),
    "intermolecular_interactions": "RG18 ADIM6 S22 S66 HEAVY28 WATER27 CARBHB12 PNICO23 HAL59 AHB21 CHB6 IL16".split(),
    "intramolecular_interactions": "IDISP ICONF ACONF Amino20x4 PCONF21 MCONF SCONF UPU23 BUT14DIOL".split(),
}
DECIMAL = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
INTEGER = re.compile(r"[+-]?\d+\Z")
SUPPLEMENT = {
    "BH76/.averageRC": "9d9c8639b68f3d2ee9e156c5eaf43ee621d0bbcf",
    "BH76/.numenRC": "64bb6b746dceaf12b0ba8c08f310b0426babde44",
}


def git_blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def expand_structure_token(token):
    """Only the single comma-brace form present in the pinned data is legal."""
    if not token.endswith("/$f"):
        raise ValueError(f"Unexpected structure/method placeholder: {token}")
    token = token[:-3]
    if "{" in token or "}" in token:
        match = re.fullmatch(r"([^{}]*)\{([^{}]+)\}([^{}]*)", token)
        if match is None or "," not in match[2]:
            raise ValueError(f"Unsupported brace grammar: {token}")
        expanded = [match[1] + item + match[3] for item in match[2].split(",")]
    else:
        expanded = [token]
    for item in expanded:
        if not re.fullmatch(r"[A-Za-z0-9_+.,()=-]+", item) or item in (".", ".."):
            raise ValueError(f"Unsupported compound identifier: {item}")
    return expanded


def parse_definition(text, subset, directory, known_ids):
    rows = []
    header_lines = {
        'if [ "$TMER" == "" ]', "then", "tmer=tmer2++", "else", "tmer=$TMER", "fi", "f=$1", "w=$2",
    }
    for line_number, line in enumerate(text.splitlines(), 1):
        tokens = shlex.split(line, comments=True)
        if not tokens:
            continue
        if tokens[0] not in ("$tmer", "tmer2++"):
            if line.strip() not in header_lines:
                raise ValueError(f"Unrecognized non-data line in {subset}:{line_number}: {line}")
            continue
        if tokens.count("x") != 1 or tokens.count("$w") != 1:
            raise ValueError(f"Invalid reaction delimiters: {subset}:{line_number}")
        cut, weight = tokens.index("x"), tokens.index("$w")
        ids = [f"{directory}/{name}" for token in tokens[1:cut] for name in expand_structure_token(token)]
        coefficients = tokens[cut+1:weight]
        if len(ids) != len(coefficients) or not ids or any(not INTEGER.fullmatch(value) for value in coefficients):
            raise ValueError(f"Invalid stoichiometry: {subset}:{line_number}")
        if not set(ids) <= known_ids:
            raise ValueError(f"Unknown structure: {subset}:{line_number}: {set(ids)-known_ids}")
        if weight+1 >= len(tokens) or not DECIMAL.fullmatch(tokens[weight+1]):
            raise ValueError(f"Invalid reference: {subset}:{line_number}")
        trailing = tokens[weight+2:]
        # The pinned templates use only neutral numeric options 0,1 and
        # descriptive words. Never interpret them as extra energies.
        if trailing and DECIMAL.fullmatch(trailing[0]):
            if len(trailing) < 2 or trailing[:2] not in (["0", "1"], ["0", "1.0"]):
                raise ValueError(f"Unsupported nonneutral evaluator options: {subset}:{line_number}")
            labels = trailing[2:]
        else:
            labels = trailing
        if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", label) for label in labels):
            raise ValueError(f"Unsupported reaction annotation: {subset}:{line_number}")
        rows.append({"id": f"{subset}/{len(rows)+1:03d}", "subset": subset,
                     "source_line": line_number, "structure_ids": ids,
                     "coefficients": [int(value) for value in coefficients],
                     "reference_kcal_per_mol_text": tokens[weight+1],
                     "source_trailing_annotations": trailing})
    if not rows:
        raise ValueError(f"Empty reaction set: {subset}")
    return rows


class GMTKN55Suite:
    def __init__(self, research_root, protocol):
        self.protocol = protocol
        manifest_path = Path(research_root)/protocol["manifest"]
        manifest = json.loads(manifest_path.read_text())
        if manifest["commit_sha"] != protocol["source_commit"]:
            raise ValueError("Wrong benchmark source version")
        self.manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        self.structures = {row["id"]: row for row in manifest["structures"]}
        if len(self.structures) != protocol["expected_structure_inventory"]:
            raise ValueError("Incomplete structure inventory")
        files = {row["path"]: row for row in manifest["files"]}
        folder = manifest_path.parent/"data"
        self.source_hashes = {}

        def read(relative):
            data = (folder/relative).read_bytes()
            if relative in SUPPLEMENT:
                if git_blob(data) != SUPPLEMENT[relative]:
                    raise ValueError(f"Wrong supplemental pinned metadata: {relative}")
            elif relative not in files or hashlib.sha256(data).hexdigest() != files[relative]["sha256"]:
                raise ValueError(f"Pinned benchmark metadata changed: {relative}")
            self.source_hashes[relative] = hashlib.sha256(data).hexdigest()
            return data.decode("utf-8")

        self.reactions, self.subsets = [], {}
        expected_sets = {name for names in GROUPS.values() for name in names}
        if set(manifest["evaluation_sets"]) != expected_sets or len(expected_sets) != protocol["expected_evaluation_sets"]:
            raise ValueError("Incomplete evaluation sets or classification")
        for subset in sorted(expected_sets):
            directory = "BH76" if subset == "BH76RC" else subset
            suffix = "RC" if subset == "BH76RC" else ""
            source = f"{directory}/.res{suffix}"
            rows = parse_definition(read(source), subset, directory, set(self.structures))
            count_text, average_text = read(f"{directory}/.numen{suffix}").strip(), read(f"{directory}/.average{suffix}").strip()
            if not INTEGER.fullmatch(count_text) or int(count_text) != len(rows) or not DECIMAL.fullmatch(average_text):
                raise ValueError(f"Incorrect reaction count/average metadata: {subset}")
            start = len(self.reactions)
            self.reactions.extend(rows)
            self.subsets[subset] = {"start": start, "stop": len(self.reactions), "count": len(rows),
                                     "source": source, "average_kcal_per_mol_text": average_text,
                                     "group": next(group for group, names in GROUPS.items() if subset in names)}
        if len(self.reactions) != protocol["expected_reactions"]:
            raise ValueError("Incomplete reaction suite")
        self.required_ids = sorted({name for row in self.reactions for name in row["structure_ids"]})
        self.required_index = {name: index for index, name in enumerate(self.required_ids)}

    def tensors(self, device):
        import torch
        if torch.device(device).type != "cuda":
            raise RuntimeError("GMTKN55 numerical evaluation requires CUDA")
        indices, columns, coefficients = [], [], []
        for i, row in enumerate(self.reactions):
            indices.extend([i]*len(row["structure_ids"]))
            columns.extend(self.required_index[name] for name in row["structure_ids"])
            coefficients.extend(row["coefficients"])
        matrix = torch.zeros((len(self.reactions), len(self.required_ids)), dtype=torch.float64, device=device)
        matrix.index_put_((torch.tensor(indices, device=device), torch.tensor(columns, device=device)),
                          torch.tensor(coefficients, dtype=torch.float64, device=device), accumulate=True)
        references = torch.tensor([float(row["reference_kcal_per_mol_text"]) for row in self.reactions], dtype=torch.float64, device=device)
        means = torch.tensor([float(self.subsets[row["subset"]]["average_kcal_per_mol_text"]) for row in self.reactions], dtype=torch.float64, device=device)
        if not bool(torch.isfinite(references).all() & torch.isfinite(means).all() & (means > 0).all()):
            raise ValueError("Nonfinite references or invalid normalization weights")
        normalization = references.new_tensor(float(self.protocol["normalization_kcal_per_mol"]))
        return {"matrix": matrix, "references": references, "weights": normalization/means,
                "hartree_to_kcal": references.new_tensor(float(self.protocol["hartree_to_kcal_per_mol"]))}

    def score(self, energies, accepted, tensors):
        """Accepted flags exclude failed, unsupported or physically invalid cases."""
        import torch
        matrix = tensors["matrix"]
        if energies.device != matrix.device or accepted.device != matrix.device or energies.dtype != torch.float64 or accepted.dtype != torch.bool:
            raise ValueError("Energy and acceptance inputs must be float64/bool CUDA tensors")
        if energies.shape != (len(self.required_ids),) or accepted.shape != energies.shape:
            raise ValueError("Wrong complete structure prediction indexing")
        accepted = accepted & torch.isfinite(energies)
        missing_terms = (matrix != 0).to(torch.float64) @ (~accepted).to(torch.float64)
        available = missing_terms == 0
        safe_energy = torch.where(accepted, energies, torch.zeros_like(energies))
        predictions = (matrix @ safe_energy)*tensors["hartree_to_kcal"]
        errors = predictions-tensors["references"]
        weighted = errors.abs()*tensors["weights"]
        observed = torch.where(available, weighted, torch.zeros_like(weighted))
        complete = bool(available.all())
        measured_count = int(available.sum())
        result = {
            "complete_public_suite_scored": complete,
            "wtmad2_kcal_per_mol": float(observed.mean()) if complete else None,
            "measured_reactions": measured_count, "total_reactions": len(self.reactions),
            "accepted_structures": int(accepted.sum()), "required_structures": len(self.required_ids),
            "measured_nonnegative_contribution_lower_bound_kcal_per_mol": float(observed.mean()) if measured_count else None,
            "full_suite_point_estimate_kcal_per_mol": float(observed.mean()) if complete else None,
            "uncertainty": "Exact benchmark aggregation of supplied predictions" if complete else "Unmeasured errors have no finite upper bound; this coverage is not a probability sample",
            "maximum_absolute_reaction_error_kcal_per_mol": float(errors.abs().max()) if complete else None,
            "every_reaction_error_below_one_kcal_per_mol": bool((errors.abs() < 1.).all()) if complete else False,
            "subsets": {}, "groups": {}, "reaction_results": [],
        }
        for name, metadata in self.subsets.items():
            section = slice(metadata["start"], metadata["stop"])
            mask = available[section]
            count = int(mask.sum())
            result["subsets"][name] = {
                "measured_reactions": count, "total_reactions": metadata["count"],
                "mad_kcal_per_mol": float(errors[section].abs().mean()) if bool(mask.all()) else None,
                "wtmad2_contribution_kcal_per_mol": float(observed[section].sum()/len(self.reactions)),
                "all_reactions_scored": bool(mask.all()),
            }
        for group, names in GROUPS.items():
            mask = torch.tensor([row["subset"] in names for row in self.reactions], device=matrix.device)
            count = int((available & mask).sum())
            size = int(mask.sum())
            result["groups"][group] = {"measured_reactions": count, "total_reactions": size,
                                       "wtmad2_kcal_per_mol": float(observed[mask].mean()) if count == size else None}
        for i, row in enumerate(self.reactions):
            valid = bool(available[i])
            result["reaction_results"].append({"id": row["id"], "scored": valid,
                "prediction_kcal_per_mol": float(predictions[i]) if valid else None,
                "error_kcal_per_mol": float(errors[i]) if valid else None,
                "missing_or_failed_structure_ids": [] if valid else [name for name in row["structure_ids"] if not bool(accepted[self.required_index[name]])]})
        result["public_accuracy_threshold_met"] = complete and result["wtmad2_kcal_per_mol"] < 1.
        result["organizer_private_tests_passed"] = False
        return result

    def read_predictions(self, path, device):
        """Require explicit candidate provenance and individual acceptance flags."""
        import torch
        document = json.loads(Path(path).read_text())
        if document.get("schema") != "gmtkn55_total_energies_v1":
            raise ValueError("Unknown prediction contract")
        if document.get("benchmark_commit") != self.protocol["source_commit"] or document.get("manifest_sha256") != self.manifest_sha256:
            raise ValueError("Predictions refer to different benchmark inputs")
        if document.get("units") != "hartree" or document.get("energy_kind") != self.protocol["prediction_energy_kind"]:
            raise ValueError("Predictions must include electronic and nuclear energies in Hartree")
        candidate = document.get("candidate", {})
        if not candidate.get("id") or not candidate.get("basis") or any(not re.fullmatch(r"[0-9a-f]{64}", candidate.get(field, "")) for field in ("protocol_sha256", "source_sha256")):
            raise ValueError("Frozen candidate identity, basis and source/protocol hashes are required")
        if document.get("benchmark_reference_used_for_prediction") is not False:
            raise ValueError("Benchmark references must be excluded from predictions")
        energies = torch.full((len(self.required_ids),), float("nan"), device=device, dtype=torch.float64)
        accepted = torch.zeros(len(self.required_ids), device=device, dtype=torch.bool)
        seen = set()
        status_counts = {name: 0 for name in ("accepted", "failed", "unsupported", "uncomputed")}
        for row in document.get("predictions", []):
            name = row.get("id")
            if name not in self.required_index or name in seen:
                raise ValueError(f"Unknown/duplicate prediction: {name}")
            seen.add(name)
            status = row.get("status")
            if status not in status_counts:
                raise ValueError(f"Unknown prediction status: {name}")
            status_counts[status] += 1
            if status != "accepted":
                if not row.get("reason"):
                    raise ValueError(f"An unsuccessful prediction needs an explicit reason: {name}")
                continue
            if row.get("converged") is not True or row.get("physical_checks_passed") is not True:
                raise ValueError(f"Accepted energy lacks convergence/physical acceptance: {name}")
            value = row.get("total_energy_hartree")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Invalid energy value: {name}")
            index = self.required_index[name]
            energies[index] = value
            if not bool(torch.isfinite(energies[index])):
                raise ValueError(f"Accepted energy is nonfinite: {name}")
            accepted[index] = True
        status_counts["uncomputed"] += len(self.required_ids)-len(seen)
        return energies, accepted, {"candidate": candidate, "prediction_status_counts": status_counts,
                                    "prediction_file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
