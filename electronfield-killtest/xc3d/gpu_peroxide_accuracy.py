"""New 18-electron peroxide accuracy test with post-prediction reference."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import torch
from .gpu_charge_spin_accuracy import prepare, model_prediction, PhysicalAction
from .gpu_canonical_density import slater_vector
from .gpu_four_particle import cached_ritz

KCAL = 627.509474


def main():
    device = torch_setup()
    started = time.perf_counter()
    specification = json.loads(Path("research/peroxide_accuracy_protocol_20261117.json").read_text())
    base = json.loads(Path("research/charge_spin_accuracy_protocol_20261117.json").read_text())
    protocol = {**base, "maximum_adaptive_updates": specification["maximum_adaptive_updates"]}
    case = specification["case"]
    data = prepare(case, device)
    action = PhysicalAction(data)
    with torch.no_grad():
        pairs = case["electrons"]//2
        occupied = data["orbitals"][:, :pairs]
        seed = slater_vector(occupied, torch.arange(pairs, device=device), data["bits"])
        tensor = action.six_action.to_tensor(seed)
        roundtrip = float((action.six_action.to_fock(tensor)-seed).abs().max())
        fast, direct = action.operator(tensor), action.operator.naive(tensor)
        action_error = float((fast-direct).abs().max())
        density = 2*occupied @ occupied.T
        coulomb = torch.einsum("ijkl,kl->ij", data["eri"], density)
        exchange = torch.einsum("ikjl,kl->ij", data["eri"], density)
        wick = (density*(data["h"]+.5*coulomb-.25*exchange)).sum()+data["nuclear"]
        wick_error = float((seed @ action(seed)-wick).abs())
        del tensor, fast, direct
    checks = {"fock_roundtrip_max_error": roundtrip, "independent_action_max_error_hartree": action_error,
              "input_wick_error_hartree": wick_error}
    if max(roundtrip, action_error, wick_error) > 1e-8:
        raise RuntimeError(f"New six-hole operator failed its physical identities: {checks}")
    print(json.dumps({"id": case["id"], "phase": "six_hole_checks_passed", **checks}), flush=True)
    prediction = model_prediction(data, protocol)
    # Independent same-Hamiltonian label; no energy used before this point.
    reference = cached_ritz(action, prediction["unprojected_reference_seed"],
                            maximum=specification["maximum_reference_ritz_dimension"],
                            tolerance=specification["reference_residual_tolerance"])
    error = float((prediction["energy"]-reference["energy"])*KCAL) if reference["converged"] else None
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "protocol": specification,
              "case": case, "spatial_orbitals": len(data["h"]), "full_fixed_sz_dimension": len(data["bits"]),
              "operator_checks": checks, "energy_error_kcal": error, "below_one_kcal": error is not None and abs(error) < 1,
              "total_energy_hartree": float(prediction["energy"]), "reference_energy_hartree": float(reference["energy"]),
              "reference_converged": reference["converged"], "reference_residual": reference["residual"],
              "reference_dimension": reference["dimension"], "reference_energy_fitted": False,
              "seconds": time.perf_counter()-started, "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
    report.update({key: value for key, value in prediction.items() if key not in ("state", "energy", "unprojected_reference_seed", "block_checkpoints")})
    output = Path("research/peroxide_accuracy_gpu_20261117.json")
    output.write_text(json.dumps(report, indent=2)+"\n")
    torch.save({"state": prediction["state"].cpu(), "reference_state": reference["state"].cpu(),
                "block_checkpoints": prediction["block_checkpoints"], "report": report},
               ROOT / "data/charge_spin_accuracy" / f"{case['id']}.model.pt")
    print(json.dumps({key: value for key, value in report.items() if key not in ("history", "local_multiplets", "protocol")}), flush=True)


if __name__ == "__main__":
    main()
