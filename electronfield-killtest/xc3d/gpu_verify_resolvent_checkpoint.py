"""Resume one interrupted analytic problem, never repeat its completed steps."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_density_resolvent import proximal_state


def main():
    output = Path("research/resolvent_checkpoint_checks_gpu_20261117.json")
    if output.exists(): raise RuntimeError("This checkpoint-recovery check already has a report")
    device = torch_setup(17112052)
    matrix = torch.tensor([[.3, .11], [.11, -.2]], device=device)
    operators = torch.tensor([[[1., 0.], [0., -1.]], [[0., .4], [.4, 0.]]], device=device)
    auxiliary = torch.tensor([.17, .08], device=device)
    initial = torch.eye(2, device=device)/2
    expected = torch.tensor([[.485, -.175], [-.175, .515]], device=device)
    saved = None
    class DeliberateInterruption(Exception): pass
    def interrupt(value):
        nonlocal saved
        saved = {key: data.clone() if isinstance(data, torch.Tensor) else list(data) if isinstance(data, list) else data for key, data in value.items()}
        raise DeliberateInterruption()
    try:
        proximal_state(matrix, operators, auxiliary, .8, initial, maximum=4000, gap_tolerance=1e-11, checkpoint_callback=interrupt)
    except DeliberateInterruption:
        pass
    resumed = proximal_state(matrix, operators, auxiliary, .8, initial, maximum=4000, gap_tolerance=1e-11, checkpoint=saved)
    error = float((resumed["state"]-expected).norm())
    completed = {"state": resumed["state"], "point": resumed["state"], "acceleration": 1., "energy": resumed["functional"],
                 "history": resumed["history"], "gap": resumed["gap"], "iteration": resumed["iteration"], "epsilon": .8,
                 "total": 1., "cap": None, "maximum": 4000, "gap_tolerance": 1e-11, "auxiliary": auxiliary}
    recovered = proximal_state(matrix, operators, auxiliary, .8, initial, maximum=4000, gap_tolerance=1e-11, checkpoint=completed)
    history_indices = [row["iteration"] for row in resumed["history"]]
    valid = (saved["iteration"] == 0 and resumed["converged"] and error < 1e-9 and
             len(history_indices) == len(set(history_indices)) and recovered["iteration"] == resumed["iteration"]
             and torch.equal(recovered["state"], resumed["state"]))
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "passed": valid,
              "interrupted_after_iteration": saved["iteration"], "completed_iteration": resumed["iteration"],
              "analytic_state_error": error, "gap_hartree": resumed["gap"],
              "duplicate_history_steps": len(history_indices)-len(set(history_indices)),
              "completed_checkpoint_required_new_steps": recovered["iteration"] != resumed["iteration"],
              "new_chemical_or_reference_solve": False}
    output.write_text(json.dumps(report, indent=2)+"\n"); print(json.dumps(report), flush=True)
    if not valid: raise RuntimeError("Proximal checkpoint continuation failed its analytic state check")


if __name__ == "__main__": main()
