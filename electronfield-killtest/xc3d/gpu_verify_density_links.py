"""Check the new density links against independent particle/hole actions."""
from .runtime import configure, torch_setup
ROOT = configure(gpu_only_compatible=True)

from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from .gpu_density_links import DensityLinks
from .gpu_charge_spin_accuracy import PhysicalAction
from .intrinsic_density_frame import one_body_density


@torch.no_grad()
def main():
    device = torch_setup()
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": device, "results": []}
    for name in ("BeH2_631g_1.65_new", "H2F_plus_new", "H2O2_skew_new"):
        data = torch.load(ROOT / "data/charge_spin_accuracy" / f"{name}.pt", map_location=device, weights_only=True)
        links = DensityLinks(data["bits"], len(data["h"]))
        basis, _ = torch.linalg.qr(torch.randn(len(data["bits"]), 3, device=device), mode="reduced")
        fields = torch.randn(4, len(data["h"]), len(data["h"]), device=device)
        fields = (fields+fields.transpose(-1, -2))/2
        fields /= fields.flatten(1).norm(dim=1)[:, None, None]
        generators = links.project(basis)
        reduced = links.reduced(fields, generators)
        images = links.response_images(fields, basis)
        direct = torch.stack([links.image(field, basis) for field in fields])
        projection_error = float((reduced-torch.einsum("ia,bij->baj", basis, images)).abs().max())
        image_error = float((images-direct).abs().max())
        base = PhysicalAction(data)
        changed = PhysicalAction({**data, "h": data["h"]+fields[0]})
        independent_error = float((changed(basis[:, 0])-base(basis[:, 0])-direct[0, :, 0]).norm())
        small = torch.randn(3, 3, device=device); small = small @ small.T; small /= small.trace()
        density = links.density(small, generators)
        weights, rotation = torch.linalg.eigh(small)
        states = basis @ rotation
        original = sum(weights[index]*one_body_density(states[:, index], data["bits"], len(data["h"])) for index in range(3))
        density_error = float((density-original).abs().max())
        number_error = float((density.trace()-data["case"]["electrons"]).abs())
        valid = max(projection_error, image_error, independent_error, density_error, number_error) < 1e-9
        row = {"id": name, "passed": valid, "electrons": data["case"]["electrons"],
               "projection_error": projection_error, "response_image_error": image_error,
               "independent_particle_hole_action_error": independent_error,
               "ensemble_density_error": density_error, "number_error": number_error,
               "occupation_link_bytes": links.storage_bytes(), "new_reference_generated": False}
        report["results"].append(row); print(json.dumps(row), flush=True)
        del data, links, basis, fields, generators, reduced, images, direct, base, changed
        torch.cuda.empty_cache()
    report["passed"] = all(row["passed"] for row in report["results"])
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    Path("research/density_link_checks_gpu_20261117.json").write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]:
        raise RuntimeError("Density links failed their independent operator checks")


if __name__ == "__main__":
    main()
