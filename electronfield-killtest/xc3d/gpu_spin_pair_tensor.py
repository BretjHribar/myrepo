"""Charge/spin Schmidt tensors for the new physical correlation state.

This exports the global coefficient into local multiplets and a physical
spin-coupled bond. No global determinant-count truncation is imposed. Ordinary
Schmidt decomposition/CG coefficients are primitives, not novelty claims.
The two-site representation is a foundation for a general block tensor chain;
it does not yet implement a large-molecule Hamiltonian solver.
"""
import torch
from .gpu_fock_blocks import rectangular_words


@torch.no_grad()
def magnetic_states(cache, number, twice_spin, highest):
    words = torch.tensor([((i, 1, 1), (i, 0, 0)) for i in range(cache.sites)], device=cache.device)
    coefficients = highest.new_ones(cache.sites)
    states, magnetic, state = {twice_spin: highest}, twice_spin, highest
    while magnetic > -twice_spin:
        up, down = (number+magnetic)//2, (number-magnetic)//2
        matrix = rectangular_words(words, coefficients, cache.bits(up-1, down+1), cache.bits(up, down))
        normalization = highest.new_tensor((twice_spin+magnetic)*(twice_spin-magnetic+2)).sqrt()/2
        state = matrix @ state/normalization
        magnetic -= 2; states[magnetic] = state
    return states


class SpinCoupledPairTensor:
    def __init__(self, electrons, sectors):
        self.electrons, self.sectors = electrons, sectors
        if any(value["first_number"]+value["second_number"] != electrons for value in sectors):
            raise ValueError("A spin tensor sector changes total electron number")

    @classmethod
    @torch.no_grad()
    def from_fock(cls, vector, groups, first, second):
        matrices = groups.matrices(vector, 0)
        sectors = []
        for (number, spin), left in first.entries.items():
            right_number = groups.up+groups.down-number
            if (right_number, spin) not in second.entries:
                continue
            right = second.entries[right_number, spin]
            reduced = vector.new_zeros(left["coordinates"].shape[1], right["coordinates"].shape[1])
            for magnetic in range(spin, -spin-1, -2):
                key = (number+magnetic)//2, (number-magnetic)//2
                if key not in matrices:
                    continue
                sign = 1-2*(((spin-magnetic)//2) % 2)
                coefficient = vector.new_tensor(sign)/vector.new_tensor(spin+1).sqrt()
                reduced += coefficient*(left["states"][magnetic].T @ matrices[key] @ right["states"][-magnetic])
            u, singular, vh = torch.linalg.svd(reduced, full_matrices=False)
            selected = singular > max(1e-15, float(singular[0])*1e-13)
            if not bool(selected.any()):
                continue
            sectors.append({"first_number": number, "second_number": right_number, "twice_spin": spin,
                            "first_highest": left["states"][spin] @ u[:, selected],
                            "second_highest": right["states"][spin] @ vh[selected].T,
                            "singular_values": singular[selected]})
        return cls(groups.up+groups.down, sectors)

    @torch.no_grad()
    def norm_squared(self):
        return sum(value["singular_values"].square().sum() for value in self.sectors)

    @torch.no_grad()
    def to_fock(self, groups):
        prototype = self.sectors[0]["singular_values"]
        matrices = {group["a"]: prototype.new_zeros(group["shape"]) for group in groups.groups}
        for value in self.sectors:
            number, right_number, spin = value["first_number"], value["second_number"], value["twice_spin"]
            left = magnetic_states(groups.first, number, spin, value["first_highest"])
            right = magnetic_states(groups.second, right_number, spin, value["second_highest"])
            for magnetic in range(spin, -spin-1, -2):
                key = (number+magnetic)//2, (number-magnetic)//2
                if key not in matrices:
                    continue
                sign = 1-2*(((spin-magnetic)//2) % 2)
                coefficient = prototype.new_tensor(sign)/prototype.new_tensor(spin+1).sqrt()
                matrices[key] += coefficient*((left[magnetic]*value["singular_values"]) @ right[-magnetic].T)
        result = prototype.new_zeros(len(groups.bits))
        for group in groups.groups:
            result[group["order"]] = matrices[group["a"]].flatten()
        return result

    def payload(self):
        return {"electrons": self.electrons, "global_twice_spin": 0,
                "sectors": [{key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in sector.items()}
                            for sector in self.sectors]}

    def summary(self):
        return [{"first_number": value["first_number"], "second_number": value["second_number"],
                 "twice_spin": value["twice_spin"], "spin_schmidt_channels": len(value["singular_values"]),
                 "weight": float(value["singular_values"].square().sum())} for value in self.sectors]
