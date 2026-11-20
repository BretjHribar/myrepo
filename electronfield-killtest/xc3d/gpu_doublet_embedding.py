"""Physical doublet ensembles represented by a singlet purification.

An inert singly occupied auxiliary site supplies spin one half. The physical
prefix is explicitly fixed to its own N,S=(odd N,1/2); coupling the two spins
to total zero and tracing the auxiliary yields a positive doublet ensemble.
The auxiliary has no physical kinetic, potential or Coulomb terms. These
are standard Clebsch-Gordan/purification identities, not a novelty claim.
This module requires independent validation before molecular use.
"""
from dataclasses import dataclass
import torch
from .gpu_local_tensor import LocalSpinBasis
from .gpu_spin_response import SpinChain
from .gpu_two_site_tensor import TwoSiteSpinBasis, pair_problem
from .gpu_tensor_density import density_matrix
from .gpu_manybody_ops import spin_words
from .gpu_fermion_mpo import FermionMPO
from .gpu_purified_spin import purified_products, purified_norm


class RestrictedTwoSiteBasis(TwoSiteSpinBasis):
    def __init__(self, full, selected):
        self.left, self.right = full.left, full.right
        self.paths = [full.paths[index] for index in selected]
        ids = torch.tensor(selected, dtype=torch.long, device=full.coefficient.device)
        inverse = torch.full((full.dimension,), -1, dtype=torch.long, device=ids.device)
        inverse[ids] = torch.arange(len(ids), device=ids.device)
        keep = inverse[full.parameter] >= 0
        self.index = full.index[keep]
        self.parameter = inverse[full.parameter[keep]]
        self.coefficient = full.coefficient[keep]
        self.norm_squared = full.norm_squared[ids]
        self.inverse_norm = self.norm_squared.rsqrt()
        self.selected = ids


@dataclass(frozen=True)
class DoubletEmbedding:
    physical_sites: int
    physical_electrons: int

    def __post_init__(self):
        if self.physical_sites < 1 or self.physical_electrons % 2 != 1 or not 1 <= self.physical_electrons < 2*self.physical_sites:
            raise ValueError("The doublet domain requires an odd physical electron count in its stated orbitals")

    def seed(self, device):
        if torch.device(device).type != "cuda": raise RuntimeError("Doublet coefficients require CUDA; no CPU fallback")
        pairs = self.physical_electrons//2
        number, spin = 0, 0
        labels = [[(0, 0, 0, 0)]]
        for site in range(self.physical_sites):
            physical = 2 if site < pairs else 1 if site == pairs else 0
            number += physical
            if physical == 1: spin = 1
            labels.append([(number, spin, 0, magnetic) for magnetic in range(-spin, spin+1, 2)])
        labels.append([(number+1, 0, 0, 0)])
        cores = []
        for site in range(self.physical_sites+1):
            basis = LocalSpinBasis(labels[site], labels[site+1], device)
            if basis.dimension != 1: raise RuntimeError("The specified doublet seed does not have a unique reduced CG path")
            cores.append(basis.core(torch.ones(1, 1, device=device)))
        return SpinChain(cores, labels, number+1, 0).normalized()

    def pad_integrals(self, h, eri):
        if h.device.type != "cuda" or eri.device != h.device:
            raise RuntimeError("Doublet embedding requires CUDA coefficients")
        if h.shape != (self.physical_sites,)*2 or eri.shape != (self.physical_sites,)*4:
            raise ValueError("Physical operator dimensions differ from the declared domain")
        size = self.physical_sites+1
        first = h.new_zeros(size, size); first[:-1, :-1] = h
        second = eri.new_zeros((size,)*4); second[:-1, :-1, :-1, :-1] = eri
        return first, second

    def problem(self, mpo, state, center, maximum_environment_bytes):
        if (len(state.cores), state.electrons, state.twice_spin) != (self.physical_sites+1, self.physical_electrons+1, 0):
            raise ValueError("The singlet purification does not match its physical doublet domain")
        if any(label[:2] != (self.physical_electrons, 1) for label in state.labels[self.physical_sites]):
            raise RuntimeError("Physical charge/spin sector was changed at the auxiliary boundary")
        full = pair_problem(mpo, state, center, maximum_environment_bytes)
        selected = []
        for index, path in enumerate(full["basis"].paths):
            if center+1 == self.physical_sites and (path[4:6] != (self.physical_electrons, 1) or path[6] != 1):
                continue
            selected.append(index)
        if not selected: raise RuntimeError("The exact physical doublet section is empty")
        restricted = RestrictedTwoSiteBasis(full["basis"], selected)
        seed = full["seed"][restricted.selected]
        if abs(float(seed.square().sum())-1.) > 1e-9:
            raise RuntimeError("Physical doublet sector projection lost the retained input")
        def action(value):
            expanded = value.new_zeros(full["basis"].dimension)
            expanded[restricted.selected] = value
            image = full["action"](expanded)
            kept = image[restricted.selected]
            outside = image.clone(); outside[restricted.selected] = 0.
            if float(outside.norm()) > 1e-9*max(1., float(value.norm())):
                raise RuntimeError("The literal Hamiltonian mixes physical and auxiliary charge/spin sectors")
            return kept
        return {**full, "basis": restricted, "action": action, "seed": seed,
                "unrestricted_section_dimension": full["basis"].dimension,
                "fixed_physical_prefix": {"electrons": self.physical_electrons, "twice_spin": 1}}

    def physical_checks(self, state, tolerance=1e-8):
        full_density = density_matrix(state)
        density = full_density[:-1, :-1]
        words, coefficient = spin_words(self.physical_sites, density.device)
        longitudinal, signs = [], []
        for first in range(self.physical_sites):
            for second in range(self.physical_sites):
                for spin in range(2):
                    for other in range(2):
                        longitudinal.append(((first, spin, 1), (first, spin, 0), (second, other, 1), (second, other, 0)))
                        signs.append(1 if spin == other else -1)
        words = torch.cat((words, torch.tensor(longitudinal, device=density.device)))
        coefficient = torch.cat((coefficient, .25*torch.tensor(signs, device=density.device)))
        spin = FermionMPO(words, coefficient, self.physical_sites+1).expectation(state.cores)
        operators = torch.eye(4, device=density.device).expand(1, self.physical_sites+1, 4, 4).clone()
        operators[0, -1] = torch.diag(density.new_tensor([1., 0., 0., 1.]))
        auxiliary_charge_variance = purified_products(state.cores, operators)[0]/purified_norm(state.cores)
        occupations = torch.linalg.eigvalsh(density)
        number_error = float((density.trace()-self.physical_electrons).abs())
        spin_error = float((spin-.75).abs())
        auxiliary_error = float(auxiliary_charge_variance.abs())
        valid = (max(number_error, spin_error, auxiliary_error) < tolerance and float(occupations.min()) >= -tolerance
                 and float(occupations.max()) <= 2+tolerance)
        return density, {"passed": valid, "physical_electrons": self.physical_electrons, "physical_twice_spin": 1,
            "physical_number_error": number_error, "physical_spin_squared": float(spin), "physical_spin_error": spin_error,
            "auxiliary_charge_variance": auxiliary_error, "minimum_physical_occupation": float(occupations.min()),
            "maximum_physical_occupation": float(occupations.max()), "auxiliary_included_in_reported_physical_density": False}
