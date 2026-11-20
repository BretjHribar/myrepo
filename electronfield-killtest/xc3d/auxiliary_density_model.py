"""Universal auxiliary-density evaluations using physical singlet ensembles.

Full-space response checks are distinct from the convex reduced-space gap.
The state is always physical; the auxiliary coordinate need not be a density.
This implementation still uses global Fock vectors, not a scalable model.
"""
import torch
from .gpu_canonical_density import canonical_lift, slater_vector
from .gpu_fock_blocks import SingletProjector
from .gpu_density_links import DensityLinks
from .gpu_density_completion import append_directions
from .gpu_charge_spin_accuracy import PhysicalAction
from .gpu_four_particle import incremental_ritz
from .ensemble_spatial_model import leading_completion
from .gpu_density_resolvent import proximal_state


@torch.no_grad()
def recover_physical_checkpoint(universal, density_seed, checkpoint, protocol):
    """Recover an occupied singlet without trusting unused legacy directions."""
    number, bits = universal["electrons"], universal["bits"]
    singlet = SingletProjector(bits, len(density_seed), number//2)
    physical = PhysicalAction({"h": universal["kinetic"], "eri": universal["eri"], "bits": bits,
                               "case": {"electrons": number}, "nuclear": density_seed.new_zeros(())})
    basis_old, ensemble_old = checkpoint["basis"], checkpoint["reduced_ensemble"]
    values, vectors = torch.linalg.eigh((ensemble_old+ensemble_old.T)/2)
    cutoff = 64*torch.finfo(values.dtype).eps*len(values)*values.abs().max()
    selected = values > cutoff
    occupied = basis_old @ vectors[:, selected]
    spin_error = max(float(singlet.spin_squared(vector).norm()) for vector in occupied.unbind(1))
    if spin_error > 1e-8:
        raise RuntimeError(f"Saved occupied state is not a singlet and cannot be resumed: {spin_error}")
    occupations, natural = torch.linalg.eigh((density_seed+density_seed.T)/2)
    reference = canonical_lift(natural, occupations, number, bits)["vectors"].T
    purified = torch.stack([singlet(vector) for vector in occupied.unbind(1)], 1)
    basis = append_directions(basis_old.new_zeros(len(bits), 0), torch.cat((purified, reference), 1),
                              protocol["maximum_response_space_dimension"])
    overlap = basis.T @ basis_old
    ensemble = overlap @ ensemble_old @ overlap.T
    trace_loss = float((1-ensemble.trace()).abs())
    ensemble /= ensemble.trace()
    operator_error = float((overlap.T @ ensemble @ overlap-ensemble_old).norm())
    images = torch.stack([physical(vector) for vector in basis.unbind(1)], 1)
    matrix = basis.T @ images
    old_energy = sum(values[index]*(vector @ physical(vector)) for index, vector in
                     zip(torch.nonzero(selected).flatten().tolist(), occupied.unbind(1)))
    energy_error = float(((matrix*ensemble).sum()-old_energy).abs())
    links = DensityLinks(bits, len(density_seed))
    density = links.density(ensemble, links.project(basis))
    density_error = float((density-checkpoint["physical_density"]).norm())
    reference_error = float((basis @ (basis.T @ reference)-reference).norm())
    valid = trace_loss < 1e-10 and operator_error < 1e-10 and energy_error < 1e-8 and density_error < 1e-8 and reference_error < 1e-8
    report = {"passed": valid, "old_dimension": basis_old.shape[1], "restarted_dimension": basis.shape[1],
              "retained_ensemble_channels": int(selected.sum()), "discarded_trace": trace_loss,
              "ensemble_operator_error": operator_error, "universal_energy_change_hartree": energy_error,
              "density_change_norm": density_error, "input_retention_error": reference_error,
              "saved_occupied_spin_squared_norm": spin_error, "full_physical_hamiltonian_used": True}
    if not valid: raise RuntimeError(f"Physical checkpoint recovery failed retention: {report}")
    return {"basis": basis, "images": images, "reduced_ensemble": ensemble}, report


class AuxiliaryDensityModel:
    @torch.no_grad()
    def __init__(self, universal, density_seed, protocol, checkpoint=None):
        if set(universal) != {"kinetic", "eri", "operators", "bits", "electrons"}:
            raise ValueError("Universal interface excludes external potentials, charges and labels")
        self.universal, self.protocol = universal, protocol
        self.kinetic, self.eri, self.bits, self.number = (universal[key] for key in ("kinetic", "eri", "bits", "electrons"))
        self.operators, self.moments = universal["operators"], universal["operators"][1:]
        self.epsilon = protocol["epsilon_inverse_hartree"]
        if density_seed.device.type != "cuda" or self.number % 2 or min(self.number, 2*len(density_seed)-self.number) not in (4, 6):
            raise ValueError("This CUDA singlet backend requires four/six particles or holes")
        if any(universal[key].device != density_seed.device for key in ("kinetic", "eri", "operators", "bits")):
            raise ValueError("Every universal tensor must share the seed's CUDA device")
        if density_seed.dtype != torch.float64 or not bool(torch.isfinite(density_seed).all()) or float((density_seed-density_seed.T).norm()) > 1e-9:
            raise ValueError("Seed density must be finite, real symmetric float64")
        occupations, natural = torch.linalg.eigh((density_seed+density_seed.T)/2)
        if float(occupations.min()) < -1e-9 or float(occupations.max()) > 2+1e-9 or float((density_seed.trace()-self.number).abs()) > 1e-8:
            raise ValueError("Seed density violates fixed-N Pauli representability")
        identity = torch.eye(len(density_seed), device=density_seed.device)
        if float((self.operators[0]-identity/identity.new_tensor(len(identity)).sqrt()).norm()) > 1e-8:
            raise ValueError("The first density operator must be normalized fixed number")
        lift = canonical_lift(natural, occupations, self.number, self.bits)
        self.singlet = SingletProjector(self.bits, len(density_seed), self.number//2)
        self.links = DensityLinks(self.bits, len(density_seed))
        self.physical = PhysicalAction({"h": self.kinetic, "eri": self.eri, "bits": self.bits,
                                        "case": {"electrons": self.number}, "nuclear": self.kinetic.new_zeros(())})
        self.reference_vectors = lift["vectors"].T
        if checkpoint is not None:
            self.basis, self.images, self.ensemble = (checkpoint[key] for key in ("basis", "images", "reduced_ensemble"))
            if float((self.basis @ (self.basis.T @ self.reference_vectors)-self.reference_vectors).norm()) > 1e-8:
                raise RuntimeError("Restored auxiliary space lost the original physical input")
            self.projected_dimension = 0
            self.refresh()
            return
        maximum = protocol["maximum_response_space_dimension"]
        self.basis = append_directions(self.kinetic.new_zeros(len(self.bits), 0), lift["vectors"].T, maximum)
        self.images = torch.stack([self.action(vector) for vector in self.basis.unbind(1)], 1)
        self.basis = append_directions(self.basis, self.images, maximum)
        initial_prefix = self.basis
        active = lift["weights"].argsort(descending=True)[:protocol["maximum_density_response_states"]]
        sources = self.links.response_images(self.moments, lift["vectors"][active].T*lift["weights"][active].sqrt())
        self.basis, _, _ = leading_completion(self.basis, sources.permute(1, 0, 2).reshape(len(self.bits), -1),
                                                protocol["density_response_directions"], maximum)
        self.basis = self.purify_extension(initial_prefix, self.basis)
        support = self.basis.T @ lift["vectors"].T
        if float((self.basis @ support-lift["vectors"].T).norm()) > 1e-8:
            raise RuntimeError("Auxiliary state compiler lost its physical input")
        self.ensemble = (support*lift["weights"]) @ support.T
        self.projected_dimension = 0
        self.refresh()

    @torch.no_grad()
    def saved_point(self, auxiliary):
        """Recover a completed point's diagnostics without optimizing it again."""
        self.refresh()
        moments = torch.einsum("aij,ij->a", self.reduced, self.ensemble)
        gradient = (auxiliary-moments)/self.epsilon
        universal = (self.ensemble*self.matrix).sum()
        functional = universal+(moments-auxiliary).square().sum()/(2*self.epsilon)
        field = torch.einsum("a,aij->ij", -gradient, self.moments)
        effective_images = self.images+self.links.image(field, self.basis)
        effective = self.matrix+torch.einsum("a,aij->ij", -gradient, self.reduced)
        values, vectors = torch.linalg.eigh((self.ensemble+self.ensemble.T)/2)
        root = vectors*values.clamp_min(0).sqrt()
        residual = (effective_images-self.basis @ (self.basis.T @ effective_images)) @ root
        gap = float((effective*self.ensemble).sum()-torch.linalg.eigvalsh(effective)[0])
        return {"state": self.ensemble, "moments": moments, "functional": functional, "gradient": gradient,
                "effective": effective, "gap": gap, "converged": abs(gap) < self.protocol["reduced_gap_tolerance_hartree"],
                "physical_density": self.links.density(self.ensemble, self.generators),
                "outside_stationarity_residual": float(residual.norm()), "universal_energy": universal,
                "basis_dimension": len(self.matrix)}

    @torch.no_grad()
    def compact_checkpoint(self):
        """Retain the occupied ensemble and input, discard unused responses.

        This is a checked restart after the numerical rank-selection repair,
        not an increase in physical rank or a reference-error-based selection.
        """
        values, vectors = torch.linalg.eigh((self.ensemble+self.ensemble.T)/2)
        cutoff = 64*torch.finfo(values.dtype).eps*len(values)*values.abs().max()
        selected = values > cutoff
        states = self.basis @ vectors[:, selected]
        states = torch.stack([self.singlet(vector) for vector in states.unbind(1)], 1)
        basis = append_directions(self.basis.new_zeros(len(self.bits), 0),
                                  torch.cat((states, self.reference_vectors), 1), self.protocol["maximum_response_space_dimension"])
        overlap = basis.T @ self.basis
        ensemble = overlap @ self.ensemble @ overlap.T
        trace_loss = float((1-ensemble.trace()).abs())
        ensemble /= ensemble.trace()
        reconstruction = overlap.T @ ensemble @ overlap
        operator_error = float((reconstruction-self.ensemble).norm())
        images = torch.stack([self.action(vector) for vector in basis.unbind(1)], 1)
        matrix = basis.T @ images
        energy_error = float(((matrix*ensemble).sum()-(self.matrix*self.ensemble).sum()).abs())
        generators = self.links.project(basis)
        previous_density = self.links.density(self.ensemble, self.generators)
        density = self.links.density(ensemble, generators)
        density_error = float((density-previous_density).norm())
        reference_error = float((basis @ (basis.T @ self.reference_vectors)-self.reference_vectors).norm())
        valid = (trace_loss < 1e-10 and operator_error < 1e-10 and energy_error < 1e-8
                 and density_error < 1e-8 and reference_error < 1e-8)
        report = {"passed": valid, "old_dimension": self.basis.shape[1], "restarted_dimension": basis.shape[1],
                  "retained_ensemble_channels": int(selected.sum()), "discarded_trace": trace_loss,
                  "ensemble_operator_error": operator_error, "universal_energy_change_hartree": energy_error,
                  "density_change_norm": density_error, "input_retention_error": reference_error}
        if not valid:
            raise RuntimeError(f"Response-space restart did not retain the physical state: {report}")
        return {"basis": basis, "images": images, "reduced_ensemble": ensemble}, report

    def action(self, vector):
        # Energy is always the literal physical Hamiltonian. Spin restriction
        # belongs to the state space, never to an unmatched part of the energy.
        return self.physical(vector)

    @torch.no_grad()
    def purify_extension(self, prefix, candidate):
        new = candidate[:, prefix.shape[1]:]
        if not new.shape[1]: return prefix
        purified = torch.stack([self.singlet(vector) for vector in new.unbind(1)], 1)
        return append_directions(prefix, purified, self.protocol["maximum_response_space_dimension"])

    @torch.no_grad()
    def refresh(self):
        if self.projected_dimension != self.basis.shape[1]:
            added = self.basis[:, self.projected_dimension:]
            spin_error = max(float(self.singlet.spin_squared(vector).norm()) for vector in added.unbind(1))
            if spin_error > 1e-8:
                raise RuntimeError(f"Response space violates the singlet domain: {spin_error}")
        if self.images.shape[1] < self.basis.shape[1]:
            self.images = torch.cat((self.images, torch.stack([self.action(vector) for vector in self.basis[:, self.images.shape[1]:].unbind(1)], 1)), 1)
        if self.projected_dimension != self.basis.shape[1]:
            matrix = self.basis.T @ self.images
            if float((matrix-matrix.T).norm()) > 1e-7:
                raise RuntimeError("Universal Hamiltonian projection lost Hermiticity")
            self.matrix = (matrix+matrix.T)/2
            self.generators = self.links.project(self.basis)
            self.reduced = self.links.reduced(self.moments, self.generators)
            if len(self.ensemble) != len(self.matrix):
                embedded = self.matrix.new_zeros(self.matrix.shape)
                embedded[:len(self.ensemble), :len(self.ensemble)] = self.ensemble
                self.ensemble = embedded
            self.projected_dimension = len(self.matrix)

    @torch.no_grad()
    def evaluate(self, auxiliary, maximum_updates=None, callback=None):
        maximum_updates = self.protocol["response_updates_per_call"] if maximum_updates is None else maximum_updates
        history = []
        for update in range(maximum_updates+1):
            self.refresh()
            solution = proximal_state(self.matrix, self.reduced, auxiliary, self.epsilon, self.ensemble,
                                       maximum=self.protocol["maximum_reduced_iterations"], gap_tolerance=self.protocol["reduced_gap_tolerance_hartree"])
            self.ensemble = solution["state"]
            populations, rotations = torch.linalg.eigh((self.ensemble+self.ensemble.T)/2)
            root = rotations*populations.clamp_min(0).sqrt()
            field = torch.einsum("a,aij->ij", -solution["gradient"], self.moments)
            effective_images = self.images+self.links.image(field, self.basis)
            residual = (effective_images-self.basis @ (self.basis.T @ effective_images)) @ root
            size = float(residual.norm())
            row = {"response_update": update, "dimension": len(self.matrix), "reduced_gap_hartree": solution["gap"],
                   "outside_weighted_stationarity_residual": size, "reduced_converged": solution["converged"]}
            history.append(row)
            if callback: callback(row)
            if (not solution["converged"] or size < self.protocol["outside_stationarity_tolerance_hartree"]
                    or update == maximum_updates):
                break
            original = self.basis
            updated, added, capacity = leading_completion(original, residual, self.protocol["stationarity_response_directions"],
                                                           self.protocol["maximum_response_space_dimension"])
            updated = self.purify_extension(original, updated)
            if capacity:
                row["response_capacity_reached"] = True
                break
            active = populations.argsort(descending=True)[:self.protocol["maximum_density_response_states"]]
            states = (original @ rotations[:, active])*populations[active].clamp_min(0).sqrt()
            responses = self.links.response_images(self.moments, states).permute(1, 0, 2).reshape(len(self.bits), -1)
            pure_prefix = updated
            updated, density_added, capacity = leading_completion(updated, responses, self.protocol["density_response_directions"],
                                                                   self.protocol["maximum_response_space_dimension"])
            updated = self.purify_extension(pure_prefix, updated)
            if capacity or updated.shape[1] == original.shape[1]:
                row["response_capacity_reached"] = capacity
                break
            row["added_stationarity_directions"], row["added_density_directions"] = added, density_added
            self.basis = updated
        density = self.links.density(self.ensemble, self.generators)
        return {**solution, "physical_density": density, "outside_stationarity_residual": size,
                "universal_energy": (self.ensemble*self.matrix).sum(), "response_history": history,
                "basis_dimension": len(self.matrix)}

    @torch.no_grad()
    def kinetic_search(self, auxiliary, initial):
        return proximal_state(self.kinetic, self.moments, auxiliary, self.epsilon, initial,
                               total=self.number, cap=2., maximum=self.protocol["maximum_reduced_iterations"],
                               gap_tolerance=self.protocol["reduced_gap_tolerance_hartree"])

    @torch.no_grad()
    def xc(self, auxiliary, interacting, kinetic):
        # This Hartree term is on the auxiliary coordinate. The returned
        # interacting and noninteracting densities remain positive ensembles.
        matrix = torch.eye(len(self.kinetic), device=self.kinetic.device)*(self.number/len(self.kinetic))
        matrix += torch.einsum("a,aij->ij", auxiliary, self.moments)
        hartree_matrix = torch.einsum("ijkl,kl->ij", self.eri, matrix)
        hartree = .5*(matrix*hartree_matrix).sum()
        energy = interacting["functional"]-kinetic["functional"]-hartree
        gradient = interacting["gradient"]-kinetic["gradient"]-torch.einsum("aij,ij->a", self.moments, hartree_matrix)
        return {"energy": energy, "gradient": gradient, "auxiliary_hartree": hartree,
                "auxiliary_hartree_gradient": torch.einsum("aij,ij->a", self.moments, hartree_matrix)}

    @torch.no_grad()
    def certify(self, interacting):
        generator = torch.Generator(device=self.kinetic.device).manual_seed(self.protocol["global_check_seed"])
        orbitals, _ = torch.linalg.qr(torch.randn(len(self.kinetic), self.number//2, device=self.kinetic.device, generator=generator), mode="reduced")
        _, vectors = torch.linalg.eigh(interacting["effective"])
        seed = self.basis @ vectors[:, 0]+.01*slater_vector(orbitals, torch.arange(self.number//2, device=self.kinetic.device), self.bits)
        field = torch.einsum("a,aij->ij", -interacting["gradient"], self.moments)
        ground = incremental_ritz(lambda value: self.singlet(self.action(value)+self.links.image(field, value)), seed,
                                   maximum=self.protocol["global_check_ritz_dimension"],
                                   tolerance=self.protocol["global_check_residual_tolerance"], projector=self.singlet)
        full_gap = float((self.ensemble*interacting["effective"]).sum()-ground["energy"])
        spin = max(float(self.singlet.spin_squared(vector).norm()) for vector in self.basis.unbind(1))
        return {"converged": ground["converged"], "state_residual": ground["residual"],
                "full_oracle_frank_wolfe_gap_hartree": full_gap, "spin_squared_basis_norm": spin,
                "passed": ground["converged"] and abs(full_gap) < self.protocol["global_gap_tolerance_hartree"] and spin < 1e-8,
                "limit": "Ritz lowest-state evidence, not a rigorous lower bound from an independent Hamiltonian implementation"}

    def checkpoint(self):
        return {"basis": self.basis, "images": self.images, "reduced_ensemble": self.ensemble}
