"""Conditional auxiliary-density search in physical tensor sections.

The objective uses only T,W,Q,N,S. Fixed surrounding tensors define a
conditional section, not the complete universal search. Within that section
the positive-state problem is convex and the scalar derivative is consistent.
Global tensor stationarity, context independence and chemical transfer remain
separate acceptance requirements. No global occupation vectors are used.
"""
import torch
from .gpu_fermion_mpo import FermionMPO
from .gpu_tensor_density import one_body_mpo, density_matrix
from .gpu_local_tensor import LocalSpinBasis, physical_local_space, pullback_operator
from .gpu_spin_response import SpinChain
from .gpu_purified_spin import purified_norm
from .gpu_density_resolvent import proximal_state


class UniversalTensorOperators:
    @torch.no_grad()
    def __init__(self, universal):
        if set(universal) != {"kinetic", "eri", "operators", "electrons", "twice_spin"}:
            raise ValueError("Universal tensor operators accept only T,W,Q,N,S; no external potentials, charges, labels or Fock bits")
        self.kinetic, self.eri, self.operators = (universal[key] for key in ("kinetic", "eri", "operators"))
        self.number, self.spin = universal["electrons"], universal["twice_spin"]
        self.sites, self.device = len(self.kinetic), self.kinetic.device
        if self.device.type != "cuda" or any(value.device != self.device or value.dtype != torch.float64 for value in (self.kinetic, self.eri, self.operators)):
            raise ValueError("Universal numerical tensors must share a CUDA float64 device")
        if not all(bool(torch.isfinite(value).all()) for value in (self.kinetic, self.eri, self.operators)):
            raise ValueError("Universal operators must be finite")
        identity = torch.eye(self.sites, device=self.device)
        if float((self.operators[0]-identity/identity.new_tensor(self.sites).sqrt()).norm()) > 1e-9:
            raise ValueError("The first density operator must be normalized fixed number")
        if float((self.operators[1:].diagonal(dim1=-2, dim2=-1).sum(-1)).norm()) > 1e-8:
            raise ValueError("Variable density coordinates must be traceless")
        gram = self.operators.flatten(1) @ self.operators.flatten(1).T
        if float((gram-torch.eye(len(gram), device=self.device)).norm()) > 1e-8:
            raise ValueError("Density coordinates must form an orthonormal represented spatial frame")
        self.moments = self.operators[1:]
        self.hamiltonian = FermionMPO.molecular(self.kinetic, self.eri)

    def section(self, context, center, protocol, callback=None, compiled=None, checkpoint_callback=None):
        return LocalTensorSection(self, context, center, protocol, callback, compiled, checkpoint_callback)


class LocalTensorSection:
    @torch.no_grad()
    def __init__(self, domain, context, center, protocol, callback=None, compiled=None, checkpoint_callback=None):
        if (len(context.cores), context.electrons, context.twice_spin) != (domain.sites, domain.number, domain.spin):
            raise ValueError("The physical context does not match the universal number/spin sector")
        self.domain, self.context, self.center, self.protocol = domain, context, center, protocol
        self.epsilon = protocol["epsilon_inverse_hartree"]
        if self.epsilon <= 0: raise ValueError("Auxiliary regularization must be positive")
        if compiled is None:
            self.space = physical_local_space(context, center,
                maximum_embedding_bytes=protocol["maximum_embedding_bytes"],
                relative_null_tolerance=protocol["local_metric_null_tolerance"])
        else:
            if (compiled["center"], compiled["electrons"], compiled["twice_spin"], compiled["sites"]) != (center, domain.number, domain.spin, domain.sites):
                raise ValueError("Compiled conditional section does not match its physical context")
            self.space = dict(compiled["space"])
            self.space["basis"] = LocalSpinBasis(context.labels[center], context.labels[center+1], domain.device)
        if callback:
            callback({"phase": "physical_local_space", **{key: value for key, value in self.space.items() if not isinstance(value, torch.Tensor) and key != "basis"}})
        rank = self.space["physical_dimension"]
        required = len(domain.operators)*rank*rank*8
        if required > protocol["maximum_local_operator_bytes"]:
            raise RuntimeError(f"Compiled local density operators exceed their declared cap: {required} bytes")
        self.compiled_density_bytes = required
        basis, transform = self.space["basis"], self.space["transform"]
        self.matrix = (pullback_operator(domain.hamiltonian, context, center, basis, transform,
            maximum_environment_bytes=protocol["maximum_environment_bytes"]) if compiled is None else compiled["matrix"])
        self.reduced = domain.kinetic.new_zeros(len(domain.moments), rank, rank)
        if compiled is None:
            fixed = pullback_operator(one_body_mpo(domain.operators[0]), context, center, basis, transform,
                maximum_environment_bytes=protocol["maximum_environment_bytes"])
            fixed_error = float((fixed-domain.number/domain.kinetic.new_tensor(domain.sites).sqrt()*torch.eye(rank, device=domain.device)).norm())
        else:
            fixed_error = compiled["fixed_number_error"]
        if fixed_error > 1e-7: raise RuntimeError(f"Local physical operators lost exact fixed number: {fixed_error}")
        self.fixed_number_error = fixed_error
        if compiled is None and checkpoint_callback:
            checkpoint_callback({"phase": "base", "center": center, "electrons": domain.number, "twice_spin": domain.spin, "sites": domain.sites,
                                 "space": {key: value for key, value in self.space.items() if key != "basis"},
                                 "matrix": self.matrix, "fixed_number_error": fixed_error})
        start = 0
        if compiled is not None:
            for chunk in compiled.get("density_chunks", []):
                if chunk["start"] != start: raise ValueError("Compiled local density chunks have a gap or overlap")
                self.reduced[start:chunk["stop"]] = chunk["values"]
                start = chunk["stop"]
        chunk_start = start
        for index in range(start, len(domain.moments)):
            moment = domain.moments[index]
            self.reduced[index] = pullback_operator(one_body_mpo(moment), context, center, basis, transform,
                maximum_environment_bytes=protocol["maximum_environment_bytes"])
            if callback and (index % 8 == 0 or index+1 == len(domain.moments)):
                callback({"phase": "local_density_operators", "center": center, "compiled": index+1,
                          "total": len(domain.moments), "physical_dimension": rank})
            if checkpoint_callback and (index+1-chunk_start >= 8 or index+1 == len(domain.moments)):
                checkpoint_callback({"phase": "density_chunk", "start": chunk_start, "stop": index+1,
                                     "values": self.reduced[chunk_start:index+1]})
                chunk_start = index+1
        self.ensemble = self.space["initial_ensemble"]

    @torch.no_grad()
    def evaluate(self, auxiliary, initial=None, callback=None, checkpoint=None, checkpoint_callback=None):
        if auxiliary.device != self.domain.device or auxiliary.shape != (len(self.domain.moments),):
            raise ValueError("Auxiliary coordinates must match the CUDA spatial frame")
        solved = proximal_state(self.matrix, self.reduced, auxiliary, self.epsilon,
            self.ensemble if initial is None else initial,
            maximum=self.protocol["maximum_reduced_iterations"], gap_tolerance=self.protocol["reduced_gap_tolerance_hartree"], callback=callback,
            checkpoint=checkpoint, checkpoint_callback=checkpoint_callback)
        return {**solved, "universal_energy": (solved["state"]*self.matrix).sum(),
                "conditional_context_fixed": True, "global_tensor_stationarity_certified": False,
                "full_hilbert_stationarity_certified": False}

    @torch.no_grad()
    def kinetic(self, auxiliary, initial):
        if self.domain.spin or self.domain.number % 2:
            raise ValueError("Matching kinetic/XC envelope is currently validated only for even singlets")
        return proximal_state(self.domain.kinetic, self.domain.moments, auxiliary, self.epsilon, initial,
            total=self.domain.number, cap=2., maximum=self.protocol["maximum_reduced_iterations"],
            gap_tolerance=self.protocol["reduced_gap_tolerance_hartree"])

    @torch.no_grad()
    def xc(self, auxiliary, interacting, kinetic):
        if not interacting["converged"] or not kinetic["converged"]:
            raise RuntimeError("No conditional XC result is accepted from an unconverged inner search")
        domain = self.domain
        matrix = torch.eye(domain.sites, device=domain.device)*(domain.number/domain.sites)
        matrix += torch.einsum("a,aij->ij", auxiliary, domain.moments)
        hartree_matrix = torch.einsum("ijkl,kl->ij", domain.eri, matrix)
        hartree = .5*(matrix*hartree_matrix).sum()
        return {"energy": interacting["functional"]-kinetic["functional"]-hartree,
                "gradient": interacting["gradient"]-kinetic["gradient"]-torch.einsum("aij,ij->a", domain.moments, hartree_matrix),
                "scope": "Fixed-context conditional scalar and derivative, not a complete universal tensor functional"}

    @torch.no_grad()
    def install(self, solved, verify=True):
        if not solved["converged"]: raise RuntimeError("Do not install an unconverged conditional state")
        population, rotation = torch.linalg.eigh((solved["state"]+solved["state"].T)/2)
        retained = population > population.max()*len(population)*64*torch.finfo(population.dtype).eps
        discarded = float(population[~retained].clamp_min(0).sum())
        if discarded > 1e-10: raise RuntimeError("Physical purification would discard material probability")
        parameters = self.space["transform"] @ (rotation[:, retained]*population[retained].clamp_min(0).sqrt())
        core = self.space["basis"].core(parameters)
        cores = list(self.context.cores); cores[self.center] = core
        result = SpinChain(cores, self.context.labels, self.context.electrons, self.context.twice_spin)
        norm = purified_norm(result.cores)
        check = {"discarded_probability": discarded, "new_local_ancilla": int(retained.sum()),
                 "norm_error": float((norm-1).abs()), "verification_performed": verify}
        if verify:
            density = density_matrix(result)
            moments = torch.einsum("aij,ij->a", self.domain.moments, density)
            energy = self.domain.hamiltonian.expectation(result.cores)
            check.update({"density_moment_error": float((moments-solved["moments"]).norm()),
                          "universal_energy_error_hartree": float((energy-solved["universal_energy"]).abs()),
                          "particle_number_error": float((density.trace()-self.domain.number).abs()),
                          "minimum_spatial_occupation": float(torch.linalg.eigvalsh(density).min()),
                          "maximum_spatial_occupation": float(torch.linalg.eigvalsh(density).max())})
            if (max(check["norm_error"], check["density_moment_error"], check["universal_energy_error_hartree"], check["particle_number_error"]) > 1e-7
                    or check["minimum_spatial_occupation"] < -1e-8 or check["maximum_spatial_occupation"] > 2+1e-8):
                raise RuntimeError(f"Installed physical tensor ensemble failed literal identities: {check}")
        return result, check
