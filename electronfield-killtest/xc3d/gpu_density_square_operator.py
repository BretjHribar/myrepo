"""Literal density-square Hamiltonian without four-orbital word enumeration.

E(A)=sum_ij,s A_ij c^dagger_is c_js. For symmetric real factors L,
W=1/2 sum_P [E(L_P)^2-E(L_P L_P)]. Products act in the full physical
space before any local pullback. Squaring a projected E would be wrong.
These are established operator identities, not a novelty/accuracy claim.
"""
import torch
from .gpu_manybody_ops import local_fermions
from .gpu_fermion_mpo import FermionMPO
from .gpu_purified_spin import purified_norm
from .gpu_two_site_tensor import pair_problem


def compact_block(operator, inputs, outputs, values):
    input_, columns = torch.unique(inputs, return_inverse=True)
    output, rows = torch.unique(outputs, return_inverse=True)
    matrix = torch.sparse_coo_tensor(torch.stack((rows, columns)), values,
                                     (len(output), len(input_)), device=values.device).coalesce()
    return operator, input_, output, matrix


class OneBodyMPO(FermionMPO):
    """Direct Jordan-Wigner automaton: O(n^2) coefficients, O(n) channels."""
    def __init__(self, matrix, row_chunk=64, maximum_host_metadata_bytes=1_000_000_000):
        if matrix.device.type != "cuda" or matrix.dtype != torch.float64 or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("A square CUDA float64 one-body matrix is required")
        if not bool(torch.isfinite(matrix).all()) or float((matrix-matrix.T).abs().max()) > 1e-10:
            raise ValueError("The one-body Hamiltonian must be finite and real symmetric")
        self.sites, self.row_chunk = len(matrix), row_chunk
        # The explicit endpoint-label implementation creates exactly 2*n^2
        # bond labels. This conservative dictionary/tuple budget prevents a
        # large target case from exhausting host RAM before CUDA guards run.
        self.estimated_host_metadata_bytes = 2*self.sites**2*256
        if self.estimated_host_metadata_bytes > maximum_host_metadata_bytes:
            raise MemoryError("Explicit one-body endpoint metadata exceeds its host budget; use an implicit channel representation")
        identity, parity, fermions = local_fermions(matrix.device)
        number = torch.diag(matrix.new_tensor([0., 1., 1., 2.]))
        openings = [fermions[spin+2] @ parity for spin in range(2)]+[parity @ fermions[spin] for spin in range(2)]
        closings = [fermions[spin] for spin in range(2)]+[fermions[spin+2] for spin in range(2)]
        self.operators = torch.stack([identity, parity, number]+openings+closings)
        idle, done = (-1,), (-2,)
        bonds = [[idle]]+[[idle, done]+[(endpoint, channel) for endpoint in range(cut, self.sites) for channel in range(4)]
                                     for cut in range(1, self.sites)]+[[done]]
        positions = [{label: i for i, label in enumerate(bond)} for bond in bonds]
        self.bond_dimensions = [len(bond) for bond in bonds]
        self.blocks, self.edge_count = [], 0
        self.coefficient = matrix.flatten()
        self.input_word_count = self.canonical_word_count = 0
        for site in range(self.sites):
            entries = {}
            def edge(operator, source, target, value):
                row = entries.setdefault(operator, {"input": [], "output": [], "values": []})
                row["input"].append(positions[site][source]); row["output"].append(positions[site+1][target]); row["values"].append(value)
            one = matrix.new_ones(())
            if site+1 < self.sites: edge(0, idle, idle, one)
            if site: edge(0, done, done, one)
            edge(2, idle, done, matrix[site, site])
            for endpoint in range(site+1, self.sites):
                for channel in range(4):
                    label = endpoint, channel
                    coefficient = matrix[site, endpoint] if channel < 2 else matrix[endpoint, site]
                    edge(3+channel, idle, label, coefficient)
                    if site: edge(1, label, label, one)
            if site:
                for channel in range(4): edge(7+channel, (site, channel), done, one)
            blocks = []
            for operator, row in entries.items():
                block = compact_block(operator, torch.tensor(row["input"], device=matrix.device),
                                      torch.tensor(row["output"], device=matrix.device), torch.stack(row["values"]))
                blocks.append(block); self.edge_count += block[3]._nnz()
            self.blocks.append(blocks)


class ProductMPO(FermionMPO):
    """Exact operator product; no intermediate physical projection."""
    def __init__(self, first, second, maximum_storage_bytes=500_000_000):
        if first.sites != second.sites or first.operators.device != second.operators.device:
            raise ValueError("Operator products must share physical sites and CUDA device")
        self.sites, self.row_chunk = first.sites, min(first.row_chunk, second.row_chunk)
        if isinstance(first, OneBodyMPO) and isinstance(second, OneBodyMPO):
            # Parity-propagation x parity-propagation alone retains 16*k^2
            # distinct nonzeros for k future endpoints at each interior site.
            # Each sparse nonzero needs two int64 indices and one float64.
            # This is a lower bound on the present expanded implementation,
            # not a lower bound on all possible factor contractions.
            n = self.sites
            minimum_bytes = 24*(16*(n-2)*(n-1)*(2*n-3)//6) if n > 2 else 0
            if minimum_bytes > maximum_storage_bytes:
                raise MemoryError(f"Expanded density-square propagation alone exceeds storage cap: {minimum_bytes}; use an implicit contraction")
        self.bond_dimensions = [a*b for a, b in zip(first.bond_dimensions, second.bond_dimensions)]
        self.blocks, operators = [], []
        self.edge_count = self.input_word_count = self.canonical_word_count = 0
        self.coefficient = first.operators.new_empty(0)
        estimated_storage = 0
        for site in range(self.sites):
            blocks = []
            for op_a, input_a, output_a, mat_a in first.blocks[site]:
                for op_b, input_b, output_b, mat_b in second.blocks[site]:
                    physical = first.operators[op_a] @ second.operators[op_b]
                    if not bool((physical != 0).any()): continue
                    requested = mat_a._nnz()*mat_b._nnz()
                    estimated_storage += requested*56
                    if estimated_storage > maximum_storage_bytes:
                        raise RuntimeError("Exact density-square product exceeds its declared sparse storage cap")
                    row_a, col_a = mat_a.indices(); row_b, col_b = mat_b.indices()
                    inputs = (input_a[col_a, None]*second.bond_dimensions[site]+input_b[col_b][None]).flatten()
                    outputs = (output_a[row_a, None]*second.bond_dimensions[site+1]+output_b[row_b][None]).flatten()
                    values = (mat_a.values()[:, None]*mat_b.values()[None]).flatten()
                    block = compact_block(len(operators), inputs, outputs, values)
                    blocks.append(block); operators.append(physical); self.edge_count += block[3]._nnz()
            self.blocks.append(blocks)
        self.operators = torch.stack(operators)


class DensitySquareOperator:
    """Stream one exact density factor/product at a time, preserving N and spin.

This first implementation accepts supplied CUDA factors. It does not yet
provide a large-basis GPU integral-factor producer or certify model accuracy.
"""
    def __init__(self, h, factors, maximum_product_storage_bytes=500_000_000):
        if h.device.type != "cuda" or h.dtype != torch.float64 or factors.device != h.device or factors.dtype != h.dtype:
            raise ValueError("Density-square coefficients must be CUDA float64")
        if factors.ndim != 3 or factors.shape[1:] != h.shape or not bool(torch.isfinite(factors).all()):
            raise ValueError("One finite real symmetric matrix per Coulomb factor is required")
        if float((factors-factors.transpose(-1, -2)).abs().max()) > 1e-10:
            raise ValueError("Real Coulomb factors must be symmetric")
        self.sites, self.factors = len(h), factors
        self.maximum_product_storage_bytes = maximum_product_storage_bytes
        correction = torch.zeros_like(h)
        for factor in factors: correction += factor @ factor
        self.one_body = OneBodyMPO(h-.5*correction)

    def terms(self):
        yield 1., self.one_body
        for factor in self.factors:
            density = OneBodyMPO(factor)
            yield .5, ProductMPO(density, density, self.maximum_product_storage_bytes)

    def expectation(self, cores):
        value = cores[0].new_zeros(())
        for weight, operator in self.terms(): value += weight*operator.contract(cores)
        return value/purified_norm(cores)

    def problem(self, operator, state, center, maximum_environment_bytes):
        if operator is not self: raise ValueError("Wrong density-square operator")
        base = pair_problem(self.one_body, state, center, maximum_environment_bytes)
        def action(value):
            result = value.new_zeros(value.shape)
            for weight, term in self.terms():
                local = pair_problem(term, state, center, maximum_environment_bytes)
                if local["basis"].paths != base["basis"].paths:
                    raise RuntimeError("A density factor changed the physical local section")
                result += weight*local["action"](value)
            return result
        return {**base, "action": action, "full_operator_product_before_pullback": True,
                "density_factors_streamed": len(self.factors), "four_orbital_words_enumerated": False}
