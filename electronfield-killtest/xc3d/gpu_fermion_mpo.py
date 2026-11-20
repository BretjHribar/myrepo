"""Exact sparse fermion-word automaton for physical tensor contractions.

The automaton/MPO construction is established mathematics. We implement it
here as a scope primitive for our physical density-response state. It does
not use a full occupation vector, discard integral terms, or change H to
a projected Hamiltonian. Operator coefficients and contractions stay CUDA.
Integer word/graph enumeration on the host is shape/label bookkeeping only.
"""
from collections import defaultdict
import torch
from .gpu_manybody_ops import local_fermions, molecular_words
from .gpu_purified_spin import purified_norm


def _local_catalog(device):
    identity, parity, fermions = local_fermions(device)
    letters = torch.cat((identity[None], parity[None], fermions))
    codes = torch.arange(6**4, device=device)
    products = identity.expand(len(codes), -1, -1).clone()
    for slot in range(4):
        products = products @ letters[(codes//6**slot) % 6]
    flat = products.flatten(1)
    first = flat.abs().argmax(1)
    sign = flat.gather(1, first[:, None]).squeeze(1)
    normalized = products/torch.where(sign != 0, sign, 1)[:, None, None]
    operators, lookup = torch.unique(normalized.flatten(1), dim=0, return_inverse=True)
    operators = operators.reshape(-1, 4, 4)
    conjugated = parity @ operators @ parity
    even = (conjugated-operators).flatten(1).norm(dim=1) < 1e-12
    odd = (conjugated+operators).flatten(1).norm(dim=1) < 1e-12
    if not bool((even | odd).all()):
        raise RuntimeError("A local operator lacks a definite fermion parity")
    return operators, lookup, sign, (~even).long(), int(lookup[0]), int(lookup[1])


def _canonical_events(words, coefficient, sites, lookup, sign, alphabet):
    if words.shape[1:] != (4, 3):
        raise ValueError("Four padded fermion slots are required")
    location = words[:, :, 0]
    ordered = torch.sort(torch.where(location >= 0, location, sites), dim=1).values
    valid = (ordered < sites)
    valid[:, 1:] &= ordered[:, 1:] != ordered[:, :-1]
    rank = valid.cumsum(1)-1
    row = torch.arange(len(words), device=words.device)[:, None].expand(-1, 4)
    positions = torch.full_like(location, -1)
    positions[row[valid], rank[valid]] = ordered[valid]
    local_code = torch.zeros_like(positions)
    for slot in range(4):
        where, spin, creation = words[:, slot].unbind(1)
        factor = torch.where(where[:, None] > positions, 1, 0)
        factor = torch.where(where[:, None] == positions, (2+spin+2*creation)[:, None], factor)
        factor = torch.where((positions >= 0) & (where[:, None] >= 0), factor, 0)
        local_code += factor*6**slot
    local_operator = lookup[local_code]
    coefficient = coefficient*sign[local_code].prod(1)
    base = sites*alphabet+1
    if base**4 >= 2**63:
        raise ValueError("Integer operator-word key would overflow")
    digit = torch.where(positions >= 0, positions*alphabet+local_operator+1, 0)
    key = sum(digit[:, slot]*base**slot for slot in range(4))
    unique, inverse = torch.unique(key, return_inverse=True)
    combined = coefficient.new_zeros(len(unique)).index_add(0, inverse, coefficient)
    retained = combined != 0
    return unique[retained], combined[retained], base


def _transfer(environment, core, operator, bra=None):
    # core: left, physical, local spinless ancilla, right.
    first = torch.einsum("xab,bckt->xackt", environment, core)
    first = torch.einsum("dc,xackt->xadkt", operator, first)
    return torch.einsum("adkr,xadkt->xrt", (core if bra is None else bra).conj(), first)


class FermionMPO:
    def __init__(self, words, coefficient, sites, row_chunk=64):
        if coefficient.device.type != "cuda" or coefficient.dtype != torch.float64 or words.device != coefficient.device:
            raise ValueError("CUDA float64 coefficients and CUDA word metadata are required")
        if sites < 1 or row_chunk < 1:
            raise ValueError("Positive site and contraction chunk counts are required")
        if not bool((((words[:, :, 0] >= 0).sum(1) % 2) == 0).all()):
            raise ValueError("This Hamiltonian automaton requires even fermion words")
        self.sites, self.row_chunk = sites, row_chunk
        operators, lookup, sign, parity, identity, parity_operator = _local_catalog(coefficient.device)
        keys, coefficient, base = _canonical_events(words, coefficient, sites, lookup, sign, len(operators))
        self.operators = operators
        self.canonical_word_count = len(keys)
        self.input_word_count = len(words)
        self.coefficient = coefficient
        operator_parity = parity.tolist()  # Discrete operator labels, no coefficients.
        idle, done = (0,), (1,)
        ranges = {idle: (0, sites-1), done: (1, sites)}
        state_parity = {idle: 0, done: 0}
        unit_edges = set()
        weighted = []

        def event(digit):
            return divmod(digit-1, len(operators))

        def activate(state, cut):
            previous = ranges.get(state, (cut, cut))
            ranges[state] = min(previous[0], cut), max(previous[1], cut)

        def edge(site, source, target, operator, index=None):
            activate(source, site); activate(target, site+1)
            value = (site, source, target, operator)
            if index is None:
                unit_edges.add(value)
            else:
                weighted.append((*value, index+1))

        for index, encoded in enumerate(keys.tolist()):
            digits = []
            for slot in range(4):
                digit = (encoded//base**slot) % base
                if digit:
                    digits.append(digit)
            if not digits:
                raise ValueError("An empty operator word needs an explicit scalar constant")
            events = [event(value) for value in digits]
            if len(events) == 1:
                edge(events[0][0], idle, done, events[0][1], index)
                continue
            first = (2, digits[0])
            state_parity[first] = operator_parity[events[0][1]]
            edge(events[0][0], idle, first, events[0][1])
            if len(events) == 2:
                edge(events[1][0], first, done, events[1][1], index)
                continue
            last = (4, digits[-1])
            state_parity[last] = operator_parity[events[-1][1]]
            edge(events[-1][0], last, done, events[-1][1])
            if len(events) == 3:
                edge(events[1][0], first, last, events[1][1], index)
            else:
                second = (3, digits[0], digits[1])
                state_parity[second] = operator_parity[events[0][1]] ^ operator_parity[events[1][1]]
                edge(events[1][0], first, second, events[1][1])
                edge(events[2][0], second, last, events[2][1], index)

        bonds = [sorted(state for state, (low, high) in ranges.items() if low <= cut <= high)
                 for cut in range(sites+1)]
        self.bond_dimensions = [len(value) for value in bonds]
        indices = [{state: index for index, state in enumerate(states)} for states in bonds]
        for site in range(sites):
            for state in indices[site].keys() & indices[site+1].keys():
                operator = parity_operator if state_parity[state] else identity
                unit_edges.add((site, state, state, operator))
        site_edges = [defaultdict(list) for _ in range(sites)]
        for site, source, target, operator, value in [(*edge, 0) for edge in sorted(unit_edges)]+weighted:
            site_edges[site][operator].append((indices[site+1][target], indices[site][source], value))
        values = torch.cat((coefficient.new_ones(1), coefficient))
        self.blocks = []
        self.edge_count = 0
        for entries in site_edges:
            blocks = []
            for operator, edges in sorted(entries.items()):
                metadata = torch.tensor(edges, dtype=torch.long, device=coefficient.device)
                output, out_inverse = torch.unique(metadata[:, 0], return_inverse=True)
                input_, in_inverse = torch.unique(metadata[:, 1], return_inverse=True)
                matrix = torch.sparse_coo_tensor(torch.stack((out_inverse, in_inverse)), values[metadata[:, 2]],
                                                 (len(output), len(input_)), device=coefficient.device).coalesce()
                blocks.append((operator, input_, output, matrix))
                self.edge_count += matrix._nnz()
            self.blocks.append(blocks)

    @classmethod
    def molecular(cls, h, eri, row_chunk=64):
        words, coefficient = molecular_words(h, eri, threshold=0.)
        return cls(words, coefficient, len(h), row_chunk)

    def contract(self, cores, bra=None):
        if len(cores) != self.sites or any(value.device != self.operators.device or value.ndim != 4 for value in cores):
            raise ValueError("One CUDA purified core per spatial site is required")
        environment = cores[0].new_ones(1, 1, 1)
        bra = cores if bra is None else bra
        if len(bra) != self.sites or any(first.shape[2] != second.shape[2] for first, second in zip(cores, bra)):
            raise ValueError("Bra and ket must share physical sites and local ancillas")
        for site, core in enumerate(cores):
            outputs, labels = [], []
            for operator, input_, output, matrix in self.blocks[site]:
                selected = environment.index_select(0, input_)
                merged = torch.sparse.mm(matrix, selected.flatten(1)).reshape(len(output), bra[site].shape[0], core.shape[0])
                transferred = []
                for start in range(0, len(output), self.row_chunk):
                    transferred.append(_transfer(merged[start:start+self.row_chunk], core, self.operators[operator], bra[site]))
                outputs.append(torch.cat(transferred)); labels.append(output)
            fresh = core.new_zeros(self.bond_dimensions[site+1], bra[site].shape[-1], core.shape[-1])
            environment = fresh.index_add(0, torch.cat(labels), torch.cat(outputs)) if outputs else fresh
        return environment[0, 0, 0].real

    def local_image(self, site, carry, core, maximum_raw_bytes):
        """One uncompressed physical site of H|state>, no global vector.

        carry: retained left physical basis, left MPO channel, source bond.
        The output's right axis is (right MPO channel, source right bond).
        A caller must retain spin multiplets when compressing that axis.
        """
        shape = (carry.shape[0], 4, core.shape[2], self.bond_dimensions[site+1], core.shape[-1])
        elements = 1
        for value in shape: elements *= value
        raw_bytes = elements*core.element_size()
        if raw_bytes > maximum_raw_bytes:
            raise RuntimeError(f"Physical response intermediate exceeds its declared memory cap: {raw_bytes} bytes")
        outputs, labels = [], []
        for operator, input_, output, matrix in self.blocks[site]:
            selected = carry.index_select(1, input_).permute(1, 0, 2)
            merged = torch.sparse.mm(matrix, selected.flatten(1)).reshape(len(output), carry.shape[0], core.shape[0]).permute(1, 0, 2)
            part = torch.einsum("xul,lckr->xuckr", merged, core)
            part = torch.einsum("dc,xuckr->xdkur", self.operators[operator], part)
            outputs.append(part); labels.append(output)
        raw = core.new_zeros(shape)
        if outputs:
            raw = raw.index_add(3, torch.cat(labels), torch.cat(outputs, dim=3))
        return raw.flatten(3), raw_bytes

    def expectation(self, cores):
        return self.contract(cores)/purified_norm(cores)

    def summary(self):
        size = self.operators.numel()*self.operators.element_size()+self.coefficient.numel()*self.coefficient.element_size()
        for blocks in self.blocks:
            for _, input_, output, matrix in blocks:
                size += (input_.numel()+output.numel()+matrix.indices().numel())*8+matrix.values().numel()*8
        return {"sites": self.sites, "input_fermion_words": self.input_word_count,
                "canonical_product_words": self.canonical_word_count, "mpo_bond_dimensions": self.bond_dimensions,
                "sparse_edges": self.edge_count, "cuda_operator_storage_bytes": size,
                "full_occupation_vector_allocated": False, "integral_truncation_threshold": 0.}
