"""Density-conditioned charge/spin blocks for our correlation-state research.

Local multiplets retain exact N,S and all magnetic partners. Their tensor
product is fragment closed and the unprojected fermionic Hamiltonian supplies
all charge/spin transfers. Current accuracy implementation uses a fixed-Sz
Fock coefficient vector; it is not yet a scalable global tensor solver.
"""
from collections import defaultdict
import torch
from .gpu_manybody_ops import fixed_sector, molecular_words, dense_operator, spin_words


def parity(value):
    value = value ^ (value >> 32)
    value = value ^ (value >> 16)
    value = value ^ (value >> 8)
    value = value ^ (value >> 4)
    value = value ^ (value >> 2)
    value = value ^ (value >> 1)
    return value & 1


@torch.no_grad()
def word_connections(words, coefficients, bra_bits, ket_bits):
    if not len(bra_bits) or not len(ket_bits):
        empty = ket_bits.new_zeros(0)
        return empty, empty, coefficients.new_zeros(0)
    values = ket_bits[None].expand(len(words), -1)
    factors = coefficients[:, None].expand_as(values).clone()
    for slot in range(words.shape[1]-1, -1, -1):
        location, spin, creation = words[:, slot].unbind(-1)
        active = location >= 0
        mode = (2*location+spin).clamp_min(0)
        occupied = ((values >> mode[:, None]) & 1).bool()
        allowed = torch.where(creation[:, None].bool(), ~occupied, occupied) | (~active[:, None])
        sign = 1-2*parity(values & ((1 << mode[:, None])-1))
        factors *= allowed*torch.where(active[:, None], sign, 1)
        values = values ^ torch.where(active[:, None], 1 << mode[:, None], 0)
    rows = torch.searchsorted(bra_bits, values.contiguous())
    safe = rows.clamp_max(len(bra_bits)-1)
    valid = (rows < len(bra_bits)) & (bra_bits[safe] == values) & (factors != 0)
    columns = torch.arange(len(ket_bits), device=ket_bits.device)[None].expand_as(values)
    return rows[valid], columns[valid], factors[valid]


@torch.no_grad()
def rectangular_words(words, coefficients, bra_bits, ket_bits):
    rows, columns, values = word_connections(words, coefficients, bra_bits, ket_bits)
    matrix = coefficients.new_zeros(len(bra_bits), len(ket_bits))
    matrix.view(-1).scatter_add_(0, rows*len(ket_bits)+columns, values)
    return matrix


@torch.no_grad()
def spin_lower_norm(vector, bits, sites, up, down):
    if not up or down >= sites:
        return vector.new_zeros(())
    target, _ = fixed_sector(sites, up-1, down+1, bits.device)
    words = torch.tensor([((i, 1, 1), (i, 0, 0)) for i in range(sites)], device=bits.device)
    coefficients = vector.new_ones(sites)
    rows, columns, factors = word_connections(words, coefficients, target, bits)
    result = vector.new_zeros(len(target))
    result.scatter_add_(0, rows, factors*vector[columns])
    return result.norm()


class SingletProjector:
    """Exact finite-sector polynomial in S^2; never construct a dense matrix."""
    def __init__(self, bits, sites, pairs):
        target, _ = fixed_sector(sites, pairs-1, pairs+1, bits.device)
        words = torch.tensor([((i, 1, 1), (i, 0, 0)) for i in range(sites)], device=bits.device)
        self.rows, self.columns, self.factors = word_connections(words, torch.ones(sites, device=bits.device), target, bits)
        self.source_size, self.target_size = len(bits), len(target)
        self.maximum_spin = min(pairs, sites-pairs)

    def spin_squared(self, vector):
        lowered = vector.new_zeros(self.target_size)
        lowered.scatter_add_(0, self.rows, self.factors*vector[self.columns])
        result = vector.new_zeros(self.source_size)
        result.scatter_add_(0, self.columns, self.factors*lowered[self.rows])
        return result

    def __call__(self, vector):
        for spin in range(self.maximum_spin, 0, -1):
            vector = vector-self.spin_squared(vector)/(spin*(spin+1))
        return vector


class SectorCache:
    def __init__(self, sites, device):
        self.sites, self.device, self.cache = sites, device, {}

    def bits(self, up, down):
        key = up, down
        if key not in self.cache:
            self.cache[key] = fixed_sector(self.sites, up, down, self.device)[0]
        return self.cache[key]


class ProductGroups:
    """Reorder global Fock amplitudes into local charge/Sz product matrices."""
    def __init__(self, bits, first, second, up, down):
        self.bits, self.first, self.second = bits, first, second
        self.up, self.down = up, down
        left = bits & ((1 << (2*first.sites))-1)
        right = bits >> (2*first.sites)
        sites = torch.arange(first.sites, device=bits.device)
        local_up = ((left[:, None] >> (2*sites[None])) & 1).sum(-1)
        local_down = ((left[:, None] >> (2*sites[None]+1)) & 1).sum(-1)
        self.groups = []
        for a_up in range(max(0, up-second.sites), min(first.sites, up)+1):
            for a_down in range(max(0, down-second.sites), min(first.sites, down)+1):
                b_up, b_down = up-a_up, down-a_down
                selected = torch.nonzero((local_up == a_up) & (local_down == a_down)).flatten()
                a_bits, b_bits = first.bits(a_up, a_down), second.bits(b_up, b_down)
                row = torch.searchsorted(a_bits, left[selected].contiguous())
                column = torch.searchsorted(b_bits, right[selected].contiguous())
                if len(selected) != len(a_bits)*len(b_bits):
                    raise RuntimeError("Product Fock ordering lost a charge sector")
                order = bits.new_empty(len(selected))
                order[row*len(b_bits)+column] = selected
                self.groups.append({"a": (a_up, a_down), "b": (b_up, b_down),
                                    "shape": (len(a_bits), len(b_bits)), "order": order})

    def matrices(self, vector, side=0):
        result = {}
        for group in self.groups:
            matrix = vector[group["order"]].reshape(group["shape"])
            result[group["a"] if side == 0 else group["b"]] = matrix if side == 0 else matrix.T
        return result


def allowed_multiplets(first_sites, second_sites, electrons):
    result = [set(), set()]
    for first_number in range(max(0, electrons-2*second_sites), min(2*first_sites, electrons)+1):
        second_number = electrons-first_number
        maximum = min(first_number, 2*first_sites-first_number, second_number, 2*second_sites-second_number)
        for spin in range(first_number % 2, maximum+1, 2):
            result[0].add((first_number, spin)); result[1].add((second_number, spin))
    return [sorted(value) for value in result]


@torch.no_grad()
def append_orthogonal(basis, candidates, tolerance=1e-12):
    for value in candidates.unbind(-1):
        for _ in range(2):
            value = value-basis @ (basis.T @ value)
        length = value.norm()
        if float(length) > tolerance:
            basis = torch.cat((basis, (value/length)[:, None]), -1)
    return basis


class MultipletBlock:
    @classmethod
    def from_checkpoint(cls, h, eri, cache, checkpoint):
        result = object.__new__(cls)
        result.h, result.eri, result.cache = h, eri, cache
        result.entries, result.sector_cache = {}, {}
        result.lower_words = torch.tensor([((i, 1, 1), (i, 0, 0)) for i in range(cache.sites)], device=cache.device)
        result.lower_coefficients = h.new_ones(cache.sites)
        for saved in checkpoint:
            key = saved["electrons"], saved["twice_spin"]
            result.entries[key] = {"spin_basis": saved["spin_basis"], "coordinates": saved["coordinates"],
                                   "source_support_added": saved["source_support_added"], "source_energies": saved["source_energies"], "states": {}}
            result.refresh(*key)
        return result

    @torch.no_grad()
    def __init__(self, embedded_h, eri, cache, multiplets, reference_matrices, initial=16):
        self.h, self.eri, self.cache = embedded_h, eri, cache
        self.entries, self.sector_cache = {}, {}
        self.words, self.coefficients = molecular_words(embedded_h, eri)
        self.spin_words, self.spin_coefficients = spin_words(cache.sites, cache.device)
        self.lower_words = torch.tensor([((i, 1, 1), (i, 0, 0)) for i in range(cache.sites)], device=cache.device)
        self.lower_coefficients = embedded_h.new_ones(cache.sites)
        for number, twice_spin in multiplets:
            up, down = (number+twice_spin)//2, (number-twice_spin)//2
            bits = cache.bits(up, down)
            if number == 0:
                spin_basis = embedded_h.new_ones(1, 1)
                energies = embedded_h.new_zeros(1)
                energy_vectors = embedded_h.new_ones(1, 1)
            else:
                matrix = dense_operator(self.words, self.coefficients, bits, cache.sites)
                if twice_spin == min(number, 2*cache.sites-number):
                    spin_basis = torch.eye(len(bits), device=cache.device)
                else:
                    spin_matrix = dense_operator(self.spin_words, self.spin_coefficients, bits, cache.sites)
                    values, vectors = torch.linalg.eigh(spin_matrix)
                    spin_basis = vectors[:, (values-twice_spin/2).abs() < 1e-8]
                if not spin_basis.shape[1]:
                    raise RuntimeError(f"Missing physical local multiplet N={number},2S={twice_spin}")
                projected = spin_basis.T @ matrix @ spin_basis
                energies, energy_vectors = torch.linalg.eigh((projected+projected.T)/2)
            count = min(initial, len(energies))
            boundary = energies[count-1]
            selected_energy = energies <= boundary+max(1e-10, float(energies.abs().max())*1e-10)
            retained = energy_vectors[:, selected_energy]
            # Exact local Schmidt support of the stated input determinant is
            # added to the energy-selected states, so reference retention is a
            # property of the family rather than a hoped-for optimizer outcome.
            source = reference_matrices.get((up, down))
            support_count = 0
            if source is not None:
                source = spin_basis.T @ source
                vectors, singular, _ = torch.linalg.svd(source, full_matrices=False)
                selected = singular > max(1e-14, float(singular[0])*1e-12)
                before = retained.shape[1]
                retained = append_orthogonal(retained, vectors[:, selected])
                support_count = retained.shape[1]-before
            self.entries[number, twice_spin] = {"spin_basis": spin_basis, "coordinates": retained,
                                                 "source_support_added": support_count, "source_energies": energies,
                                                 "states": {}}
            self.refresh(number, twice_spin)

    @torch.no_grad()
    def refresh(self, number, twice_spin):
        entry = self.entries[number, twice_spin]
        state = entry["spin_basis"] @ entry["coordinates"]
        magnetic = twice_spin
        entry["states"] = {magnetic: state}
        while magnetic > -twice_spin:
            up, down = (number+magnetic)//2, (number-magnetic)//2
            source = self.cache.bits(up, down)
            target = self.cache.bits(up-1, down+1)
            matrix = rectangular_words(self.lower_words, self.lower_coefficients, target, source)
            normalization = state.new_tensor((twice_spin+magnetic)*(twice_spin-magnetic+2)).sqrt()/2
            state = matrix @ state/normalization
            magnetic -= 2
            entry["states"][magnetic] = state
        self.sector_cache.clear()

    def sector(self, up, down):
        key = up, down
        if key not in self.sector_cache:
            number, magnetic = up+down, up-down
            states = [entry["states"][magnetic] for (n, spin), entry in self.entries.items()
                      if n == number and magnetic in entry["states"]]
            self.sector_cache[key] = torch.cat(states, -1) if states else self.h.new_zeros(len(self.cache.bits(up, down)), 0)
        return self.sector_cache[key]

    @torch.no_grad()
    def enrich(self, residual_matrices, directions=2, threshold=1e-8):
        added = 0
        for (number, twice_spin), entry in self.entries.items():
            key = (number+twice_spin)//2, (number-twice_spin)//2
            if key not in residual_matrices:
                continue
            source = entry["spin_basis"].T @ residual_matrices[key]
            basis = entry["coordinates"]
            source = source-basis @ (basis.T @ source)
            vectors, singular, _ = torch.linalg.svd(source, full_matrices=False)
            eligible = torch.nonzero(singular > threshold).flatten()
            if len(eligible):
                boundary = singular[eligible[min(directions, len(eligible))-1]]
                chosen = torch.nonzero((singular > threshold) & (singular >= boundary-max(1e-12, float(boundary)*1e-7))).flatten()
            else:
                chosen = eligible
            before = basis.shape[1]
            entry["coordinates"] = append_orthogonal(basis, vectors[:, chosen])
            added += entry["coordinates"].shape[1]-before
            if entry["coordinates"].shape[1] > before:
                self.refresh(number, twice_spin)
        return added

    def summary(self):
        return [{"electrons": number, "twice_spin": spin, "multiplets": entry["coordinates"].shape[1],
                 "complete_multiplets": entry["spin_basis"].shape[1], "input_support_added": entry["source_support_added"]}
                for (number, spin), entry in self.entries.items()]


class ChargeSpinProductProjector:
    def __init__(self, groups, first, second):
        self.groups, self.first, self.second = groups, first, second

    def __call__(self, vector):
        result = torch.zeros_like(vector)
        for group in self.groups.groups:
            left = self.first.sector(*group["a"])
            right = self.second.sector(*group["b"])
            if not left.shape[1] or not right.shape[1]:
                continue
            matrix = vector[group["order"]].reshape(group["shape"])
            coefficient = left.T @ matrix @ right
            result[group["order"]] = (left @ coefficient @ right.T).flatten()
        return result

    def dimension(self):
        return sum(self.first.sector(*group["a"]).shape[1]*self.second.sector(*group["b"]).shape[1]
                   for group in self.groups.groups)

    def charge_weights(self, vector):
        result = defaultdict(lambda: vector.new_zeros(()))
        for group in self.groups.groups:
            result[sum(group["a"])] += vector[group["order"]].square().sum()
        return {str(key): float(value) for key, value in sorted(result.items())}
