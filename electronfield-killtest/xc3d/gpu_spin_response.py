"""SU(2)/number-preserving tensor response, without global occupations.

The left isometry at every cut retains complete spin multiplets. A local
spinless ancilla carries mixture labels. Direct SVD is used rather than
squaring small singular values in a Gram matrix. Local discarded norms are
indicators, not a certified error bound for the complete Hamiltonian image.
This is a scope primitive, not a new tensor-network theorem or chemistry model.
"""
from dataclasses import dataclass
import torch
from .gpu_spin_chain import quantum_bonds, transitions
from .gpu_purified_spin import purified_norm


@dataclass
class SpinChain:
    cores: list
    labels: list
    electrons: int
    twice_spin: int

    @classmethod
    def from_state(cls, state, reference=False):
        return cls(state.cores(reference), state.bond_basis, state.electrons, state.twice_spin)

    def normalized(self):
        total = purified_norm(self.cores)
        if not bool(torch.isfinite(total)) or float(total) <= 1e-28:
            raise RuntimeError("Response has zero or nonfinite physical norm")
        return SpinChain([self.cores[0]/total.sqrt(), *self.cores[1:]], self.labels, self.electrons, self.twice_spin)


def overlap(first, second):
    if len(first.cores) != len(second.cores):
        raise ValueError("Tensor chains have different site counts")
    environment = first.cores[0].new_ones(1, 1)
    for bra, ket in zip(first.cores, second.cores):
        if bra.shape[2] != ket.shape[2]:
            raise ValueError("Tensor chains have incompatible ancillas")
        environment = torch.einsum("ij,idar,jdas->rs", environment, bra.conj(), ket)
    return environment[0, 0].real


def closed_shell_reference(sites, electrons, device):
    return product_reference(sites, electrons, 0, device)


def product_reference(sites, electrons, twice_spin, device):
    if (electrons-twice_spin) % 2 or min(electrons, twice_spin) < 0 or electrons > 2*sites or twice_spin > min(electrons, 2*sites-electrons):
        raise ValueError("The requested highest-weight determinant does not fit these sites")
    cores, labels = [], [[(0, 0, 0, 0)]]
    number, spin = 0, 0
    pairs = (electrons-twice_spin)//2
    for site in range(sites):
        occupied, unpaired = site < pairs, pairs <= site < pairs+twice_spin
        core = torch.zeros(1, 4, 1, 1, device=device)
        core[0, 3 if occupied else 1 if unpaired else 0, 0, 0] = 1.
        cores.append(core); number += 2 if occupied else 1 if unpaired else 0
        spin += 1 if unpaired else 0
        labels.append([(number, spin, 0, spin)])
    return SpinChain(cores, labels, electrons, twice_spin)


def superposition(states, coefficients):
    """Exact coherent sum; no fixed global determinant budget or truncation."""
    first = states[0]
    if any((len(value.cores), value.electrons, value.twice_spin) !=
           (len(first.cores), first.electrons, first.twice_spin) for value in states):
        raise ValueError("Superposed states must share sites and the physical sector")
    if any([core.shape[2] for core in value.cores] != [core.shape[2] for core in first.cores] for value in states):
        raise ValueError("Superposed purifications must share local ancilla spaces")
    labels = [first.labels[0]]
    for cut in range(1, len(first.cores)):
        combined, offsets = [], {}
        for state in states:
            counts = {}
            for number, spin, channel, magnetic in state.labels[cut]:
                key = number, spin
                combined.append((number, spin, offsets.get(key, 0)+channel, magnetic))
                counts[key] = max(counts.get(key, 0), channel+1)
            for key, count in counts.items(): offsets[key] = offsets.get(key, 0)+count
        labels.append(combined)
    labels.append(first.labels[-1])
    cores = []
    for site in range(len(first.cores)):
        if len(first.cores) == 1:
            cores.append(sum(coefficient*state.cores[0] for coefficient, state in zip(coefficients, states)))
        elif site == 0:
            cores.append(torch.cat([coefficient*state.cores[site] for coefficient, state in zip(coefficients, states)], dim=-1))
        elif site+1 == len(first.cores):
            cores.append(torch.cat([state.cores[site] for state in states], dim=0))
        else:
            prototype = first.cores[site]
            core = prototype.new_zeros(sum(state.cores[site].shape[0] for state in states), 4, prototype.shape[2],
                                        sum(state.cores[site].shape[-1] for state in states))
            left, right = 0, 0
            for state in states:
                value = state.cores[site]
                core[left:left+value.shape[0], :, :, right:right+value.shape[-1]] = value
                left += value.shape[0]; right += value.shape[-1]
            cores.append(core)
    return SpinChain(cores, labels, first.electrons, first.twice_spin)


def _coupled_rows(left, ancilla, allowed, device, final_m=None):
    """Discrete CG maps from (left,physical,ancilla) to prefix irreps."""
    groups, lookup = {}, {}
    for index, (number, spin, channel, magnetic) in enumerate(left):
        lookup[number, spin, channel, magnetic] = index
        for physical, next_number, next_spin in transitions(number, spin):
            if (next_number, next_spin) not in allowed:
                continue
            key = next_number, next_spin
            groups.setdefault(key, {})
            for local_ancilla in range(ancilla):
                groups[key].setdefault((number, spin, channel, physical, local_ancilla), len(groups[key]))
    result = []
    for (number, spin), channels in sorted(groups.items()):
        magnetic_values = [final_m] if final_m is not None else list(range(-spin, spin+1, 2))
        m_lookup = {value: index for index, value in enumerate(magnetic_values)}
        rows, components, numerators, denominators, signs = [], [], [], [], []
        for (old_number, old_spin, channel, physical, local_ancilla), multiplicity in channels.items():
            for magnetic in range(-old_spin, old_spin+1, 2):
                old = lookup.get((old_number, old_spin, channel, magnetic))
                if old is None: continue
                if physical in (0, 2):
                    if magnetic not in m_lookup: continue
                    rows.append((old*4+(0 if physical == 0 else 3))*ancilla+local_ancilla)
                    components.append(multiplicity*len(magnetic_values)+m_lookup[magnetic])
                    numerators.append(1); denominators.append(1); signs.append(1)
                else:
                    for local, local_m in ((1, 1), (2, -1)):
                        if magnetic+local_m not in m_lookup: continue
                        if spin == old_spin+1:
                            numerator = old_spin+magnetic+2 if local_m == 1 else old_spin-magnetic+2
                            sign = 1
                        else:
                            numerator = old_spin-magnetic if local_m == 1 else old_spin+magnetic
                            sign = -1 if local_m == 1 else 1
                        if not numerator: continue
                        rows.append((old*4+local)*ancilla+local_ancilla)
                        components.append(multiplicity*len(magnetic_values)+m_lookup[magnetic+local_m])
                        numerators.append(numerator); denominators.append(2*(old_spin+1)); signs.append(sign)
        coefficient = (torch.tensor(numerators, device=device)/torch.tensor(denominators, device=device)).sqrt()*torch.tensor(signs, device=device)
        result.append({"quantum": (number, spin), "channels": len(channels), "magnetic": magnetic_values,
                       "rows": torch.tensor(rows, device=device), "components": torch.tensor(components, device=device),
                       "coefficient": coefficient})
    return result


@torch.no_grad()
def _isometry(raw, left_labels, allowed, maximum_bond, relative_tolerance):
    flat = raw.flatten(0, 2)
    maps = _coupled_rows(left_labels, raw.shape[2], allowed, raw.device)
    candidates, blocks = [], []
    for group in maps:
        magnetic_count = len(group["magnetic"])
        coupled = flat.new_zeros(group["channels"]*magnetic_count, flat.shape[1])
        coupled.index_add_(0, group["components"], group["coefficient"][:, None]*flat[group["rows"]])
        coupled = coupled.reshape(group["channels"], magnetic_count, -1)
        u, singular, _ = torch.linalg.svd(coupled.flatten(1), full_matrices=False)
        group["u"], group["singular"] = u, singular
        block = len(blocks); blocks.append(group)
        # Selection decisions are host metadata; all singular-value algebra is CUDA.
        for channel in range(len(singular)):
            candidates.append((block, channel, magnetic_count))
    if not candidates: raise RuntimeError("No physical prefix sector remains")
    values = torch.cat([group["singular"] for group in blocks])
    order = values.argsort(descending=True).tolist()
    threshold = max(1e-14, relative_tolerance)*values.max()
    eligible = (values > threshold).tolist()
    selected = {index: [] for index in range(len(blocks))}
    total, capacity = 0, False
    for index in order:
        block, channel, magnetic_count = candidates[index]
        if not eligible[index]: continue
        if total+magnetic_count > maximum_bond:
            capacity = True
            continue
        selected[block].append(channel); total += magnetic_count
    if not total: raise RuntimeError("Physical response bond cannot retain a single spin multiplet")
    matrix = flat.new_zeros(flat.shape[0], total)
    labels, offset, discarded = [], 0, flat.new_zeros(())
    for block, group in enumerate(blocks):
        keep = sorted(selected[block])
        keep_index = torch.tensor(keep, dtype=torch.long, device=raw.device)
        magnetic_count = len(group["magnetic"])
        retained = group["u"][:, keep_index]
        multiplier = torch.arange(len(keep), device=raw.device)[None]*magnetic_count
        magnetic = group["components"] % magnetic_count
        multiplicity = group["components"]//magnetic_count
        destination = offset+multiplier+magnetic[:, None]
        contribution = group["coefficient"][:, None]*retained[multiplicity]
        index = group["rows"][:, None]*total+destination
        matrix.view(-1).index_add_(0, index.flatten(), contribution.flatten())
        for channel in range(len(keep)):
            labels.extend((*group["quantum"], channel, magnetic) for magnetic in group["magnetic"])
        discarded += group["singular"].square().sum()-group["singular"][keep_index].square().sum()
        offset += len(keep)*magnetic_count
    carry = matrix.T @ flat
    orthogonality = float((matrix.T @ matrix-torch.eye(total, device=raw.device)).norm())
    if orthogonality > 1e-9: raise RuntimeError("Coupled response isometry lost orthogonality")
    return matrix.reshape(*raw.shape[:3], total), carry, labels, {
        "retained_dimension": total, "capacity_limited": capacity,
        "local_coupled_discarded_squared_norm": float(discarded.clamp_min(0)),
        "local_full_discarded_squared_norm": float((flat.square().sum()-carry.square().sum()).clamp_min(0)),
        "isometry_error": orthogonality}


@torch.no_grad()
def _project_endpoint(raw, left_labels, number, spin):
    if raw.shape[-1] != 1: raise ValueError("A physical endpoint must have one right channel")
    flat = raw.flatten(0, 2)
    groups = _coupled_rows(left_labels, raw.shape[2], {(number, spin)}, raw.device, final_m=spin)
    projected = torch.zeros_like(flat)
    for group in groups:
        coupled = flat.new_zeros(group["channels"], 1)
        coupled.index_add_(0, group["components"], group["coefficient"][:, None]*flat[group["rows"]])
        projected.index_add_(0, group["rows"], group["coefficient"][:, None]*coupled[group["components"]])
    return projected.reshape_as(raw), float((flat-projected).square().sum())


@torch.no_grad()
def apply_physical(mpo, state, maximum_bond=64, relative_tolerance=1e-11, maximum_raw_bytes=1_500_000_000, callback=None):
    """Compress H|state> with exact N/S, complete magnetic multiplets.

    Every returned chain is physical. Compression changes H|state>, not H;
    its local indicators MUST NOT be reported as a full residual certificate.
    """
    if len(state.cores) != mpo.sites: raise ValueError("Hamiltonian and state site counts differ")
    allowed = quantum_bonds(mpo.sites, state.electrons, state.twice_spin, 1)
    labels, cores = [[(0, 0, 0, 0)]], []
    carry = state.cores[0].new_ones(1, 1, 1)
    history = []
    for site, source in enumerate(state.cores):
        raw, raw_bytes = mpo.local_image(site, carry, source, maximum_raw_bytes)
        if site+1 == mpo.sites:
            # The right endpoint is the fixed target N,S,M sector. The
            # original scalar operator and invariant earlier projectors
            # retain it; an independent sector check is required in tests.
            if raw.shape[-1] != 1: raise RuntimeError("Physical response has an open right boundary")
            raw, _ = _project_endpoint(raw, labels[-1], state.electrons, state.twice_spin)
            cores.append(raw)
            labels.append([(state.electrons, state.twice_spin, 0, state.twice_spin)])
            break
        core, right, new_labels, row = _isometry(raw, labels[-1], allowed[site+1], maximum_bond, relative_tolerance)
        cores.append(core); labels.append(new_labels)
        carry = right.reshape(len(new_labels), mpo.bond_dimensions[site+1], source.shape[-1])
        row.update({"site": site, "raw_intermediate_bytes": raw_bytes})
        history.append(row)
        if callback: callback(row)
    result = SpinChain(cores, labels, state.electrons, state.twice_spin)
    return result, {"history": history, "maximum_bond": max(value.shape[-1] for value in cores),
                    "capacity_limited": any(row["capacity_limited"] for row in history),
                    "full_image_error_certified": False, "full_stationarity_certified": False}
