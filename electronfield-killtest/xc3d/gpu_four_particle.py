"""Matrix-free four-particle/hole Hamiltonian for our larger-basis research.

The antisymmetric alpha-pair/beta-pair tensor needs n^4 values, not a dense
C(n,2)^2 square Hamiltonian. Exact pair antisymmetry reduces ten contractions
to five. Particle-hole conjugation also covers closed-shell four-hole cases.
These are numerical identities, not claims of a new electronic method.
"""
from itertools import combinations
import torch


class FourParticleOperator:
    def __init__(self, h, eri, electrons, nuclear=None):
        if h.device.type != "cuda" or eri.device.type != "cuda":
            raise RuntimeError("CUDA is required; no CPU numerical fallback")
        self.sites = len(h)
        self.holes = electrons != 4
        if self.holes and 2*self.sites-electrons != 4:
            raise ValueError("This exact backend requires four particles or four holes")
        self.electrons = electrons
        self.constant = h.new_zeros(()) if nuclear is None else nuclear.clone()
        if self.holes:
            identity = torch.eye(len(h), device=h.device, dtype=h.dtype)
            coulomb = torch.einsum("ijkl,kl->ij", eri, 2*identity)
            exchange = torch.einsum("ikjl,kl->ij", eri, 2*identity)
            self.constant += (2*identity*(h+.5*coulomb-.25*exchange)).sum()
            self.h = -(h+coulomb-.5*exchange).T
        else:
            self.h = h
        n = self.sites
        self.pair = eri.permute(0, 2, 1, 3).contiguous().reshape(n*n, n*n)

    @staticmethod
    def project(value):
        first = value-value.transpose(0, 1)
        return (first-first.transpose(2, 3))/4

    def naive(self, vector):
        n = self.sites
        value = vector.reshape(n, n, n, n)
        result = self.constant*value
        for axis in range(4):
            rotated = value.movedim(axis, 0)
            result = result+(self.h @ rotated.reshape(n, -1)).reshape_as(rotated).movedim(0, axis)
        for first, second in combinations(range(4), 2):
            order = [first, second]+[axis for axis in range(4) if axis not in (first, second)]
            reverse = [order.index(axis) for axis in range(4)]
            rotated = value.permute(order)
            contribution = (self.pair @ rotated.reshape(n*n, -1)).reshape_as(rotated).permute(reverse)
            result = result+contribution
        return self.project(result).flatten()

    def __call__(self, vector):
        n = self.sites
        value = vector.reshape(n, n, n, n)
        first = (self.h @ value.reshape(n, -1)).reshape_as(value)
        result = first-first.transpose(0, 1)
        rotated = value.movedim(2, 0)
        first = (self.h @ rotated.reshape(n, -1)).reshape_as(rotated).movedim(0, 2)
        result = result+first-first.transpose(2, 3)
        flat = value.reshape(n*n, n*n)
        result = result+(self.pair @ flat).reshape_as(value)+(flat @ self.pair.T).reshape_as(value)
        rotated = value.permute(0, 2, 1, 3)
        cross = (self.pair @ rotated.reshape(n*n, n*n)).reshape_as(rotated).permute(0, 2, 1, 3)
        first = cross-cross.transpose(0, 1)
        result = result+first-first.transpose(2, 3)+self.constant*value
        return self.project(result).flatten()

    def cast(self, dtype, shift=0.):
        result = object.__new__(type(self))
        result.sites, result.holes, result.electrons = self.sites, self.holes, self.electrons
        result.h = self.h.to(dtype)
        result.pair = self.pair.to(dtype)
        result.constant = (self.constant-shift).to(dtype)
        return result

    def seed(self):
        value = self.h.new_zeros((self.sites,)*4)
        i, j = (self.sites-2, self.sites-1) if self.holes else (0, 1)
        value[i, j, i, j] = value[j, i, j, i] = .5
        value[j, i, i, j] = value[i, j, j, i] = -.5
        return value.flatten()

    def coordinates(self, electron_bits):
        n = self.sites
        bits = ((1 << (2*n))-1) ^ electron_bits if self.holes else electron_bits
        sites = torch.arange(n, device=bits.device)
        up = ((bits[:, None] >> (2*sites[None])) & 1).bool()
        down = ((bits[:, None] >> (2*sites[None]+1)) & 1).bool()
        if not bool((up.sum(-1) == 2).all() & (down.sum(-1) == 2).all()):
            raise ValueError("The supplied sector is not two-up/two-down quasiparticles")
        up_index = sites[None].expand(len(bits), -1)[up].reshape(len(bits), 2)
        down_index = sites[None].expand(len(bits), -1)[down].reshape(len(bits), 2)
        inversions = (up_index[:, :, None] > down_index[:, None, :]).sum((-1, -2))
        if self.holes:
            inversions = inversions+(2*up_index).sum(-1)+(2*down_index+1).sum(-1)
        sign = 1-2*inversions.remainder(2)
        return up_index, down_index, sign

    def from_fock(self, amplitudes, bits):
        up, down, sign = self.coordinates(bits)
        value = amplitudes.new_zeros((self.sites,)*4)
        coefficient = amplitudes*sign/2
        i, j, k, l = up[:, 0], up[:, 1], down[:, 0], down[:, 1]
        value[i, j, k, l] = coefficient
        value[j, i, k, l] = -coefficient
        value[i, j, l, k] = -coefficient
        value[j, i, l, k] = coefficient
        return value.flatten()

    def to_fock(self, vector, bits):
        up, down, sign = self.coordinates(bits)
        value = vector.reshape((self.sites,)*4)
        return 2*sign*value[up[:, 0], up[:, 1], down[:, 0], down[:, 1]]


@torch.no_grad()
def cached_ritz(action, seed, maximum=192, tolerance=1e-8, log_every=25):
    """One new operator application per expansion; keep earlier images."""
    state = seed/seed.norm()
    basis, images = state[:, None], action(state)[:, None]
    history = []
    for iteration in range(maximum):
        projected = basis.T @ images
        values, vectors = torch.linalg.eigh((projected+projected.T)/2)
        state = basis @ vectors[:, 0]
        image = images @ vectors[:, 0]
        energy = values[0]
        residual = image-energy*state
        size = float(residual.norm())
        row = {"dimension": basis.shape[1], "energy_hartree": float(energy), "residual": size}
        if iteration == 0 or (iteration+1) % log_every == 0 or size < tolerance:
            history.append(row); print(row, flush=True)
        if size < tolerance or basis.shape[1] >= maximum:
            break
        direction = residual
        for _ in range(2):
            direction = direction-basis @ (basis.T @ direction)
        length = direction.norm()
        if float(length) < 1e-13:
            break
        direction = direction/length
        basis = torch.cat((basis, direction[:, None]), 1)
        images = torch.cat((images, action(direction)[:, None]), 1)
    return {"state": state, "energy": energy, "residual": size, "converged": size < tolerance,
            "dimension": basis.shape[1], "operator_applications": images.shape[1], "history": history,
            "basis_image_bytes": (basis.numel()+images.numel())*basis.element_size()}


@torch.no_grad()
def incremental_ritz(action, seed, maximum=192, tolerance=1e-8, log_every=25, projector=None):
    """Preallocate state/image storage and update only the new projected row.

    The physical operator, Rayleigh quotient and stopping criterion are the
    same; this removes repeated whole-basis actions/projections and copies.
    """
    storage = seed.new_empty(len(seed), maximum)
    image_storage = torch.empty_like(storage)
    projected = seed.new_zeros(maximum, maximum)
    seed = seed if projector is None else projector(seed)
    storage[:, 0] = seed/seed.norm()
    image_storage[:, 0] = action(storage[:, 0])
    projected[0, 0] = storage[:, 0] @ image_storage[:, 0]
    history = []
    for iteration in range(maximum):
        count = iteration+1
        basis, images = storage[:, :count], image_storage[:, :count]
        values, vectors = torch.linalg.eigh(projected[:count, :count].to(torch.float64))
        coefficient = vectors[:, 0].to(seed.dtype)
        state, image = basis @ coefficient, images @ coefficient
        energy = values[0].to(seed.dtype)
        residual = image-energy*state
        size = float(residual.norm())
        row = {"dimension": count, "energy_hartree": float(energy), "residual": size}
        if iteration == 0 or count % log_every == 0 or size < tolerance:
            history.append(row); print(row, flush=True)
        if size < tolerance or count >= maximum:
            break
        direction = residual if projector is None else projector(residual)
        for _ in range(2):
            direction = direction-basis @ (basis.T @ direction)
            if projector is not None:
                direction = projector(direction)
        length = direction.norm()
        if float(length) < 1e-13:
            break
        direction = direction/length
        storage[:, count] = direction
        image_storage[:, count] = action(direction)
        new_column = basis.T @ image_storage[:, count]
        new_row = images.T @ direction
        column = (new_column+new_row)/2
        projected[:count, count] = projected[count, :count] = column
        projected[count, count] = direction @ image_storage[:, count]
    return {"state": state, "energy": energy, "residual": size, "converged": size < tolerance,
            "dimension": count, "operator_applications": count, "history": history,
            "basis_image_bytes": (storage.numel()+image_storage.numel())*storage.element_size()}


class GraphOperator:
    """Static CUDA execution with private input/output buffers and safe copies."""
    def __init__(self, operator, seed):
        self.operator = operator
        self.input = seed.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.output = operator(self.input)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = operator(self.input)

    @torch.no_grad()
    def __call__(self, vector):
        self.input.copy_(vector)
        self.graph.replay()
        # Operator images must not alias the next graph replay's output.
        return self.output.clone()

    @torch.no_grad()
    def replace(self, operator):
        if self.operator.sites != operator.sites or self.operator.h.dtype != operator.h.dtype:
            raise ValueError("CUDA graph replacement requires the same shape and dtype")
        self.operator.h.copy_(operator.h)
        self.operator.pair.copy_(operator.pair)
        self.operator.constant.copy_(operator.constant)
