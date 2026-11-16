"""Run with: python -m unittest -v test_prototype.py"""
import unittest
import numpy as np
import torch
from prototype import (XCFunctional, exact_system, external_potential, generate_dataset,
                       grid_setup, centered_potential, scf)


class PhysicsChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float64)
        cls.dx = 0.45
        cls.x, cls.t, cls.w, cls.hkin = grid_setup(35, cls.dx)

    def test_exact_density_normalization(self):
        row = exact_system(self.x, self.t, self.w, self.hkin, self.dx, 1.1, 0.9, 1.8)
        n = np.array(row['n'])
        self.assertAlmostEqual(np.sum(n)*self.dx, 2, places=10)
        self.assertLess(row['e'], -1.0)
        self.assertLess(row['e_c'], 0)

    def test_inverted_potential_recovers_reference_density(self):
        row = exact_system(self.x, self.t, self.w, self.hkin, self.dx, 1.1, 0.9, 2.3)
        n = np.array(row['n'])
        vs = np.array(row['v']) + 0.5*self.dx*self.w.dot(n) + np.array(row['vc'])
        eigs, vecs = np.linalg.eigh(self.t + np.diag(vs))
        n_inv = 2 * vecs[:,0]**2 / self.dx
        self.assertLess(np.max(np.abs(n - n_inv)), 1e-8)

    def test_functional_derivative_matches_finite_difference(self):
        torch.manual_seed(5)
        model = XCFunctional(self.x, self.w, self.t, self.dx)
        n = torch.tensor([generate_dataset(seed=6)[0]['n']],dtype=torch.float64).requires_grad_(True)
        v = model.corr_potential(n)[0,18].item()
        delta = 1.e-5
        no = n.detach().clone()
        np_ = no.clone(); nm_ = no.clone()
        np_[0,18] += delta; nm_[0,18] -= delta
        fd = ((model.corr_energy(np_)-model.corr_energy(nm_))/(2*delta*self.dx)).item()
        self.assertAlmostEqual(v, fd, places=6)

    def test_reproducible_dataset(self):
        x = generate_dataset(seed=42)
        y = generate_dataset(seed=42)
        self.assertEqual([z['id'] for z in x], [z['id'] for z in y])
        np.testing.assert_allclose([z['e'] for z in x], [z['e'] for z in y], atol=1e-10, rtol=0)
        self.assertEqual((sum(z['split']=='train' for z in x),sum(z['split']=='test_id' for z in x),sum(z['split']=='test_ood' for z in x)), (28,8,8))

    def test_scf_converges_for_zero_correlation(self):
        model=XCFunctional(self.x, self.w, self.t, self.dx)
        for p in model.parameters():
            torch.nn.init.zeros_(p)
        v=external_potential(self.x,1.1,.9,2.0)
        with torch.enable_grad():
            e,n,r=scf(model,torch.tensor(v).view(1,-1),steps=250,mix=.15)
        self.assertLess(float(r.detach()[0]),1e-7)
        self.assertAlmostEqual(float(n.detach().sum()*self.dx),2.0,places=7)


if __name__=='__main__':
    unittest.main()
