import numpy as np
import pde

class DiffusionRates:
    def __init__(self, d, e, md, mde, me):
        self.d = d
        self.e = e
        self.md = md
        self.mde = mde
        self.me = me

class KineticRates:
    def __init__(self, d0, d1, e_a, e_d, de0, de0_prime, k_de):
        self.d0 = d0
        self.d1 = d1
        self.e_a = e_a
        self.e_d = e_d
        self.de0 = de0
        self.de0_prime = de0_prime
        self.k_de = k_de

class MinDMinEBonnyPDE(pde.PDEBase):
    def __init__(self, diffusion_rate, rate, c_max, h):
        super().__init__()
        self.diffusion_rate = diffusion_rate
        self.rate = rate
        self.c_max = c_max
        self.h = h

    def evolution_rate(self, state, t=0) -> pde.FieldCollection:
        c_d, c_e, m_d, m_de, m_e = state

        J_d_on = (self.rate.d0 + self.rate.d1 * m_d) * (1 - (m_d + m_de) / self.c_max) * c_d
        J_e_on = self.rate.e_a * m_d * c_e
        
        J_d_off = (self.rate.de0 + self.rate.de0_prime) * m_de 
        
        J_e_off_from_mde = self.rate.de0 * m_de 
        J_e_off_from_me = self.rate.e_d * m_e

        J_membrane_complex = self.rate.k_de * m_d * m_e

        c_d_tdiff = (
            self.diffusion_rate.d * c_d.laplace(bc='neumann') 
            + J_d_off / self.h 
            - J_d_on / self.h
        )
        c_e_tdiff = (
            self.diffusion_rate.e * c_e.laplace(bc='neumann')
            + J_e_off_from_mde / self.h  # FIXED: MinE now correctly returns to cytosol
            + J_e_off_from_me / self.h
            - J_e_on / self.h
        )

        m_d_tdiff = (
            self.diffusion_rate.md * m_d.laplace(bc='neumann')
            + J_d_on 
            - J_e_on 
            - J_membrane_complex
        )
        m_de_tdiff = (
            self.diffusion_rate.mde * m_de.laplace(bc='neumann')
            + J_e_on 
            + J_membrane_complex 
            - J_d_off  
        )
        m_e_tdiff = (
            self.diffusion_rate.me * m_e.laplace(bc='neumann')
            + self.rate.de0_prime * m_de 
            - J_membrane_complex 
            - J_e_off_from_me
        )

        return pde.FieldCollection([c_d_tdiff, c_e_tdiff, m_d_tdiff, m_de_tdiff, m_e_tdiff])

def main():
    grid = pde.CartesianGrid([[0, 4]], [100])
    
    c_d = pde.ScalarField(grid, 2200)
    c_e = pde.ScalarField(grid, 1500)
    m_d = pde.ScalarField.random_uniform(grid, 0.0, 1000)
    m_de = pde.ScalarField.random_uniform(grid, 0.0, 1000)
    m_e = pde.ScalarField.random_uniform(grid, 0.0, 1000)
    
    initial_state = pde.FieldCollection([c_d, c_e, m_d, m_de, m_e])
    
    diff_rates = DiffusionRates(d=14, e=14, md=0.06, mde=0.06, me=0.6)
    kin_rates = KineticRates(d0=0.1, d1=0.0088, e_a=0.00007, e_d=0.5, de0=0.08, de0_prime=1.5, k_de=0.0049)
    
    eq = MinDMinEBonnyPDE(diffusion_rate=diff_rates, rate=kin_rates, c_max=5400, h=0.25)
    
    print("Starting...")
    storage = pde.MemoryStorage()
    
    result = eq.solve(
        initial_state, 
        t_range=10,
        solver="scipy",
        method="BDF",
        tracker=["progress", storage.tracker(0.1)]
    )
    
    pde.plot_kymograph(storage, field_index=2, title="Membrane-bound minD concentration")

if __name__ == '__main__':
    main()