import numpy as np
import pde
from dataclasses import dataclass

@dataclass
class DiffusionRates:
    d: float
    e: float

@dataclass
class RateDAttachParams:
    d0: float
    d1_d: float
    d1_de: float
    def get_rate_d_attach(self, m_d, m_de):
        return self.d0 + self.d1_d*m_d + self.d1_de*m_de

@dataclass
class RateEAttachParams:
    e0: float
    def get_rate_e(self, m_d):
        return self.e0 * m_d

@dataclass
class KineticRates:
    nuclear_exchange: float
    de_detach: float
    d_attach: RateDAttachParams
    e_attach: RateEAttachParams

class MinDMinEHuangPDE(pde.PDEBase):
    def __init__(self, diffusion_rates: DiffusionRates, kinetic_rates: KineticRates, h, laplace_bc='neumann'):
        super().__init__()
        self.diffusion_rates = diffusion_rates
        self.kinetic_rates = kinetic_rates
        self.h = h
        self.laplace_bc = laplace_bc

    def evolution_rate(self, state, t=0) -> pde.FieldCollection:
        c_d_adp, c_d_atp, c_e, m_d, m_de = state

        rate_d_attach = self.kinetic_rates.d_attach.get_rate_d_attach(m_d, m_de)
        rate_e_attach = self.kinetic_rates.e_attach.get_rate_e(m_d)

        nuclear_exchange_density = self.kinetic_rates.nuclear_exchange * c_d_adp
        flux_de_detach = self.kinetic_rates.de_detach * m_de
        flux_d_attach = rate_d_attach * c_d_atp
        flux_e_attach = rate_e_attach * c_e

        dt_c_d_adp = (
            self.diffusion_rates.d * c_d_adp.laplace(bc=self.laplace_bc) 
            - nuclear_exchange_density
            + flux_de_detach / self.h
        )
        dt_c_d_atp = (
            self.diffusion_rates.d * c_d_atp.laplace(bc=self.laplace_bc) 
            + nuclear_exchange_density
            - flux_d_attach / self.h
        )
        dt_c_e = (
            self.diffusion_rates.e * c_e.laplace(bc=self.laplace_bc)
            - flux_e_attach / self.h
            + flux_de_detach / self.h
        )
        dt_m_d = (
            flux_d_attach
            - flux_e_attach
        )
        dt_m_de = (
            flux_e_attach
            - flux_de_detach
        )

        return pde.FieldCollection([dt_c_d_adp, dt_c_d_atp, dt_c_e, dt_m_d, dt_m_de])

def main():
    grid = pde.CartesianGrid([[0, 4]], [100])
    
    c_d_adp = pde.ScalarField(grid, 0)
    c_d_atp = pde.ScalarField(grid, 660)
    c_e = pde.ScalarField(grid, 250)
    m_d = pde.ScalarField.random_uniform(grid, 0.0, 1)
    m_de = pde.ScalarField.random_uniform(grid, 0.0, 0)
    
    initial_state = pde.FieldCollection([c_d_adp, c_d_atp, c_e, m_d, m_de])
    
    diff_rates = DiffusionRates(2.5, 2.5)
    kin_rates = KineticRates(1, 0.7, RateDAttachParams(0.025, 0.0015, 0.0015), RateEAttachParams(0.093))
    
    eq = MinDMinEHuangPDE(diffusion_rates=diff_rates, kinetic_rates=kin_rates, h=0.25)
    
    print("Starting...")
    storage = pde.MemoryStorage()
    
    result = eq.solve(
        initial_state, 
        t_range=20,
        solver="scipy",
        method="BDF",
        tracker=["progress", storage.tracker(0.1)]
    )
    
    pde.plot_kymograph(storage, field_index=3, title="Membrane-bound minD concentration")

if __name__ == '__main__':
    main()