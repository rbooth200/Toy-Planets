import sys
import numpy as np

from planet import Planet
from constants import Mearth, year, k_B, m_H, sig_SB, G

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    planets = [ ]
    for pl in sys.argv[1:]:
        planets.append( Planet.from_hdf5(pl) )

    planets = sorted(planets, key=lambda x: x.age)

    fig1, axes1 = plt.subplots(5,1, sharex='all')
    plt.subplots_adjust(hspace=0)

    for pl in planets:

        disc = pl._atmo

        r_e = pl.radii / 1e10
        r_c = 0.5*(r_e[1:] + r_e[:-1])

        print("Planet age (yr):",pl.age/year)
        print("Core Mass (Mearth):",pl.core.mass/Mearth)
        print("Envelope Mass (Mearth):",pl.shell_mass.sum()/Mearth)
        print("K-H time (yr):", pl.KelvinHelmholtzTime()/year)
        print("Number of cells:", pl.radii.size)

        axes1[0].loglog(r_e, pl.luminosity/3.83e33, 
                        label=str(pl.age/year)+'yr')
        axes1[0].set_ylabel('luminosity (Lsun)')

        axes1[1].loglog(r_c, pl.pressure/10**6, label=str(0/year)+'yr')
        axes1[1].set_ylabel('P [bar]')

        axes1[2].loglog(r_c, pl.get_density())    
        axes1[2].set_ylabel('rho [g/cm^3]')

        axes1[3].loglog(r_c, pl.temperature)
        axes1[3].set_ylabel('T [K]')

        axes1[4].semilogx(r_e[1:], pl.velocity[1:])
        axes1[4].set_ylabel('V [cm/s]')

        axes1[4].set_xlabel('R [10^10 cm]')

    axes1[0].legend(loc='best', ncol=3)
    plt.show()
