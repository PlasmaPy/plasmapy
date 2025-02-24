atm_per_gbar = 986923169.31427


def ptau_to_nTtau_E(ptau):
    """Convert ptau (in atm s) to nTtauE (in m^-3 keV s)

    Note that P=2nT. So ptau = 2nTtau_E.

    Keyword arguments:
    ptau -- pressure times confinement time in atm s
    """
    atm_s_to_keV_per_m3_s = 6.325e20
    nTtauE_avg = (1.0 / 2.0) * ptau * atm_s_to_keV_per_m3_s
    return nTtauE_avg


def nTtau_E_to_ptau(nTtau_E):
    """Convert nTtau_E (in m^-3 keV s) to ptau (in atm s)

    Note that nT=P/2. So nTtau_E = ptau/2

    Keyword arguments:
    nTtau_E -- fusion triple product
    """
    keV_per_m3_s_to_atm_s = 1 / 6.325e20
    ptau = 2 * nTtau_E * keV_per_m3_s_to_atm_s
    return ptau
