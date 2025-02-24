from plasmapy.reaction_rates import fusionlib, plasmaprofile


class BaseExperiment:
    def __init__(self):
        """Base class inhertited by all experiments. Currently only D-T fuel.

        self.name - name
        self.displayname = Display name
        self.profile - a plasmaprofile
        self.xi - Ratio T_e/T_i
        self.Zeff - Zeff
        self.Zbar - Zbar
        self.brem_frac - Fraction of bremsstrahlung which is lost
        self.eta_abs - P_in / P_ext (absorbtion efficiency of external heating)
        self.self_heating - Self heating on (True) or off (False)
        self.marker - Marker for plotting
        """
        pass

    def lawson_parameter_Q_fuel(self, T_i0, Q_fuel):
        """Required confinement parameter n_e tau_E required to achieve Q_fuel
        at temperature T_i.
        """
        # Handle case of ICF experiments which have additional efficiency
        # loss 'eta_hs'
        if hasattr(self, "eta_hs"):
            Q_fuel = Q_fuel / self.eta_hs
        # Evaluate Lawson paramter
        cf = fusionlib.DT_adjusted_lawson_parameter(
            T_i0=T_i0,
            Q_fuel=Q_fuel,
            lambda_F=self.profile.lambda_F_of_T(T_i0),
            lambda_B=self.profile.lambda_B(),
            lambda_kappa=self.profile.lambda_kappa(),
            xi=self.xi,
            Z_eff=self.Z_eff,
            Z_bar=self.Z_bar,
            brem_frac=self.brem_frac,
            self_heating=self.self_heating,
        )
        return cf

    def lawson_parameter_Q_sci(self, T_i0, Q_sci):
        """Required confinement parameter n_e tau_E required to achieve Q_sci
        at temperature T_i.
        """
        # Handle ICF case
        if hasattr(self, "eta_hs"):
            Q_fuel = Q_sci / (self.eta_abs * self.eta_hs)
        # Handle all other cases
        else:
            Q_fuel = Q_sci / self.eta_abs

        return self.lawson_parameter_Q_fuel(T_i0, Q_fuel)

    def triple_product_Q_fuel(self, T_i0, Q_fuel):
        return T_i0 * self.lawson_parameter_Q_fuel(T_i0=T_i0, Q_fuel=Q_fuel)

    def triple_product_Q_sci(self, T_i0, Q_sci):
        return T_i0 * self.lawson_parameter_Q_sci(T_i0=T_i0, Q_sci=Q_sci)


class UniformProfileDTExperiment(BaseExperiment):
    def __init__(self):
        self.name = "uniform_profile_experiment"
        self.displayname = "Uniform profile experiment"
        self.profile = plasmaprofile.UniformProfile()
        self.xi = 1
        self.Z_eff = 1
        self.Z_bar = 1
        self.brem_frac = 1
        self.eta_abs = 1
        self.self_heating = True


class UniformProfileHalfBremsstrahlungDTExperiment(BaseExperiment):
    def __init__(self):
        self.name = "uniform_profile_half_bremsstrahlung_experiment"
        self.displayname = "Uniform experiment half bremsstrahlung experiment"
        self.profile = plasmaprofile.UniformProfile()
        self.xi = 1
        self.Z_eff = 1
        self.Z_bar = 1
        self.brem_frac = 0.5
        self.eta_abs = 1
        self.self_heating = True


class UniformProfileHighEtaDTExperiment(BaseExperiment):
    def __init__(self):
        self.name = "uniform_profile_high_eta_experiment"
        self.displayname = "Uniform experiment high eta experiment"
        self.profile = plasmaprofile.UniformProfile()
        self.xi = 1
        self.Z_eff = 1
        self.Z_bar = 1
        self.brem_frac = 1
        self.eta_abs = 0.9
        self.self_heating = True


class UniformProfileLowEtaDTExperiment(BaseExperiment):
    def __init__(self):
        self.name = "uniform_profile_low_eta_experiment"
        self.displayname = "Uniform experiment low eta experiment"
        self.profile = plasmaprofile.UniformProfile()
        self.xi = 1
        self.Z_eff = 1
        self.Z_bar = 1
        self.brem_frac = 1
        self.eta_abs = 0.05
        self.self_heating = True


class ParabolicProfileDTExperiment(BaseExperiment):
    def __init__(self):
        self.name = "parabolic_experiment"
        self.displayname = "Parabolic experiment"
        self.profile = plasmaprofile.ParabolicProfile()
        self.xi = 1
        self.Z_eff = 1
        self.Z_bar = 1
        self.brem_frac = 1
        self.eta_abs = 1
        self.self_heating = True


class BennettProfileDTExperiment(BaseExperiment):
    def __init__(self):
        self.name = "bennett_experiment"
        self.displayname = "Bennett experiment"
        self.profile = plasmaprofile.BennettProfile()
        self.xi = 1
        self.Z_eff = 1
        self.Z_bar = 1
        self.brem_frac = 1
        self.eta_abs = 1
        self.self_heating = True


class PeakedAndBroadDTExperiment(BaseExperiment):
    def __init__(self):
        self.name = "pabdt_experiment"
        self.displayname = "pabdt experiment"
        self.profile = plasmaprofile.PeakedAndBroadProfile()
        self.xi = 1
        self.Z_eff = 1
        self.Z_bar = 1
        self.brem_frac = 1
        self.eta_abs = 1
        self.self_heating = True


class LowImpurityPeakedAndBroadDTExperiment(BaseExperiment):
    def __init__(self):
        self.name = "lipabdt_experiment"
        self.displayname = "lipabdt experiment"
        self.profile = plasmaprofile.PeakedAndBroadProfile()
        self.xi = 1
        self.Z_eff = 1.5
        self.Z_bar = 1.2
        self.brem_frac = 1
        self.eta_abs = 0.9  # representitive of MCF
        self.self_heating = True


class HighImpurityPeakedAndBroadDTExperiment(BaseExperiment):
    def __init__(self):
        self.name = "hipabdt_experiment"
        self.displayname = "hipabdt experiment"
        self.profile = plasmaprofile.PeakedAndBroadProfile()
        self.xi = 1
        self.Z_eff = 3.4
        self.Z_bar = 1.2
        self.brem_frac = 1
        self.eta_abs = 0.9  # representative of MCF
        self.self_heating = True


class DirectDriveICFDTExperiment(BaseExperiment):
    def __init__(self):
        self.name = "icf_direct_drive_dt_experiment"
        self.displayname = "icf direct_drive dt experiment"
        self.profile = plasmaprofile.UniformProfile()
        self.xi = 1
        self.Z_eff = 1
        self.Z_bar = 1
        self.brem_frac = 0
        self.eta_abs = 0.05  # representative of direct drive ICF
        self.self_heating = True


class IndirectDriveICFDTExperiment(BaseExperiment):
    def __init__(self):
        self.name = "icf_indirect_drive_dt_experiment"
        self.displayname = "icf indirect drive dt experiment"
        self.profile = plasmaprofile.UniformProfile()
        self.xi = 1
        self.Z_eff = 1
        self.Z_bar = 1
        self.brem_frac = 0
        self.self_heating = True

        # representative of indirect drive ICF (Shot N191007 at NIF)
        # Laser energy = 1.91 MJ
        # Fuel Kinetic Energy = 16.7 kJ
        # Hotspot energy = 10.9 kJ
        self.eta_abs = 0.0087
        self.eta_hs = 0.65
        self.self_heating = True


class IndirectDriveICFDTNoSelfHeatingExperiment(BaseExperiment):
    def __init__(self):
        self.name = "icf_indirect_drive_dt_no_sh_experiment"
        self.displayname = "icf indirect drive dt no sh experiment"
        self.profile = plasmaprofile.UniformProfile()
        self.xi = 1
        self.Z_eff = 1
        self.Z_bar = 1
        self.brem_frac = 0

        # representative of indirect drive ICF (Shot N191007 at NIF)
        # Laser energy = 1.91 MJ
        # Fuel Kinetic Energy = 16.7 kJ
        # Hotspot energy = 10.9 kJ
        self.eta_abs = 0.0087
        self.eta_hs = 0.65

        self.self_heating = False
