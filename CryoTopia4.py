import CoolProp as cp
from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity

import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
import csv
import os
import zipfile
import argparse
import json
from datetime import datetime

# ============================================================
# Inputs and geometry
# ============================================================

sim_duration = 20 * ureg("hours")
time_incr = 1 * ureg("second")   # you can change this; model is designed to survive large dt

ambient_pressure = 101325 * ureg("pascal")
ambient_temp = 300 * ureg("kelvin")
fluid = "Methane"

percent_fill = 0.5

relief_pressure = 250 * ureg("psi")
relief_deadband = 20 * ureg("psi")

tank_diameter = 0.5 * ureg("meter")
tank_height = 1.5 * ureg("meter")
tank_volume = np.pi * (tank_diameter / 2) ** 2 * tank_height

# pressurization setpoint (gets updated by ops)
press_set = ambient_pressure.to("pascal")

adiabatic = 0

# operations schedule (times in hours, like your original)
ops = {
   
    1 : ["vent", "time", 3600 * 5 * ureg("second"), None, None],
    5 : ["vent", "time", 3600 * ureg("second"), None, None],
    6 : ["press", "pressure", 150 * ureg("psi"), "time", 6000 * ureg("second")],

    # 5 : ["drain", "time", 240 * ureg("second"), None, None],
   
    # 6 : ["vent", "time", 3600 * ureg("second"), None, None],
    # 8 : ["vent", "pressure", ambient_pressure, None, None],

    # 9 : ["vent", "time", 3600 * 3 * ureg("second"), None, None],
    # 12 : ["press", "pressure", 150 * ureg("psi"), "time", 4* 600 * ureg("second")],
    # 12.5 : ["drain", "time", 600 * ureg("second"), None, None],
    # 14.5 : ["vent", "time", 3600 * 2 * ureg("second"), None, None],

    # 30.5: ["vent", "time", 3600 * 5 * ureg("second"), None, None],
    # 35.6 : ["press", "pressure", 150 * ureg("psi"), "time", 3800 * ureg("second")],
    # 36.6 : ["drain", "time", 200 * ureg("second"), None, None],
    # 37.0: ["vent", "time", 3600 * ureg("second"), None, None],
    # 35 : ["vent", "time", 36000 * 3 * ureg("second"), None, None],
    # 50 : ["press", "pressure", 150 * ureg("psi"), "time", 3600 * ureg("second")],
    # 90.5: ["vent", "pressure", ambient_pressure, None, None],
    
    
    



    80 : ["vent", "time", 3600 * 10 * ureg("second"), None, None],
    91: ["press", "pressure", 80 * ureg("psi"), "time", 1800 * ureg("second")],
    91.1 : ["drain", "time", 400 * ureg("second"), None, None],
    92 : ["vent", "time", 3600 * 2 * ureg("second"), None, None],
    94: ["press", "pressure", 40 * ureg("psi"), "time", 600 * ureg("second")],
    95: ["drain", "time", 1200 * ureg("second"), None, None],
    96 : ["vent", "time", 3600 * ureg("second"), None, None],
}

# Orifice geometry
relief_diameter = 0.09 * ureg("cm")
vent_diameter = 0.12 * ureg("cm")

relief_area = np.pi * (relief_diameter.to("meter") / 2) ** 2
vent_area = np.pi * (vent_diameter.to("meter") / 2) ** 2

relief_pressure = relief_pressure.to("pascal")
relief_deadband = relief_deadband.to("pascal")
sim_duration = sim_duration.to("second")

# ============================================================
# Heat transfer constants
# ============================================================

sigma = 5.670374419e-8 * ureg("watt/meter**2/kelvin**4")
gap_length = 0.02 * ureg("meter")
surface_A = 2 * np.pi * (tank_diameter / 2) ** 2 + tank_height * tank_diameter * np.pi
emissivity_eff = 0.001

gap_pressure = (10 ** -5 * ureg("torr")).to("pascal")
a1 = 0.4 * ureg("dimensionless")
a2 = 0.3 * ureg("dimensionless")
omega = 2.13 * ureg("watt/meter**2/kelvin/pascal")
accommodation_coef = (a1 * a2) / (a2 + a1 * (1 - a2))

k_stainless = 16.2 * ureg("watt/meter/kelvin")
connecting_tube_diameter = 0.2 * ureg("meter")
connecting_tube_thickness = 0.002 * ureg("meter")
connecting_tube_cross_sectional_area = np.pi * (
    (connecting_tube_diameter / 2) ** 2
    - ((connecting_tube_diameter / 2) - connecting_tube_thickness) ** 2
)
connecting_tube_length = 0.05 * ureg("meter")

# Max fraction of liquid mass that can boil off in one timestep (stability control)
MAX_BOIL_FRAC_PER_STEP = 0.05    # 5% per time step


# ============================================================
# Helper functions
# ============================================================

def orifice_equation(Cd, A, P1, P2, rho):
    """
    Simple orifice equation: mdot = Cd * A * sqrt(2 * rho * (P1 - P2))
    All args are pint Quantities. Returns mdot [kg/s].
    """
    delta_P = P1 - P2
    if delta_P.magnitude <= 0:
        mdot = -Cd * A * (2 * abs(delta_P) * rho) ** 0.5
    else:
        mdot = Cd * A * (2 * delta_P * rho) ** 0.5
    return mdot.to("kg/second")


def Q_rad(ambient_temp, sat_temp, surface_A, emissivity):
    """Radiative heat from ambient to surface [W]."""
    return emissivity * sigma * surface_A * (ambient_temp ** 4 - sat_temp ** 4)


def Q_residualgas(surface_A, accommodation_coef, omega, pressure, ambient_temp, sat_temp):
    """Residual gas conduction [W]."""
    return surface_A * accommodation_coef * omega * pressure * (ambient_temp - sat_temp)


def Q_structcond(csa, ambient_temp, sat_temp, strut_length, k):
    """Structural conduction [W]."""
    return k * csa / strut_length * (ambient_temp - sat_temp)


def liquid_gas_heat_transfer(L_temp, G_temp, L_volume, G_pressure):
    """
    Natural convection between liquid surface and gas.
    Returns Q_dot_L_from_G [W] (positive if liquid gains heat from gas).

    Includes guards so it doesn't blow up when gas volume is tiny.
    """

    # Geometry
    r = tank_diameter.to("meter").magnitude / 2.0
    A_cs = np.pi * r ** 2
    gas_vol = (tank_volume - L_volume).to("meter**3").magnitude

    # If gas layer is effectively gone, turn off convection
    if gas_vol <= 1e-4:   # 0.1 L threshold
        return 0 * ureg("watt")

    L_char = gas_vol / A_cs  # [m]

    # Scalars
    T_L = L_temp.to("kelvin").magnitude
    T_G = G_temp.to("kelvin").magnitude + 0.1
    T_film = 0.5 * (T_L + T_G)
    P_g = G_pressure.to("pascal").magnitude

    if T_film <= 1.0:
        T_film = 1.0

    # Fluid properties at film T and gas P
    mu = cp.CoolProp.PropsSI("V", "T", T_film, "P", P_g, fluid)
    rho = cp.CoolProp.PropsSI("D", "T", T_film, "P", P_g, fluid)
    k = cp.CoolProp.PropsSI("L", "T", T_film, "P", P_g, fluid)
    cp_m = cp.CoolProp.PropsSI("C", "T", T_film, "P", P_g, fluid)

    nu = mu / rho         # m^2/s
    alpha = k / (rho * cp_m)

    g = 9.81
    beta = 1.0 / T_film
    dT = T_G - T_L

    # Rayleigh number (non-negative)
    Ra = g * beta * abs(dT) * L_char ** 3 / (nu * alpha)
    if (not np.isfinite(Ra)) or (Ra < 0.0):
        Ra = 0.0

    Nu = 0.27 * Ra ** 0.25 if Ra > 0 else 0.0
    h = Nu * k / L_char          # W/m^2/K

    Qdot = h * A_cs * dT         # W (sign from T_G - T_L)

    # Safety cap on power (prevents insane spikes)
    Qdot_max = 5e4  # 50 kW
    Qdot = max(min(Qdot, Qdot_max), -Qdot_max)

    return Qdot * ureg("watt")


# ============================================================
# Simulation driver
# ============================================================

def run_it(initial_state=None, extend_seconds=None, no_plot=False, append=False):
    global press_set  # we modify it based on ops

    # Helper to deserialize possible quantity dicts from metadata
    def _to_qty(val):
        # if val is a dict with value+unit, return pint Quantity
        try:
            if isinstance(val, dict) and "value" in val and "unit" in val:
                return Q_(val["value"], val["unit"])
        except Exception:
            pass
        return val

    # -------------------------------
    # Initial gas (saturated vapor) OR resume from provided initial_state
    # -------------------------------
    # determine local simulation duration
    if extend_seconds is not None:
        local_sim_duration = extend_seconds * ureg("second")
    else:
        local_sim_duration = sim_duration

    time_offset = 0 * ureg("second")

    if initial_state:
        # initial_state may contain quantities serialized as {value, unit}
        G_mass = _to_qty(initial_state.get("G_mass", None))
        L_mass = _to_qty(initial_state.get("L_mass", None))
        G_enthalpy = _to_qty(initial_state.get("G_enthalpy", None))
        L_enthalpy = _to_qty(initial_state.get("L_enthalpy", None))
        G_temp = _to_qty(initial_state.get("G_temp", None))
        L_temp = _to_qty(initial_state.get("L_temp", None))
        G_pressure = _to_qty(initial_state.get("G_pressure", None))
        time_offset = _to_qty(initial_state.get("time_s", 0)) * ureg("second") if initial_state.get("time_s", None) is not None else 0 * ureg("second")

        # If some values weren't provided, compute sensible defaults
        if G_pressure is None:
            G_pressure = ambient_pressure
        if G_temp is None:
            try:
                G_temp = (cp.CoolProp.PropsSI("T", "Q", 1, "P", G_pressure.to("pascal").magnitude, fluid) + 0.5) * ureg("kelvin")
            except Exception:
                G_temp = ambient_temp
        if L_temp is None:
            try:
                L_temp = cp.CoolProp.PropsSI("T", "Q", 0, "P", G_pressure.to("pascal").magnitude, fluid) * ureg("kelvin")
            except Exception:
                L_temp = ambient_temp

        # compute densities and volumes
        try:
            L_density = cp.CoolProp.PropsSI("D", "T", L_temp.to("kelvin").magnitude, "P", G_pressure.to("pascal").magnitude, fluid) * ureg("kg/m^3")
        except Exception:
            L_density = cp.CoolProp.PropsSI("D", "Q", 0, "P", ambient_pressure.magnitude, fluid) * ureg("kg/m^3")

        if L_mass is None:
            # estimate from percent_fill
            L_volume = tank_volume * percent_fill
            L_mass = L_density * L_volume
        else:
            L_volume = (L_mass / L_density).to("m^3")

        G_volume = tank_volume - L_volume
        if G_mass is None:
            try:
                G_density = cp.CoolProp.PropsSI("D", "T", G_temp.to("kelvin").magnitude, "P", G_pressure.to("pascal").magnitude, fluid) * ureg("kg/m^3")
                G_mass = G_density * G_volume
            except Exception:
                G_mass = 1e-6 * ureg("kg")

        # enthalpies
        if G_enthalpy is None:
            try:
                G_enthalpy = cp.CoolProp.PropsSI("H", "T", G_temp.to("kelvin").magnitude, "P", G_pressure.to("pascal").magnitude, fluid) * ureg("J/kg")
            except Exception:
                G_enthalpy = cp.CoolProp.PropsSI("H", "Q", 1, "P", ambient_pressure.magnitude, fluid) * ureg("J/kg")
        if L_enthalpy is None:
            try:
                L_enthalpy = cp.CoolProp.PropsSI("H", "T", L_temp.to("kelvin").magnitude, "P", G_pressure.to("pascal").magnitude, fluid) * ureg("J/kg")
            except Exception:
                L_enthalpy = cp.CoolProp.PropsSI("H", "Q", 0, "P", ambient_pressure.magnitude, fluid) * ureg("J/kg")

    else:
        G_volume = tank_volume * (1 - percent_fill)
        G_temp = (cp.CoolProp.PropsSI("T", "Q", 1, "P", ambient_pressure.magnitude, fluid) + 0.5) * ureg("kelvin")
        G_enthalpy = cp.CoolProp.PropsSI("H", "Q", 1, "P", ambient_pressure.magnitude, fluid) * ureg("J/kg")
        G_density = cp.CoolProp.PropsSI("D", "H", G_enthalpy.magnitude, "P", ambient_pressure.magnitude, fluid) * ureg("kg/m^3")
        G_mass = G_density * G_volume
        G_pressure = ambient_pressure

        # -------------------------------
        # Initial liquid (saturated)
        # -------------------------------
        L_volume = tank_volume * percent_fill
        L_temp = cp.CoolProp.PropsSI("T", "Q", 0, "P", ambient_pressure.magnitude, fluid) * ureg("kelvin")
        L_enthalpy = cp.CoolProp.PropsSI("H", "Q", 0, "P", ambient_pressure.magnitude, fluid) * ureg("J/kg")
        # define initial density from mass/volume so geometry stays consistent
        L_density_init = cp.CoolProp.PropsSI("D", "Q", 0, "P", ambient_pressure.magnitude, fluid) * ureg("kg/m^3")
        L_mass = L_density_init * L_volume

    # -------------------------------
    # State flags
    # -------------------------------
    venting = False
    relieving = False
    pressing = False
    draining = False

    vent_time = 0 * ureg("second")
    press_time = 0 * ureg("second")
    drain_time = 0 * ureg("second")
    vent_pressure = 0 * ureg("psi")

    # -------------------------------
    # Logging arrays
    # -------------------------------
    time_data = []
    gas_temp_data = []
    liquid_temp_data = []
    gas_pressure_data = []
    gas_mass_data = []
    liquid_mass_data = []
    liquid_quality = []
    liquid_enthalpy_data = []

    venting_events = []
    relief_events = []
    pressurization_events = []
    draining_events = []

    liquid_heat_conv = []
    liquid_heat_rad = []
    gas_heat_rad = []
    gas_heat_conv = []
    gas_heat_strut = []
    liq_gas_heat = []
    gas_heat = []
    liquid_heat = []

    # Time loop (use local_sim_duration and include time_offset when logging)
    time_array = np.arange(0, local_sim_duration.to("second").magnitude, time_incr.magnitude)

    for i in tqdm(range(len(time_array))):
        time = i * time_incr

        # clamp masses non-negative
        if L_mass.magnitude < 0:
            L_mass = 0 * ureg("kg")
        if G_mass.magnitude < 0:
            G_mass = 0 * ureg("kg")

        # Geometry
        r = tank_diameter / 2
        A_cs = np.pi * r ** 2

        # Use current geometry: liquid volume changes only when mass changes
        if L_mass.magnitude > 0:
            # current effective density
            L_density = (L_mass / L_volume).to("kg/m^3")
        else:
            L_volume = 0 * ureg("m^3")
            L_density = 1e3 * ureg("kg/m^3")   # arbitrary

        G_volume = tank_volume - L_volume
        if G_volume.magnitude <= 1e-9:
            G_volume = 1e-9 * ureg("m^3")

        if G_mass.magnitude > 0:
            G_density = (G_mass / G_volume).to("kg/m^3")
        else:
            G_density = 1e-9 * ureg("kg/m^3")

        # Surface areas
        L_height = (L_volume / A_cs).to("meter")
        L_height = max(0 * ureg("meter"), min(L_height, tank_height))

        L_surface_area = 2 * np.pi * r * L_height + np.pi * r ** 2
        G_surface_area = surface_A - L_surface_area

        # ----------------------------------------------------
        # Heat transfer for this step
        # ----------------------------------------------------
        # liquid-gas convection
        Qdot_LG = liquid_gas_heat_transfer(L_temp, G_temp, L_volume, G_pressure)
        Q_LG = Qdot_LG * time_incr  

        # external HT (if not adiabatic)
        if adiabatic:
            L_Q_conv = 0 * ureg("J")
            L_Q_rad = 0 * ureg("J")
            G_Q_rad = 0 * ureg("J")
            G_Q_conv = 0 * ureg("J")
            G_Q_struct = 0 * ureg("J")
        else:
            L_Q_conv = Q_residualgas(L_surface_area, accommodation_coef, omega,
                                     gap_pressure, ambient_temp, L_temp) * time_incr
            L_Q_rad = Q_rad(ambient_temp, L_temp, L_surface_area, emissivity_eff) * time_incr

            G_Q_rad = Q_rad(ambient_temp, G_temp, G_surface_area, emissivity_eff) * time_incr
            G_Q_conv = Q_residualgas(G_surface_area, accommodation_coef, omega,
                                     gap_pressure, ambient_temp, G_temp) * time_incr
            G_Q_struct = Q_structcond(connecting_tube_cross_sectional_area,
                                      ambient_temp, G_temp,
                                      connecting_tube_length, k_stainless) * time_incr

        # total heat into each phase *before* phase change
        Q_L_raw = L_Q_conv + L_Q_rad + Q_LG
        Q_G_raw = G_Q_rad + G_Q_conv + G_Q_struct - Q_LG

        # ----------------------------------------------------
        # Liquid enthalpy update + limited boiling
        # ----------------------------------------------------
        if L_mass.magnitude > 0:
            # sensible enthalpy change
            L_enthalpy = L_enthalpy + Q_L_raw / L_mass

            # saturation properties at current gas pressure
            h_f = cp.CoolProp.PropsSI("H", "Q", 0, "P", G_pressure.magnitude, fluid) * ureg("J/kg")
            h_g = cp.CoolProp.PropsSI("H", "Q", 1, "P", G_pressure.magnitude, fluid) * ureg("J/kg")
            h_fg = (h_g - h_f)

            # quality estimate based on current enthalpy (can be <0 if subcooled)
            try:
                L_quality_val = cp.CoolProp.PropsSI("Q", "H", L_enthalpy.magnitude,
                                                    "P", G_pressure.magnitude, fluid)
            except Exception:
                L_quality_val = -1.0

            # Limit: we only let at most MAX_BOIL_FRAC_PER_STEP of the liquid boil
            m_boil = 0 * ureg("kg")

            if L_quality_val > 0.0 and h_fg.magnitude > 0:
                # "thermodynamic" mass that corresponds to quality
                m_boil_thermo = L_quality_val * L_mass
                m_boil_limit = MAX_BOIL_FRAC_PER_STEP * L_mass
                m_boil = min(m_boil_thermo, m_boil_limit, L_mass)

                # shrink liquid volume & mass
                L_volume = L_volume - (m_boil / L_density)
                L_mass = L_mass - m_boil
                G_mass = G_mass + m_boil

                # after boiling, reset liquid enthalpy to saturated liquid
                L_enthalpy = h_f
                L_quality_val = 0.0

            # update liquid temperature from EOS (robust-ish)
            try:
                L_temp = cp.CoolProp.PropsSI("T", "H", L_enthalpy.magnitude,
                                             "P", G_pressure.magnitude, fluid) * ureg("kelvin")
            except Exception:
                # fallback: keep last value
                L_temp = L_temp

        else:
            L_quality_val = -1.0
            L_temp = L_temp

        # ----------------------------------------------------
        # Gas enthalpy update (no explicit latent term; boiling limited)
        # ----------------------------------------------------
        if G_mass.magnitude > 0:
            G_enthalpy = G_enthalpy + Q_G_raw / G_mass

            try:
                G_pressure = cp.CoolProp.PropsSI("P", "H", G_enthalpy.magnitude,
                                                 "D", G_density.magnitude, fluid) * ureg("pascal")
                G_temp = cp.CoolProp.PropsSI("T", "H", G_enthalpy.magnitude,
                                             "P", G_pressure.magnitude, fluid) * ureg("kelvin")
            except Exception:
                # fallback: keep last state
                G_pressure = G_pressure
                G_temp = G_temp
            
            # ----------------------------------------------------
            # Gas condensation into liquid
            # ----------------------------------------------------
            if G_mass.magnitude > 0 and abs(G_temp - L_temp).magnitude < 5.0:  # Threshold of 5 K
                # Saturation properties at current gas pressure
                h_f = cp.CoolProp.PropsSI("H", "Q", 0, "P", G_pressure.magnitude, fluid) * ureg("J/kg")
                h_g = cp.CoolProp.PropsSI("H", "Q", 1, "P", G_pressure.magnitude, fluid) * ureg("J/kg")
                h_fg = (h_g - h_f)

                if h_fg.magnitude > 0:
                # Estimate mass of gas that can condense
                    m_condense = G_mass * 0.05  # Limit to 5% of gas mass per step
                    m_condense = min(m_condense, G_mass)

                    # Update gas and liquid states
                    G_mass = G_mass - m_condense
                    L_mass = L_mass + m_condense
                    L_volume = L_volume + (m_condense / L_density)

                    # Adjust enthalpies
                    H_gas = G_enthalpy * G_mass
                    H_condensed = h_f * m_condense
                    G_enthalpy = (H_gas - H_condensed) / G_mass if G_mass.magnitude > 0 else h_g
                    L_enthalpy = (L_enthalpy * (L_mass - m_condense) + H_condensed) / L_mass
                
        else:
            G_pressure = ambient_pressure
            G_temp = ambient_temp

        # ----------------------------------------------------
        # Ops timeouts
        # ----------------------------------------------------
        if venting:
            if time.magnitude >= vent_time.magnitude or G_pressure <= vent_pressure:
                venting = False

        if pressing:
            if time.magnitude >= press_time.magnitude:
                pressing = False

        if draining:
            if time.magnitude >= drain_time.magnitude:
                draining = False

        # Trigger ops
        for op_time_hr, operation in ops.items():
            op_time = (op_time_hr * ureg("hours")).to("second")
            if abs((op_time - time).magnitude) <= time_incr.magnitude + 0.1:
                mode, trig_type, trig_val, extra_type, extra_val = operation

                if mode == "vent":
                    venting = True
                    if trig_type == "time":
                        vent_time = time + trig_val
                    elif trig_type == "pressure":
                        vent_pressure = trig_val.to("pascal")

                elif mode == "press":
                    pressing = True
                    if trig_type == "pressure":
                        press_set = trig_val.to("pascal")
                        if extra_type == "time":
                            press_time = time + extra_val
                    elif trig_type == "time":
                        press_time = time + trig_val

                elif mode == "drain":
                    draining = True
                    if trig_type == "time":
                        drain_time = time + trig_val

        # Relief valve logic
        if G_pressure >= relief_pressure:
            relieving = True
        elif G_pressure <= (relief_pressure - relief_deadband):
            relieving = False

        # ----------------------------------------------------
        # Mass flows (press, vent, relief, drain)
        # ----------------------------------------------------
        # Save mass at start of flows for limiting
        L_mass_start = L_mass
        G_mass_start = G_mass

        # Pressurization (with enthalpy mixing)
        if pressing:
            pressurization_events.append(20)
            Cd_p = 0.6
            press_mdot = orifice_equation(Cd_p, 1e-5 * ureg("m^2"),
                                          press_set, G_pressure, G_density)
            press_mass = press_mdot * time_incr

            # limit: at most 20% of starting gas mass per step
            max_press_mass = 0.2 * G_mass_start
            if press_mass > max_press_mass:
                press_mass = max_press_mass
            if press_mass.magnitude < 0:
                press_mass = 0 * ureg("kg")

            if press_mass.magnitude > 0:
                # supply gas state (assume ambient temp at press_set)
                T_supply = ambient_temp
                h_supply = cp.CoolProp.PropsSI(
                    "H", "T", T_supply.to("kelvin").magnitude,
                    "P", press_set.magnitude, fluid
                ) * ureg("J/kg")

                H_old = G_enthalpy * G_mass
                H_in = h_supply * press_mass
                G_mass = G_mass + press_mass
                G_enthalpy = (H_old + H_in) / G_mass
        else:
            pressurization_events.append(0)

        # Venting
        if venting and G_mass.magnitude > 0:
            venting_events.append(20)
            Cd_v = 0.6
            vent_mdot = orifice_equation(Cd_v, vent_area,
                                         G_pressure, ambient_pressure, G_density)
            vent_mass = vent_mdot * time_incr
            max_vent_mass = 0.2 * G_mass_start
            if vent_mass > max_vent_mass:
                print("GAAH")
                vent_mass = max_vent_mass
            elif vent_mass < 0:
                pass
            if vent_mass > G_mass:
                print("RAAH")
                vent_mass = G_mass

            H_old = G_enthalpy * G_mass
            H_out = G_enthalpy * vent_mass
            G_mass = G_mass - vent_mass
            if G_mass.magnitude > 0:
                G_enthalpy = (H_old - H_out) / G_mass
        else:
            venting_events.append(0)

        # Relief
        if relieving and G_mass.magnitude > 0:
            relief_events.append(20)
            Cd_r = 0.8
            relief_mdot = orifice_equation(Cd_r, relief_area,
                                           G_pressure, ambient_pressure, G_density)
            relief_mass = relief_mdot * time_incr
            max_relief_mass = 0.2 * G_mass_start
            if relief_mass > max_relief_mass:
                relief_mass = max_relief_mass
            if relief_mass > G_mass:
                relief_mass = G_mass

            H_old = G_enthalpy * G_mass
            H_out = G_enthalpy * relief_mass
            G_mass = G_mass - relief_mass
            if G_mass.magnitude > 0:
                G_enthalpy = (H_old - H_out) / G_mass
        else:
            relief_events.append(0)

        # Draining liquid
        if draining and L_mass.magnitude > 0:
            draining_events.append(20)
            Cd_d = 0.6
            drain_mdot = orifice_equation(Cd_d, 1e-5 * ureg("m^2"),
                                          G_pressure, ambient_pressure, L_density)
            drain_mass = drain_mdot * time_incr
            max_drain_mass = 0.2 * L_mass_start
            if drain_mass > max_drain_mass:
                drain_mass = max_drain_mass
            if drain_mass > L_mass:
                drain_mass = L_mass

            L_volume = L_volume - (drain_mass / L_density)
            L_mass = L_mass - drain_mass
        else:
            draining_events.append(0)

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------
        # record absolute time (seconds) including offset when resuming
        time_data.append((time_offset + time).to("second").magnitude)
        liquid_enthalpy_data.append(L_enthalpy.to("J/kg").magnitude)
        gas_temp_data.append(G_temp.to("kelvin").magnitude)
        liquid_temp_data.append(L_temp.to("kelvin").magnitude)
        gas_pressure_data.append(G_pressure.to("psi").magnitude)
        gas_mass_data.append(G_mass.to("kg").magnitude)
        liquid_mass_data.append(L_mass.to("kg").magnitude)
        liquid_quality.append(L_quality_val)

        liquid_heat_conv.append(L_Q_conv.to("J").magnitude)
        liquid_heat_rad.append(L_Q_rad.to("J").magnitude)
        gas_heat_rad.append(G_Q_rad.to("J").magnitude)
        gas_heat_conv.append(G_Q_conv.to("J").magnitude)
        gas_heat_strut.append(G_Q_struct.to("J").magnitude)
        liq_gas_heat.append(Q_LG.to("J").magnitude)
        gas_heat.append(Q_G_raw.to("J").magnitude)
        liquid_heat.append(Q_L_raw.to("J").magnitude)

    # ========================================================
    # Plotting (same format as your original)
    # ========================================================
    time_hr = [t / 3600 for t in time_data]

    # Temperature + Pressure
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=time_hr, y=gas_temp_data, mode='lines',
    #                          name='Gas Temperature (K)', line=dict(color='blue')))
    # fig.add_trace(go.Scatter(x=time_hr, y=liquid_temp_data, mode='lines',
    #                          name='Liquid Temperature (K)', line=dict(color='orange')))
    # fig.add_trace(go.Scatter(x=time_hr, y=gas_pressure_data, mode='lines',
    #                          name='Gas Pressure (psi)', line=dict(color='green'), yaxis='y2'))

    # fig.update_layout(
    #     title="Temperature and Pressure Changes Over Time",
    #     xaxis=dict(title="Time (hr)"),
    #     yaxis=dict(title="Temperature (K)", titlefont=dict(color="blue"), tickfont=dict(color="blue")),
    #     yaxis2=dict(title="Pressure (psi)", titlefont=dict(color="green"),
    #                 tickfont=dict(color="green"), overlaying='y', side='right'),
    #     legend=dict(x=0.01, y=0.99),
    #     template="seaborn"
    # )

    # # Masses
    # fig_mass = go.Figure()
    # fig_mass.add_trace(go.Scatter(x=time_hr, y=gas_mass_data, mode='lines',
    #                               name='Gas Mass (kg)', line=dict(color='blue')))
    # fig_mass.add_trace(go.Scatter(x=time_hr, y=liquid_mass_data, mode='lines',
    #                               name='Liquid Mass (kg)', line=dict(color='orange'), yaxis='y2'))

    # fig_mass.update_layout(
    #     title="Mass Changes Over Time",
    #     xaxis=dict(title="Time (hr)"),
    #     yaxis=dict(title="Gas Mass (kg)", titlefont=dict(color="blue"), tickfont=dict(color="blue")),
    #     yaxis2=dict(title="Liquid Mass (kg)", titlefont=dict(color="orange"),
    #                 tickfont=dict(color="orange"), overlaying='y', side='right'),
    #     legend=dict(x=0.01, y=0.99),
    #     template="plotly_white"
    # )

    # # Quality + enthalpy
    # fig_quality = go.Figure()
    # fig_quality.add_trace(go.Scatter(x=time_hr, y=liquid_quality, mode='lines',
    #                                  name='Liquid Quality', line=dict(color='purple')))
    # fig_quality.add_trace(go.Scatter(x=time_hr, y=liquid_enthalpy_data, mode='lines',
    #                                  name='Liquid Enthalpy (J/kg)', line=dict(color='red'), yaxis='y2'))

    # fig_quality.update_layout(
    #     title="Liquid Quality Over Time",
    #     xaxis=dict(title="Time (hr)"),
    #     yaxis=dict(title="Quality (dimensionless)", titlefont=dict(color="purple"), tickfont=dict(color="purple")),
    #     yaxis2=dict(title="Enthalpy (J/kg)", titlefont=dict(color="red"),
    #                 tickfont=dict(color="red"), overlaying='y', side='right'),
    #     legend=dict(x=0.01, y=0.99),
    #     template="plotly_white"
    # )

    # # Heat flows
    # fig_heat = go.Figure()
    # fig_heat.add_trace(go.Scatter(x=time_hr, y=liquid_heat, mode='lines',
    #                               name='Heat into Liquid (J)', line=dict(color='blue')))
    # fig_heat.add_trace(go.Scatter(x=time_hr, y=gas_heat, mode='lines',
    #                               name='Heat into Gas (J)', line=dict(color='orange')))
    # fig_heat.add_trace(go.Scatter(x=time_hr, y=liquid_heat_conv, mode='lines',
    #                               name='Liquid Convective Heat (J)', line=dict(color='purple')))
    # fig_heat.add_trace(go.Scatter(x=time_hr, y=liquid_heat_rad, mode='lines',
    #                               name='Liquid Radiative Heat (J)', line=dict(color='pink')))
    # fig_heat.add_trace(go.Scatter(x=time_hr, y=gas_heat_rad, mode='lines',
    #                               name='Gas Radiative Heat (J)', line=dict(color='green')))
    # fig_heat.add_trace(go.Scatter(x=time_hr, y=gas_heat_conv, mode='lines',
    #                               name='Gas Convective Heat (J)', line=dict(color='cyan')))
    # fig_heat.add_trace(go.Scatter(x=time_hr, y=gas_heat_strut, mode='lines',
    #                               name='Gas Structural Conductive Heat (J)', line=dict(color='brown')))
    # fig_heat.add_trace(go.Scatter(x=time_hr, y=liq_gas_heat, mode='lines',
    #                               name='Liquid-Gas Heat Transfer (J)', line=dict(color='red')))

    # fig_heat.update_layout(
    #     title="Heat Transfer into Liquid and Gas Over Time",
    #     xaxis=dict(title="Time (hr)"),
    #     yaxis=dict(title="Heat (J)", titlefont=dict(color="black"), tickfont=dict(color="black")),
    #     legend=dict(x=0.01, y=0.99),
    #     template="plotly_white"
    # )

    # # State markers on main plot
    # fig.add_trace(go.Scatter(x=time_hr, y=pressurization_events, mode='lines',
    #                          name='Pressurizing State', line=dict(dash='dash')))
    # fig.add_trace(go.Scatter(x=time_hr, y=relief_events, mode='lines',
    #                          name='Relief State', line=dict(dash='dash')))
    # fig.add_trace(go.Scatter(x=time_hr, y=venting_events, mode='lines',
    #                          name='Venting State', line=dict(dash='dash')))
    # fig.add_trace(go.Scatter(x=time_hr, y=draining_events, mode='lines',
    #                          name='Draining State', line=dict(dash='dash')))

    # # Show plots
    # fig_heat.show()
    # fig_quality.show()
    # fig_mass.show()
    # fig.show()

    # ----------------------------------------------------
    # Export results so they can be emailed and reused
    # ----------------------------------------------------
    try:
        export_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(export_dir, exist_ok=True)

        csv_path = os.path.join(export_dir, "simulation_results.csv")
        # build header and rows
        header = [
            "time_s",
            "time_hr",
            "gas_temp_K",
            "liquid_temp_K",
            "gas_pressure_psi",
            "gas_mass_kg",
            "liquid_mass_kg",
            "liquid_quality",
            "liquid_enthalpy_J_per_kg",
            "liquid_heat_conv_J",
            "liquid_heat_rad_J",
            "gas_heat_rad_J",
            "gas_heat_conv_J",
            "gas_heat_strut_J",
            "liq_gas_heat_J",
            "gas_heat_J",
            "liquid_heat_J",
            "pressurization_events",
            "relief_events",
            "venting_events",
            "draining_events",
        ]
        # determine write mode: append or write
        write_header = True
        open_mode = 'w'
        if append and os.path.exists(csv_path):
            open_mode = 'a'
            write_header = False

        with open(csv_path, open_mode, newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)

            for row in zip(
                time_data,
                time_hr,
                gas_temp_data,
                liquid_temp_data,
                gas_pressure_data,
                gas_mass_data,
                liquid_mass_data,
                liquid_quality,
                liquid_enthalpy_data,
                liquid_heat_conv,
                liquid_heat_rad,
                gas_heat_rad,
                gas_heat_conv,
                gas_heat_strut,
                liq_gas_heat,
                gas_heat,
                liquid_heat,
                pressurization_events,
                relief_events,
                venting_events,
                draining_events,
            ):
                # ensure all values are plain python floats/ints
                safe_row = []
                for v in row:
                    try:
                        safe_row.append(float(v))
                    except Exception:
                        # fallback: string representation
                        safe_row.append(str(v))
                writer.writerow(safe_row)

        # write metadata JSON so runs can be resumed programmatically
        try:
            def _serialize_qty(q):
                try:
                    return {"value": q.magnitude, "unit": str(q.units)}
                except Exception:
                    return q

            metadata = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "inputs": {
                    "sim_duration_s": float(sim_duration.to("second").magnitude),
                    "time_incr_s": float(time_incr.to("second").magnitude),
                    "ambient_pressure": _serialize_qty(ambient_pressure),
                    "ambient_temp": _serialize_qty(ambient_temp),
                    "fluid": fluid,
                    "percent_fill": float(percent_fill),
                    "relief_pressure": _serialize_qty(relief_pressure),
                    "relief_deadband": _serialize_qty(relief_deadband),
                    "tank_diameter": _serialize_qty(tank_diameter),
                    "tank_height": _serialize_qty(tank_height),
                },
                "last_state": {
                    "time_s": float(time_data[-1]) if len(time_data) > 0 else 0.0,
                    "G_mass": _serialize_qty(G_mass),
                    "L_mass": _serialize_qty(L_mass),
                    "G_enthalpy": _serialize_qty(G_enthalpy),
                    "L_enthalpy": _serialize_qty(L_enthalpy),
                    "G_temp": _serialize_qty(G_temp),
                    "L_temp": _serialize_qty(L_temp),
                    "G_pressure": _serialize_qty(G_pressure),
                }
            }

            meta_path = os.path.join(export_dir, "simulation_metadata.json")
            # If appending, try to merge/update existing metadata
            try:
                if append and os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r') as mf_old:
                            existing = json.load(mf_old)
                    except Exception:
                        existing = None
                    # merge inputs (prefer current inputs) and update last_state
                    if existing and isinstance(existing, dict):
                        existing_inputs = existing.get('inputs', {})
                        existing['inputs'] = existing_inputs | metadata.get('inputs', {})
                        existing['last_state'] = metadata.get('last_state', {})
                        existing['timestamp'] = metadata.get('timestamp')
                        with open(meta_path, 'w') as mf:
                            json.dump(existing, mf, indent=2)
                        # print(f"Updated metadata to: {meta_path}")
                    else:
                        with open(meta_path, "w") as mf:
                            json.dump(metadata, mf, indent=2)
                        # print(f"Saved metadata to: {meta_path}")
                else:
                    with open(meta_path, "w") as mf:
                        json.dump(metadata, mf, indent=2)
                    # print(f"Saved metadata to: {meta_path}")
            except Exception as e:
                print("Warning: failed to write metadata:", e)
        except Exception as e:
            print("Warning: failed to write metadata:", e)

        # also write a zip for easy emailing
        zip_path = os.path.join(export_dir, "simulation_results.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, arcname=os.path.basename(csv_path))

        # print(f"Saved CSV to: {csv_path}")
        # print(f"Saved ZIP to: {zip_path}")
    except Exception as e:
        print("Warning: failed to export results:", e)


def _load_last_state_from_csv(csv_path):
    # read last data row and return a last_state dict with units
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            last = None
            for row in reader:
                last = row
            if last is None:
                return None
            idx = {h: i for i, h in enumerate(header)}
            ls = {}
            # map columns from csv to our expected serialized keys
            try:
                ls['time_s'] = float(last[idx['time_s']])
                ls['G_temp'] = {'value': float(last[idx['gas_temp_K']]), 'unit': 'kelvin'}
                ls['L_temp'] = {'value': float(last[idx['liquid_temp_K']]), 'unit': 'kelvin'}
                ls['G_pressure'] = {'value': float(last[idx['gas_pressure_psi']]), 'unit': 'psi'}
                ls['G_mass'] = {'value': float(last[idx['gas_mass_kg']]), 'unit': 'kg'}
                ls['L_mass'] = {'value': float(last[idx['liquid_mass_kg']]), 'unit': 'kg'}
                ls['L_enthalpy'] = {'value': float(last[idx['liquid_enthalpy_J_per_kg']]), 'unit': 'J/kg'}
            except Exception:
                return None
            return ls
    except Exception:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CryTopia2 simulation runner (supports resume from CSV/metadata)')
    parser.add_argument('--resume', '-r', help='Path to results folder, metadata.json, zip, or CSV to resume from')
    parser.add_argument('--extend-hours', type=float, help='Additional simulation duration in hours to run when resuming (overrides default sim_duration for this run)')
    parser.add_argument('--no-plot', action='store_true', help='Do not display Plotly figures')
    parser.add_argument('--append', action='store_true', help='Append results to existing CSV instead of overwriting')
    args = parser.parse_args()

    initial_state = None
    extend_seconds = None

    if args.resume:
        p = args.resume
        metadata = None
        # if path is a directory, look for metadata.json or csv
        if os.path.isdir(p):
            meta_path = os.path.join(p, 'simulation_metadata.json')
            csv_path = os.path.join(p, 'simulation_results.csv')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as mf:
                    metadata = json.load(mf)
            elif os.path.exists(csv_path):
                metadata = {'last_state': _load_last_state_from_csv(csv_path)}
        elif os.path.isfile(p):
            lower = p.lower()
            if lower.endswith('.zip'):
                import tempfile
                tmp = tempfile.mkdtemp()
                try:
                    with zipfile.ZipFile(p, 'r') as zf:
                        # try to extract metadata.json first
                        names = zf.namelist()
                        meta_names = [n for n in names if n.endswith('simulation_metadata.json')]
                        if meta_names:
                            with zf.open(meta_names[0]) as mf:
                                metadata = json.load(mf)
                        else:
                            # extract csv and parse last row
                            csv_names = [n for n in names if n.endswith('simulation_results.csv')]
                            if csv_names:
                                zf.extract(csv_names[0], tmp)
                                csvp = os.path.join(tmp, csv_names[0])
                                metadata = {'last_state': _load_last_state_from_csv(csvp)}
                except Exception as e:
                    print('Failed to read zip resume file:', e)
            elif lower.endswith('.json'):
                try:
                    with open(p, 'r') as mf:
                        metadata = json.load(mf)
                except Exception as e:
                    print('Failed to read metadata JSON:', e)
            elif lower.endswith('.csv'):
                metadata = {'last_state': _load_last_state_from_csv(p)}

        if metadata and 'last_state' in metadata and metadata['last_state']:
            initial_state = metadata['last_state']
        else:
            print('Warning: could not locate resume metadata or CSV last-row; starting fresh')

    if args.extend_hours is not None:
        extend_seconds = args.extend_hours * 3600.0

    run_it(initial_state=initial_state, extend_seconds=extend_seconds, no_plot=args.no_plot, append=args.append)

