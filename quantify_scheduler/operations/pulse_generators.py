"""

"""
from quantify_scheduler.operations import pulse_library


def gen_rxy_drag_pulse(
    amp180, motzoi, theta, phi, port, duration, clock
) -> pulse_library.DRAGPulse:
    # G_amp is the gaussian amplitude introduced in
    # https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.103.110501
    # 180 refers to the normalization, theta is in degrees, and
    # mw_amp180 is the amplitude necessary to get the
    # maximum 180 degree theta (experimentally)
    G_amp = amp180 * theta / 180
    D_amp = G_amp * motzoi

    pulse = pulse_library.DRAGPulse(
        G_amp=G_amp, D_amp=D_amp, phase=phi, port=port, duration=duration, clock=clock
    )
