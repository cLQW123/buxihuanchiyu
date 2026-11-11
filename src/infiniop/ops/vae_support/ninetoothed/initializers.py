"""NineToothed numerical buffers created while loading the H3 VAEs."""
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed.language import libdevice


def rotary_arrangement(output, count, theta):
    return output.tile((128,)), count, theta


def rotary_application(output, count, theta):
    index = ntl.cast(output.offsets(-1), ntl.float64)
    step = ntl.full((), 1.0 / count, dtype=ntl.float64)
    exponent = ntl.cast(index * step, ntl.float32)
    base = ntl.full((), theta, dtype=ntl.float32)
    powered = libdevice.pow(base, exponent)
    output = 1.0 / powered


def premake_rotary(count, theta):
    return rotary_arrangement, rotary_application, (
        nt.Tensor(1, dtype=nt.float32),
        nt.Tensor(0, constexpr=True, value=count),
        nt.Tensor(0, constexpr=True, value=theta))


def filter_arrangement(output, count, cutoff, beta):
    return output.tile((128,)), count, cutoff, beta


def filter_application(output, count, cutoff, beta):
    index = ntl.cast(output.offsets(-1), ntl.float32)
    alpha = (count - 1) / 2.0
    normalized = (index - alpha) / alpha
    beta32 = ntl.full((), beta, dtype=ntl.float32)
    argument = beta32 * ntl.sqrt(ntl.maximum(1.0 - normalized * normalized, 0.0))
    numerator = ntl.full((128,), 1.0, dtype=ntl.float32)
    denominator = ntl.full((), 1.0, dtype=ntl.float32)
    term = ntl.full((128,), 1.0, dtype=ntl.float32)
    base_term = ntl.full((), 1.0, dtype=ntl.float32)
    for order in range(1, 32):
        term = term * (argument * argument / 4.0) / (order * order)
        base_term = base_term * (beta32 * beta32 / 4.0) / (order * order)
        numerator = numerator + term
        denominator = denominator + base_term
    window = numerator / denominator
    time = index - count // 2
    if count % 2 == 0:
        time = time + 0.5
    frequency = 2.0 * cutoff * time
    phase = 3.141592653589793 * frequency
    sinc = ntl.where(frequency == 0, 1.0, libdevice.sin(phase) / phase)
    output = 2.0 * cutoff * window * sinc


def premake_filter(count, cutoff, beta):
    return filter_arrangement, filter_application, (
        nt.Tensor(1, dtype=nt.float32),
        nt.Tensor(0, constexpr=True, value=count),
        nt.Tensor(0, constexpr=True, value=cutoff),
        nt.Tensor(0, constexpr=True, value=beta))


def normalize_arrangement(input, output):
    source = input.tile((128,))
    return source, output.tile((128,))


def normalize_application(input, output):
    total = ntl.sum(input, 0)
    output = input / total


def premake_normalize():
    return normalize_arrangement, normalize_application, (
        nt.Tensor(1, dtype=nt.float32, other=0), nt.Tensor(1, dtype=nt.float32))
