"""MT19937 stream and normal transform expressed through NineToothed.

The host bridge only serializes the released CPU generator state. Integer
recurrence, tempering, uniform conversion and Box-Muller run on the device.
"""
import functools

import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed.language import libdevice


def _serial(tensor):
    result = tensor.tile((1,)).tile((-1,))
    result.dtype.dtype = result.dtype.dtype.squeeze(0)
    return result


def stream_arrangement(state, output, initial_index, initial_twist):
    return _serial(state), _serial(output), initial_index, initial_twist


def stream_application(state, output, initial_index, initial_twist):
    cursor = initial_index
    for position in range(output.source.shape[0]):
        if cursor == 624 or (position == 0 and initial_twist):
            for index in range(624):
                first = state[index]
                second = state[(index + 1) % 624]
                bits = (first & 2147483648) | (second & 2147483647)
                distant = state[(index + 397) % 624]
                state[index] = distant ^ (bits >> 1) ^ ntl.where(
                    (second & 1) != 0, 2567483615, 0)
            cursor = 0
        value = state[cursor]
        value = value ^ (value >> 11)
        value = value ^ ((value << 7) & 2636928640)
        value = value ^ ((value << 15) & 4022730752)
        value = value ^ (value >> 18)
        output[position] = value
        cursor += 1


def premake_stream(initial_index, initial_twist):
    return stream_arrangement, stream_application, (
        nt.Tensor(shape=(624,), dtype=nt.uint32),
        nt.Tensor(1, dtype=nt.uint32),
        nt.Tensor(0, constexpr=True, value=initial_index),
        nt.Tensor(0, constexpr=True, value=initial_twist),
    )


def normal_arrangement(raw, output, count, digits):
    # A complete raw stream is available to each output tile. Dynamic indexing
    # selects paired uniforms without creating a third-party RNG dependency.
    arranged = _serial(raw).expand((output.tile((128,)).shape[0],))
    return arranged, output.tile((128,)), count, digits


def normal_application(raw, output, count, digits):
    index = output.offsets(-1)
    group = index // 16 * 16
    slot = index % 16
    if count % 16 != 0:
        final = index >= count - 16
        group = ntl.where(final, count, group)
        slot = ntl.where(final, index - (count - 16), slot)
    paired = slot % 8
    first = raw[group + paired]
    second = raw[group + paired + 8]
    u1 = ntl.cast(first & ((1 << digits) - 1), ntl.float32) * (1.0 / (1 << digits))
    u2 = ntl.cast(second & ((1 << digits) - 1), ntl.float32) * (1.0 / (1 << digits))
    if digits == 24:
        radius = ntl.sqrt(-2.0 * ntl.log(1.0 - u1))
        theta = 6.2831854820251465 * u2
        cosine = ntl.cos(theta)
        sine = ntl.sin(theta)
        output = radius * ntl.where(slot < 8, cosine, sine)
    else:
        # The released CPU low-precision normal_fill stores each scalar
        # intermediate in scalar_t; retain those rounding boundaries.
        complement = ntl.cast(1.0 - u1, output.dtype)
        logarithm = ntl.cast(libdevice.log(ntl.cast(complement, ntl.float32)), output.dtype)
        radius = ntl.cast(libdevice.sqrt(-2.0 * ntl.cast(logarithm, ntl.float32)), output.dtype)
        theta_wide = 6.283185307179586 * ntl.cast(u2, ntl.float64)
        theta = ntl.cast(ntl.cast(theta_wide, ntl.float32), output.dtype)
        cosine = ntl.cast(libdevice.cos(ntl.cast(theta, ntl.float32)), output.dtype)
        sine = ntl.cast(libdevice.sin(ntl.cast(theta, ntl.float32)), output.dtype)
        output = ntl.cast(radius, ntl.float32) * ntl.cast(
            ntl.where(slot < 8, cosine, sine), ntl.float32)


def premake_normal(count, dtype):
    digits = {nt.float32: 24, nt.float16: 11, nt.bfloat16: 8}[dtype]
    return normal_arrangement, normal_application, (
        nt.Tensor(1, dtype=nt.uint32), nt.Tensor(1, dtype=dtype),
        nt.Tensor(0, constexpr=True, value=count),
        nt.Tensor(0, constexpr=True, value=digits),
    )
