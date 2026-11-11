"""NineToothed finite linspace candidate for scheduler configuration grids."""
import functools

import ninetoothed as nt
import ninetoothed.language as ntl


def arrangement(output, start, end):
    return output.tile((128,)), start, end


def application_f64(output, start, end):
    index = ntl.cast(output.offsets(-1), ntl.float64)
    count = output.source.shape[0]
    left = ntl.full((), start, dtype=ntl.float64)
    right = ntl.full((), end, dtype=ntl.float64)
    step = (right - left) / (count - 1)
    output = ntl.where(index < count // 2,
                       left + step * index,
                       right - step * (count - 1 - index))


def application_f32(output, start, end):
    index = ntl.cast(output.offsets(-1), ntl.float32)
    count = output.source.shape[0]
    left = ntl.full((), start, dtype=ntl.float32)
    right = ntl.full((), end, dtype=ntl.float32)
    step = (right - left) / (count - 1)
    output = ntl.where(index < count // 2,
                       left + step * index,
                       right - step * (count - 1 - index))


def premake(count, start=1.0, end=0.0, wide_step=True):
    if count < 2:
        raise ValueError("Scheduler grids require at least two points")
    tensors = (nt.Tensor(shape=(count,), dtype=nt.float32),
               nt.Tensor(0, constexpr=True, value=start),
               nt.Tensor(0, constexpr=True, value=end))
    return arrangement, application_f64 if wide_step else application_f32, tensors
