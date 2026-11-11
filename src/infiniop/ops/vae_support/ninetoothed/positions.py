"""FL2VA packed rotary positions, including sequential FP64 time sums."""
import ninetoothed as nt
import ninetoothed.language as ntl


def arrangement(output, text, frames, height, width, patch_h, patch_w, audio, channels, anchors):
    return (output.tile((128, 4)), text, frames, height, width,
            patch_h, patch_w, audio, channels, anchors)


def application(output, text, frames, height, width, patch_h, patch_w, audio, channels, anchors):
    row = output.offsets(-2)
    axis = output.offsets(-1)
    rows_per_frame = (height // patch_h) * (width // patch_w)
    audio_start = text + anchors * rows_per_frame
    video_start = audio_start + audio * channels
    video_row = ntl.maximum(row - video_start, 0)
    frame = video_row // rows_per_frame
    clock = ntl.full((128,), 0.0, dtype=ntl.float64)
    for index in range(frames):
        span = ntl.full((), 1.6666666666666667, dtype=ntl.float64)
        if index % 5 != 0:
            span = span * 4.0
        clock = clock + ntl.where(index < frame, span, 0.0)
    text64 = ntl.full((), text, dtype=ntl.float64)
    time = ntl.where(row < text, ntl.cast(row, ntl.float64),
                    ntl.where(row < audio_start, text64,
                              ntl.where(row < video_start,
                                        text64 + ntl.cast((row - audio_start) % audio, ntl.float64),
                                        text64 + clock)))
    spatial_row = ntl.where(row < audio_start, row - text, video_row) % rows_per_frame
    h_index = ntl.cast(spatial_row // (width // patch_w), ntl.float64)
    w_index = ntl.cast(spatial_row % (width // patch_w), ntl.float64)
    area = ntl.full((), height * width, dtype=ntl.float64)
    root = ntl.sqrt(area)
    h_ratio = ntl.full((), height, dtype=ntl.float64) / root
    w_ratio = ntl.full((), width, dtype=ntl.float64) / root
    h_left = (1.0 - h_ratio) / 2.0
    w_left = (1.0 - w_ratio) / 2.0
    h_step = ((h_left + h_ratio) - h_left) / (height // patch_h)
    w_step = ((w_left + w_ratio) - w_left) / (width // patch_w)
    h_coord = (h_left + h_index * h_step) * 32.0
    w_coord = (w_left + w_index * w_step) * 32.0
    audio_w_index = ntl.where(row - audio_start < audio, 0, width // patch_w - 1)
    audio_w = (w_left + ntl.cast(audio_w_index, ntl.float64) * w_step) * 32.0
    is_video = (row >= text) & ((row < audio_start) | (row >= video_start))
    h_coord = ntl.where(is_video, h_coord, 0.0)
    w_coord = ntl.where(is_video, w_coord,
                        ntl.where(row >= audio_start, audio_w, 0.0))
    output = ntl.where(axis[None, :] == 0, time[:, None],
                       ntl.where(axis[None, :] == 1, h_coord[:, None], w_coord[:, None]))


def premake(text, frames, height, width, patch_h, patch_w, audio, channels, anchors):
    values = (text, frames, height, width, patch_h, patch_w, audio, channels, anchors)
    tensors = (nt.Tensor(2, dtype=nt.float64),) + tuple(
        nt.Tensor(0, constexpr=True, value=value) for value in values)
    return arrangement, application, tensors
