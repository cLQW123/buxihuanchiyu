"""Host file publication for the competition AFS filesystem; no numerical patches."""
import errno
import os
import shutil
import tempfile


_real_replace = os.replace


def _replace(src, dst, *args, **kwargs):
    try:
        return _real_replace(src, dst, *args, **kwargs)
    except OSError as exc:
        if exc.errno != errno.EXDEV or args or kwargs:
            raise

    dst_path = os.fspath(dst)
    dst_dir = os.path.dirname(dst_path) or "."
    fd, temporary_path = tempfile.mkstemp(prefix=".afs-replace-", dir=dst_dir)
    os.close(fd)

    try:
        shutil.copyfile(src, temporary_path)
        _real_replace(temporary_path, dst)
        os.unlink(src)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


os.replace = _replace
