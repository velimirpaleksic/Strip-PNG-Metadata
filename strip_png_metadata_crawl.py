"""
Recursively strip embedded metadata from every PNG file in a chosen folder.

Removes metadata stored in PNG chunks, including:
- EXIF / eXIf
- XMP
- tEXt, zTXt, and iTXt
- AI prompts, workflows, generation parameters, and application tags
- ICC profiles, gamma/chromaticity data, timestamps, DPI, comments, and software tags
- Any bytes appended after the PNG IEND chunk

How it works:
- Accepts a folder path from the command line or an interactive prompt.
- Supports spaces, Unicode, and special characters in folder/file names.
- Recursively scans the selected folder and all nested folders.
- Decodes each image and rebuilds it from pixel data only.
- Verifies the rebuilt PNG before replacing the source file.
- Uses atomic replacement, so a failed conversion does not destroy the source.
- Uses multiple worker threads.
- Does NOT create backup/original copies.

Dependency:
    py -m pip install Pillow

Examples:
    py strip_png_metadata.py "C:\\Images with spaces"
    py strip_png_metadata.py "D:\\AI Images & Exports" --workers 12

You may also run the script without arguments and paste or drag a folder path
into the prompt.
"""

from __future__ import annotations

import argparse
import os
import stat
import struct
import tempfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Static PNG chunks that contain image structure/pixel representation only.
STATIC_ALLOWED_CHUNKS = {
    b"IHDR",
    b"PLTE",
    b"IDAT",
    b"IEND",
    b"tRNS",
}

# APNG animation-control chunks. These are required to preserve animation and
# do not contain prompts, EXIF, XMP, software tags, or other descriptive data.
APNG_ALLOWED_CHUNKS = {
    b"acTL",
    b"fcTL",
    b"fdAT",
}

DEFAULT_WORKERS = min(32, max(2, (os.cpu_count() or 4) + 4))


@dataclass(frozen=True)
class CleanResult:
    path: Path
    success: bool
    message: str
    animated: bool = False


def remove_matching_outer_quotes(value: str) -> str:
    """Remove one matching pair of quotes, useful for dragged Windows paths."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def resolve_input_folder(raw_value: str) -> Path:
    """Normalize a user-supplied folder path without breaking special chars."""
    cleaned = remove_matching_outer_quotes(raw_value)
    if not cleaned:
        raise ValueError("No folder path was provided.")

    expanded = os.path.expandvars(os.path.expanduser(cleaned))
    folder = Path(expanded).resolve(strict=True)

    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    return folder


def discover_png_files(folder: Path) -> tuple[list[Path], int]:
    """Recursively find PNG files without following directory symlinks."""
    png_files: list[Path] = []
    skipped_symlinks = 0

    for root, directory_names, file_names in os.walk(folder, followlinks=False):
        root_path = Path(root)

        # Explicitly prune symlinked directories for predictable/safe scope.
        kept_directories: list[str] = []
        for directory_name in directory_names:
            directory_path = root_path / directory_name
            if directory_path.is_symlink():
                skipped_symlinks += 1
            else:
                kept_directories.append(directory_name)
        directory_names[:] = kept_directories

        for file_name in file_names:
            file_path = root_path / file_name
            if file_path.is_symlink():
                skipped_symlinks += 1
                continue
            if file_path.suffix.casefold() == ".png":
                png_files.append(file_path)

    png_files.sort(key=lambda path: str(path).casefold())
    return png_files, skipped_symlinks


def inspect_png(file_path: Path, allow_animation: bool) -> list[str]:
    """
    Validate PNG chunk framing/CRC and return forbidden chunk names.

    Also rejects any bytes after IEND, which removes appended JSON/workflows or
    any other payload stored outside the formal PNG structure.
    """
    allowed_chunks = set(STATIC_ALLOWED_CHUNKS)
    if allow_animation:
        allowed_chunks.update(APNG_ALLOWED_CHUNKS)

    forbidden: list[str] = []
    saw_ihdr = False
    saw_idat_or_fdat = False
    saw_iend = False

    with file_path.open("rb") as file:
        if file.read(8) != PNG_SIGNATURE:
            raise ValueError("Invalid PNG signature")

        while not saw_iend:
            length_bytes = file.read(4)
            if len(length_bytes) != 4:
                raise ValueError("Unexpected end of PNG before IEND")

            data_length = struct.unpack(">I", length_bytes)[0]
            chunk_type = file.read(4)
            if len(chunk_type) != 4:
                raise ValueError("Invalid PNG chunk header")

            chunk_data = file.read(data_length)
            if len(chunk_data) != data_length:
                raise ValueError("Unexpected end of PNG chunk data")

            crc_bytes = file.read(4)
            if len(crc_bytes) != 4:
                raise ValueError("Missing PNG chunk CRC")

            expected_crc = struct.unpack(">I", crc_bytes)[0]
            actual_crc = zlib.crc32(chunk_type)
            actual_crc = zlib.crc32(chunk_data, actual_crc) & 0xFFFFFFFF
            if actual_crc != expected_crc:
                name = chunk_type.decode("latin-1", errors="replace")
                raise ValueError(f"CRC verification failed for chunk {name}")

            if chunk_type == b"IHDR":
                if saw_ihdr:
                    raise ValueError("PNG contains more than one IHDR chunk")
                saw_ihdr = True
            elif chunk_type in {b"IDAT", b"fdAT"}:
                saw_idat_or_fdat = True
            elif chunk_type == b"IEND":
                if data_length != 0:
                    raise ValueError("IEND chunk must be empty")
                saw_iend = True

            if chunk_type not in allowed_chunks:
                forbidden.append(chunk_type.decode("latin-1", errors="replace"))

        if file.read(1):
            raise ValueError("Data remains after the IEND chunk")

    if not saw_ihdr:
        raise ValueError("PNG is missing IHDR")
    if not saw_idat_or_fdat:
        raise ValueError("PNG is missing image data")
    if not saw_iend:
        raise ValueError("PNG is missing IEND")

    return forbidden


def rebuild_static_image(image: Image.Image) -> Image.Image:
    """Return a new image made only from decoded pixel bytes."""
    image.load()
    converted: Image.Image | None = None

    try:
        if image.mode == "P":
            converted = image.convert("RGBA" if "transparency" in image.info else "RGB")
            pixel_source = converted
        elif image.mode in {"1", "L", "LA", "RGB", "RGBA", "I", "I;16"}:
            pixel_source = image
        else:
            # Safe PNG-compatible fallback. This also handles unusual modes.
            converted = image.convert("RGBA")
            pixel_source = converted

        clean_image = Image.frombytes(
            pixel_source.mode,
            pixel_source.size,
            pixel_source.tobytes(),
        )
        clean_image.info.clear()
        return clean_image
    finally:
        if converted is not None:
            converted.close()


def rebuild_animation_frames(
    image: Image.Image,
) -> tuple[list[Image.Image], list[int | float], int]:
    """
    Rebuild every APNG frame as a full RGBA pixel canvas.

    Full canvases avoid carrying original frame-level metadata. The visible
    animation is preserved, though the rebuilt APNG can be larger than the
    original because frame rectangles are expanded to the full canvas.
    """
    frames: list[Image.Image] = []
    durations: list[int | float] = []
    loop = int(image.info.get("loop", 0) or 0)

    for frame_index in range(int(getattr(image, "n_frames", 1))):
        image.seek(frame_index)
        duration = image.info.get("duration", 0)
        if not isinstance(duration, (int, float)):
            duration = 0

        rgba = image.convert("RGBA")
        try:
            clean_frame = Image.frombytes("RGBA", rgba.size, rgba.tobytes())
            clean_frame.info.clear()
            frames.append(clean_frame)
            durations.append(duration)
        finally:
            rgba.close()

    return frames, durations, loop


def save_clean_png(source_path: Path, temporary_path: Path) -> bool:
    """Decode/rebuild/save one PNG. Return True when it is animated."""
    with Image.open(source_path) as image:
        animated = bool(
            getattr(image, "is_animated", False)
            and int(getattr(image, "n_frames", 1)) > 1
        )

        if animated:
            frames, durations, loop = rebuild_animation_frames(image)
            if not frames:
                raise ValueError("Animated PNG contains no readable frames")

            try:
                frames[0].save(
                    temporary_path,
                    format="PNG",
                    save_all=True,
                    append_images=frames[1:],
                    duration=durations,
                    loop=loop,
                    disposal=0,
                    blend=0,
                    optimize=False,
                    compress_level=9,
                )
            finally:
                for frame in frames:
                    frame.close()
        else:
            clean_image = rebuild_static_image(image)
            try:
                clean_image.save(
                    temporary_path,
                    format="PNG",
                    optimize=True,
                    compress_level=9,
                )
            finally:
                clean_image.close()

    return animated


def clean_png(source_path: Path) -> CleanResult:
    """Strip one PNG and atomically overwrite it only after verification."""
    temporary_path: Path | None = None

    try:
        source_mode = stat.S_IMODE(source_path.stat().st_mode)

        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=".png_metadata_clean_",
            suffix=".tmp",
            dir=source_path.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)

        animated = save_clean_png(source_path, temporary_path)

        forbidden_chunks = inspect_png(
            temporary_path,
            allow_animation=animated,
        )
        if forbidden_chunks:
            unique_chunks = sorted(set(forbidden_chunks))
            return CleanResult(
                source_path,
                False,
                "Verification rejected remaining chunk(s): "
                + ", ".join(unique_chunks),
                animated,
            )

        # Preserve basic file permissions, not embedded image metadata.
        try:
            os.chmod(temporary_path, source_mode)
        except OSError:
            # Permission copying is best-effort, especially on Windows.
            pass

        # Atomic on the same filesystem. The source is untouched until here.
        os.replace(temporary_path, source_path)
        temporary_path = None

        kind = "APNG animation" if animated else "PNG"
        return CleanResult(source_path, True, f"{kind} metadata removed", animated)

    except UnidentifiedImageError:
        return CleanResult(source_path, False, "Not a readable PNG image")
    except PermissionError as error:
        return CleanResult(source_path, False, f"PermissionError: {error}")
    except Exception as error:
        return CleanResult(
            source_path,
            False,
            f"{type(error).__name__}: {error}",
        )
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively remove embedded metadata from PNG files and overwrite "
            "them safely without creating backups."
        )
    )
    parser.add_argument(
        "folder",
        nargs="?",
        help="Folder to scan recursively. Quotes are supported for paths with spaces.",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Worker threads to use (default: {DEFAULT_WORKERS}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    if args.workers < 1:
        print("Error: --workers must be at least 1.")
        return 2

    raw_folder = args.folder
    if raw_folder is None:
        try:
            raw_folder = input("Folder to scan recursively: ")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return 130

    try:
        folder = resolve_input_folder(raw_folder)
    except (OSError, ValueError) as error:
        print(f"Error: {error}")
        return 2

    png_files, skipped_symlinks = discover_png_files(folder)

    print(f"\nFolder: {folder}")
    print("Mode: recursive, overwrite originals, no backups")
    print(f"Worker threads: {args.workers}")
    print(f"PNG files found: {len(png_files)}")
    if skipped_symlinks:
        print(f"Symlinks skipped for safety: {skipped_symlinks}")

    if not png_files:
        print("No PNG files were found.")
        return 0

    cleaned = 0
    failed = 0
    animated_cleaned = 0

    try:
        with ThreadPoolExecutor(
            max_workers=args.workers,
            thread_name_prefix="png-cleaner",
        ) as executor:
            future_to_path = {
                executor.submit(clean_png, png_path): png_path
                for png_path in png_files
            }

            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = CleanResult(
                        path,
                        False,
                        f"Worker failure: {type(error).__name__}: {error}",
                    )

                relative_path = result.path.relative_to(folder)
                if result.success:
                    cleaned += 1
                    if result.animated:
                        animated_cleaned += 1
                    print(f"[CLEANED] {relative_path}")
                else:
                    failed += 1
                    print(f"[FAILED]  {relative_path} - {result.message}")

    except KeyboardInterrupt:
        print("\nCancelled. Files already replaced remain cleaned; unfinished files remain unchanged.")
        return 130

    print("\nFinished.")
    print(f"Cleaned: {cleaned}")
    print(f"Animated PNGs cleaned: {animated_cleaned}")
    print(f"Failed: {failed}")
    print("Backup copies created: 0")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
