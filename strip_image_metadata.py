"""
Recursively strip embedded metadata from PNG, JPG, and JPEG files in a chosen folder.

Removes metadata including:
- PNG EXIF/eXIf, XMP, text chunks, ICC profiles, gamma/chromaticity data,
  timestamps, DPI, comments, software tags, AI prompts/workflows, and bytes
  appended after IEND
- JPEG EXIF, XMP, ICC profiles, IPTC/Photoshop data, comments, application
  metadata, thumbnails, DPI metadata, and bytes appended after EOI

How it works:
- Accepts a folder path from the command line or an interactive prompt.
- Supports spaces, Unicode, and special characters in folder/file names.
- Recursively scans the selected folder and all nested folders.
- Decodes each image and rebuilds it from pixel data only.
- Applies JPEG EXIF orientation to the pixels before deleting the EXIF data, so
  the image keeps the same visible rotation.
- Verifies each rebuilt file before replacing the source.
- Uses atomic replacement, so a failed conversion does not destroy the source.
- Uses multiple worker threads.
- Does NOT create backup/original copies.

Important JPEG note:
- JPEG is a lossy format. Rebuilding a JPEG requires recompression. The script
  reuses the original quantization tables and chroma-subsampling when possible
  to minimize additional quality loss.

Dependency:
    py -m pip install Pillow

Examples:
    py strip_image_metadata.py "C:\\Images with spaces"
    py strip_image_metadata.py "D:\\AI Images & Exports" --workers 12

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

from PIL import Image, ImageOps, JpegImagePlugin, UnidentifiedImageError


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Static PNG chunks that contain image structure/pixel representation only.
STATIC_ALLOWED_PNG_CHUNKS = {
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
    image_type: str
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


def discover_image_files(folder: Path) -> tuple[list[Path], int]:
    """Recursively find supported images without following directory symlinks."""
    image_files: list[Path] = []
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
            if file_path.suffix.casefold() in SUPPORTED_EXTENSIONS:
                image_files.append(file_path)

    image_files.sort(key=lambda path: str(path).casefold())
    return image_files, skipped_symlinks


def inspect_png(file_path: Path, allow_animation: bool) -> list[str]:
    """
    Validate PNG chunk framing/CRC and return forbidden chunk names.

    Also rejects any bytes after IEND, which removes appended JSON/workflows or
    any other payload stored outside the formal PNG structure.
    """
    allowed_chunks = set(STATIC_ALLOWED_PNG_CHUNKS)
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


def inspect_jpeg(file_path: Path) -> list[str]:
    """
    Validate JPEG structure and return forbidden metadata markers.

    Rebuilt JPEGs may contain only:
    - APP0 JFIF with density units set to 0 (aspect ratio only, not DPI)
    - APP14 Adobe, which is structural for some color modes

    All other APP markers and COM comments are treated as metadata. Any bytes
    after the final EOI marker are rejected.
    """
    data = file_path.read_bytes()
    if not data.startswith(JPEG_SOI):
        raise ValueError("Invalid JPEG SOI signature")

    forbidden: list[str] = []
    position = 2
    saw_sos = False
    saw_eoi = False

    while position < len(data):
        if data[position] != 0xFF:
            raise ValueError(f"Expected JPEG marker at byte {position}")

        # Skip fill bytes between markers.
        while position < len(data) and data[position] == 0xFF:
            position += 1
        if position >= len(data):
            raise ValueError("Unexpected end of JPEG marker stream")

        marker = data[position]
        position += 1

        # Stuffed 0xFF byte is valid only inside entropy-coded scan data.
        if marker == 0x00:
            if not saw_sos:
                raise ValueError("Unexpected stuffed byte outside JPEG scan")
            continue

        if marker == 0xD9:  # EOI
            saw_eoi = True
            if position != len(data):
                raise ValueError("Data remains after the JPEG EOI marker")
            break

        # Standalone markers without a length field.
        if marker in {0x01, *range(0xD0, 0xD8), 0xD8}:
            continue

        if position + 2 > len(data):
            raise ValueError("Missing JPEG segment length")

        segment_length = struct.unpack(">H", data[position : position + 2])[0]
        if segment_length < 2:
            raise ValueError(f"Invalid JPEG segment length for marker FF{marker:02X}")

        segment_start = position + 2
        segment_end = position + segment_length
        if segment_end > len(data):
            raise ValueError(f"JPEG segment FF{marker:02X} exceeds file size")

        payload = data[segment_start:segment_end]
        position = segment_end

        if marker == 0xDA:  # SOS: scan until the next non-stuffed marker.
            saw_sos = True
            while position < len(data):
                next_ff = data.find(b"\xff", position)
                if next_ff < 0:
                    raise ValueError("JPEG scan data ends before EOI")
                position = next_ff

                # Preserve the marker prefix for the outer marker parser.
                probe = position + 1
                while probe < len(data) and data[probe] == 0xFF:
                    probe += 1
                if probe >= len(data):
                    raise ValueError("Unexpected end inside JPEG scan marker")

                scan_marker = data[probe]
                if scan_marker == 0x00:  # Escaped FF byte.
                    position = probe + 1
                    continue
                if 0xD0 <= scan_marker <= 0xD7:  # Restart marker.
                    position = probe + 1
                    continue

                # A real marker starts here; parse it in the outer loop.
                position = next_ff
                break
            continue

        if marker == 0xFE:  # COM
            forbidden.append("COM")
        elif 0xE0 <= marker <= 0xEF:  # APP0 through APP15
            app_number = marker - 0xE0
            marker_name = f"APP{app_number}"

            if marker == 0xE0:
                # Pillow normally writes a basic JFIF APP0 segment. It is
                # allowed only when it does not declare physical DPI units.
                if not payload.startswith(b"JFIF\x00"):
                    forbidden.append(marker_name)
                elif len(payload) < 12:
                    raise ValueError("Truncated JFIF APP0 segment")
                elif payload[7] != 0:
                    forbidden.append("APP0/JFIF-DPI")
                elif len(payload) >= 14 and (payload[12] != 0 or payload[13] != 0):
                    forbidden.append("APP0/JFIF-thumbnail")
            elif marker == 0xEE:
                # Adobe APP14 is structural, especially for CMYK/YCCK JPEGs.
                if not payload.startswith(b"Adobe"):
                    forbidden.append(marker_name)
            else:
                forbidden.append(marker_name)

    if not saw_sos:
        raise ValueError("JPEG is missing SOS image data")
    if not saw_eoi:
        raise ValueError("JPEG is missing EOI")

    return forbidden


def rebuild_png_image(image: Image.Image) -> Image.Image:
    """Return a new PNG-compatible image made only from decoded pixel bytes."""
    image.load()
    converted: Image.Image | None = None

    try:
        if image.mode == "P":
            converted = image.convert("RGBA" if "transparency" in image.info else "RGB")
            pixel_source = converted
        elif image.mode in {"1", "L", "LA", "RGB", "RGBA", "I", "I;16"}:
            pixel_source = image
        else:
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


def rebuild_jpeg_image(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation and return metadata-free JPEG-compatible pixels."""
    oriented = ImageOps.exif_transpose(image)
    converted: Image.Image | None = None

    try:
        oriented.load()
        if oriented.mode in {"L", "RGB", "CMYK"}:
            pixel_source = oriented
        else:
            # JPEG has no alpha channel. Composite transparency onto white.
            if oriented.mode in {"RGBA", "LA", "P"}:
                rgba = oriented.convert("RGBA")
                try:
                    background = Image.new("RGB", rgba.size, "white")
                    background.paste(rgba, mask=rgba.getchannel("A"))
                    converted = background
                finally:
                    rgba.close()
            else:
                converted = oriented.convert("RGB")
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
        if oriented is not image:
            oriented.close()


def rebuild_animation_frames(
    image: Image.Image,
) -> tuple[list[Image.Image], list[int | float], int]:
    """Rebuild every APNG frame as a full RGBA pixel canvas."""
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
        if image.format != "PNG":
            raise ValueError(f"File extension is PNG, but detected format is {image.format!r}")

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
            clean_image = rebuild_png_image(image)
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


def save_clean_jpeg(source_path: Path, temporary_path: Path) -> None:
    """Decode, orient, rebuild, and save one JPG/JPEG without metadata."""
    with Image.open(source_path) as image:
        if image.format != "JPEG":
            raise ValueError(f"File extension is JPEG, but detected format is {image.format!r}")

        image.load()
        progressive = bool(image.info.get("progressive") or image.info.get("progression"))

        try:
            sampling = JpegImagePlugin.get_sampling(image)
        except Exception:
            sampling = -1

        original_quantization = getattr(image, "quantization", None)
        qtables = None
        if isinstance(original_quantization, dict) and original_quantization:
            # Copy the table values so they remain valid after the source closes.
            qtables = {
                int(table_id): list(values)
                for table_id, values in original_quantization.items()
            }

        clean_image = rebuild_jpeg_image(image)
        try:
            save_options: dict[str, object] = {
                "format": "JPEG",
                "progressive": progressive,
                "optimize": False,
            }

            if sampling in {0, 1, 2}:
                save_options["subsampling"] = sampling
            else:
                save_options["subsampling"] = "keep"

            if qtables:
                save_options["qtables"] = qtables
            else:
                save_options["quality"] = 95

            clean_image.save(temporary_path, **save_options)
        finally:
            clean_image.close()


def clean_image(source_path: Path) -> CleanResult:
    """Strip one supported image and atomically overwrite after verification."""
    temporary_path: Path | None = None
    extension = source_path.suffix.casefold()
    image_type = "PNG" if extension == ".png" else "JPEG"

    try:
        source_mode = stat.S_IMODE(source_path.stat().st_mode)

        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=".image_metadata_clean_",
            suffix=".tmp",
            dir=source_path.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)

        animated = False
        if extension == ".png":
            animated = save_clean_png(source_path, temporary_path)
            forbidden_items = inspect_png(temporary_path, allow_animation=animated)
        elif extension in {".jpg", ".jpeg"}:
            save_clean_jpeg(source_path, temporary_path)
            forbidden_items = inspect_jpeg(temporary_path)
        else:
            return CleanResult(
                source_path,
                False,
                f"Unsupported extension: {extension}",
                image_type,
            )

        # Reopen and fully decode the rebuilt image before replacing the source.
        with Image.open(temporary_path) as verification_image:
            verification_image.load()
            if extension == ".png" and verification_image.format != "PNG":
                raise ValueError("Rebuilt file did not verify as PNG")
            if extension in {".jpg", ".jpeg"} and verification_image.format != "JPEG":
                raise ValueError("Rebuilt file did not verify as JPEG")

        if forbidden_items:
            unique_items = sorted(set(forbidden_items))
            return CleanResult(
                source_path,
                False,
                "Verification rejected remaining metadata marker(s): "
                + ", ".join(unique_items),
                image_type,
                animated,
            )

        # Preserve basic file permissions, not embedded image metadata.
        try:
            os.chmod(temporary_path, source_mode)
        except OSError:
            pass

        # Atomic on the same filesystem. The source is untouched until here.
        os.replace(temporary_path, source_path)
        temporary_path = None

        if animated:
            message = "APNG animation metadata removed"
        else:
            message = f"{image_type} metadata removed"
        return CleanResult(source_path, True, message, image_type, animated)

    except UnidentifiedImageError:
        return CleanResult(
            source_path,
            False,
            f"Not a readable {image_type} image",
            image_type,
        )
    except PermissionError as error:
        return CleanResult(
            source_path,
            False,
            f"PermissionError: {error}",
            image_type,
        )
    except Exception as error:
        return CleanResult(
            source_path,
            False,
            f"{type(error).__name__}: {error}",
            image_type,
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
            "Recursively remove embedded metadata from PNG, JPG, and JPEG "
            "files and overwrite them safely without creating backups."
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

    image_files, skipped_symlinks = discover_image_files(folder)
    png_count = sum(path.suffix.casefold() == ".png" for path in image_files)
    jpeg_count = len(image_files) - png_count

    print(f"\nFolder: {folder}")
    print("Mode: recursive, overwrite originals, no backups")
    print(f"Worker threads: {args.workers}")
    print(f"Supported images found: {len(image_files)}")
    print(f"PNG files found: {png_count}")
    print(f"JPG/JPEG files found: {jpeg_count}")
    if skipped_symlinks:
        print(f"Symlinks skipped for safety: {skipped_symlinks}")

    if not image_files:
        print("No PNG, JPG, or JPEG files were found.")
        return 0

    cleaned = 0
    failed = 0
    png_cleaned = 0
    jpeg_cleaned = 0
    animated_cleaned = 0

    try:
        with ThreadPoolExecutor(
            max_workers=args.workers,
            thread_name_prefix="image-cleaner",
        ) as executor:
            future_to_path = {
                executor.submit(clean_image, image_path): image_path
                for image_path in image_files
            }

            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                except Exception as error:
                    image_type = "PNG" if path.suffix.casefold() == ".png" else "JPEG"
                    result = CleanResult(
                        path,
                        False,
                        f"Worker failure: {type(error).__name__}: {error}",
                        image_type,
                    )

                relative_path = result.path.relative_to(folder)
                if result.success:
                    cleaned += 1
                    if result.image_type == "PNG":
                        png_cleaned += 1
                    else:
                        jpeg_cleaned += 1
                    if result.animated:
                        animated_cleaned += 1
                    print(f"[CLEANED] {relative_path}")
                else:
                    failed += 1
                    print(f"[FAILED]  {relative_path} - {result.message}")

    except KeyboardInterrupt:
        print(
            "\nCancelled. Files already replaced remain cleaned; "
            "unfinished files remain unchanged."
        )
        return 130

    print("\nFinished.")
    print(f"Cleaned: {cleaned}")
    print(f"PNG files cleaned: {png_cleaned}")
    print(f"Animated PNGs cleaned: {animated_cleaned}")
    print(f"JPG/JPEG files cleaned: {jpeg_cleaned}")
    print(f"Failed: {failed}")
    print("Backup copies created: 0")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
