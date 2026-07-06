"""
Strip all embedded metadata from PNG files in the same folder as this script.

Removes:
- EXIF
- XMP
- PNG tEXt / zTXt / iTXt chunks
- AI generation prompts and workflow metadata
- Software/application tags
- ICC profiles and other optional PNG metadata
- Data appended after the PNG IEND chunk

The image is decoded and rebuilt from pixel data only.

Install dependency:
    py -m pip install Pillow

Run:
    py strip_png_metadata.py
"""

from __future__ import annotations

import os
import shutil
import struct
import tempfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError


# ============================================================
# SETTINGS
# ============================================================

# Keep copies of the original PNG files before overwriting them.
MAKE_BACKUPS = True

# Backup folder created beside this script.
BACKUP_FOLDER_NAME = "_png_originals"

# Skip PNG files inside the backup folder.
SKIP_BACKUP_FOLDER = True


# ============================================================
# PNG VERIFICATION
# ============================================================

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# These chunks contain the minimum image structure.
# tRNS may be produced for palette/grayscale transparency.
ALLOWED_CHUNKS = {
    b"IHDR",
    b"PLTE",
    b"IDAT",
    b"IEND",
    b"tRNS",
}


def read_png_chunks(file_path: Path) -> list[bytes]:
    """Return the chunk names found in a PNG file."""
    chunks: list[bytes] = []

    with file_path.open("rb") as file:
        if file.read(8) != PNG_SIGNATURE:
            raise ValueError("Invalid PNG signature")

        while True:
            length_bytes = file.read(4)
            if len(length_bytes) != 4:
                raise ValueError("Unexpected end of PNG file")

            length = struct.unpack(">I", length_bytes)[0]
            chunk_type = file.read(4)

            if len(chunk_type) != 4:
                raise ValueError("Invalid PNG chunk")

            chunks.append(chunk_type)

            # Skip chunk data and CRC.
            file.seek(length + 4, os.SEEK_CUR)

            if chunk_type == b"IEND":
                break

    return chunks


def find_unexpected_chunks(file_path: Path) -> list[str]:
    """Find remaining nonessential chunks after cleaning."""
    chunks = read_png_chunks(file_path)
    return [
        chunk.decode("latin-1", errors="replace")
        for chunk in chunks
        if chunk not in ALLOWED_CHUNKS
    ]


# ============================================================
# IMAGE CLEANING
# ============================================================

def rebuild_from_pixels(image: Image.Image) -> Image.Image:
    """
    Create a new Pillow image containing pixel data only.

    Palette images are converted to RGB/RGBA because their palette and
    transparency can otherwise carry additional attached information.
    """
    image.load()

    if image.mode == "P":
        if "transparency" in image.info:
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")
    elif image.mode not in {
        "1",
        "L",
        "LA",
        "RGB",
        "RGBA",
        "I",
        "I;16",
        "F",
    }:
        # PNG-compatible fallback that preserves transparency when present.
        image = image.convert("RGBA")

    clean_image = Image.frombytes(
        image.mode,
        image.size,
        image.tobytes(),
    )

    return clean_image


def clean_png(source_path: Path, backup_folder: Path | None) -> tuple[bool, str]:
    """Strip metadata from one PNG and overwrite it atomically."""
    if backup_folder is not None:
        backup_path = backup_folder / source_path.name

        # Avoid overwriting an existing backup from an earlier run.
        if not backup_path.exists():
            shutil.copy2(source_path, backup_path)

    temporary_path: Path | None = None

    try:
        with Image.open(source_path) as image:
            if getattr(image, "is_animated", False):
                return False, "Skipped animated PNG/APNG to avoid destroying animation"

            clean_image = rebuild_from_pixels(image)

            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{source_path.stem}_clean_",
                suffix=".png",
                dir=source_path.parent,
            )
            os.close(file_descriptor)
            temporary_path = Path(temporary_name)

            # Do not pass pnginfo, exif, icc_profile, or other metadata.
            clean_image.save(
                temporary_path,
                format="PNG",
                optimize=True,
            )
            clean_image.close()

        unexpected_chunks = find_unexpected_chunks(temporary_path)

        if unexpected_chunks:
            return (
                False,
                "Verification failed; remaining chunks: "
                + ", ".join(unexpected_chunks),
            )

        # Atomic replacement where supported.
        os.replace(temporary_path, source_path)
        temporary_path = None

        return True, "Metadata removed"

    except UnidentifiedImageError:
        return False, "Not a valid PNG image"
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    script_folder = Path(__file__).resolve().parent

    backup_folder: Path | None = None
    if MAKE_BACKUPS:
        backup_folder = script_folder / BACKUP_FOLDER_NAME
        backup_folder.mkdir(exist_ok=True)

    png_files = sorted(
        path
        for path in script_folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )

    if SKIP_BACKUP_FOLDER and backup_folder is not None:
        png_files = [
            path for path in png_files
            if backup_folder not in path.parents
        ]

    if not png_files:
        print("No PNG files were found beside the script.")
        return

    successful = 0
    failed = 0

    print(f"Found {len(png_files)} PNG file(s).\n")

    for png_path in png_files:
        success, message = clean_png(png_path, backup_folder)

        if success:
            successful += 1
            print(f"[CLEANED] {png_path.name}")
        else:
            failed += 1
            print(f"[SKIPPED] {png_path.name} - {message}")

    print("\nFinished.")
    print(f"Cleaned: {successful}")
    print(f"Skipped/failed: {failed}")

    if backup_folder is not None:
        print(f"Original backups: {backup_folder}")


if __name__ == "__main__":
    main()
