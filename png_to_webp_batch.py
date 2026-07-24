from pathlib import Path
from io import BytesIO
from PIL import Image, ImageOps
import sys


# ============================================================
# SETTINGS
# ============================================================

TARGET_KB = 800
MAX_KB = 1024

TARGET_BYTES = TARGET_KB * 1024
MAX_BYTES = MAX_KB * 1024

OUTPUT_FOLDER = Path.home() / "Downloads" / "OUTPUT"

QUALITY_START = 95
QUALITY_MIN = 10
QUALITY_STEP = 5

RESIZE_STEP = 0.90
MIN_WIDTH = 300
MIN_HEIGHT = 300


# ============================================================
# FUNCTIONS
# ============================================================

def clean_input_path(raw_path: str) -> Path:
    """
    Handles paths with spaces and special characters.
    Also handles pasted paths wrapped in quotes.
    """
    raw_path = raw_path.strip().strip('"').strip("'")
    return Path(raw_path).expanduser().resolve()


def prepare_image(img: Image.Image) -> Image.Image:
    """
    Keeps transparency if the PNG has it.
    Converts unsupported modes safely for WebP.
    """
    img = ImageOps.exif_transpose(img)

    has_alpha = (
        img.mode in ("RGBA", "LA")
        or (img.mode == "P" and "transparency" in img.info)
    )

    if has_alpha:
        return img.convert("RGBA")

    return img.convert("RGB")


def encode_webp(img: Image.Image, quality: int) -> bytes:
    buffer = BytesIO()
    img.save(
        buffer,
        format="WEBP",
        quality=quality,
        method=6,
        optimize=True
    )
    return buffer.getvalue()


def compress_to_webp(src_path: Path, dest_path: Path) -> tuple[bool, int, int, tuple[int, int]]:
    """
    Compresses WebP aiming for 800KB or lower.
    Hard target is 1MB or lower when possible.

    If quality compression alone cannot hit the target,
    it gradually resizes the image until it does.
    """
    with Image.open(src_path) as original:
        img = prepare_image(original)

    current_img = img.copy()

    while True:
        best_under_target = None
        best_under_max = None

        for quality in range(QUALITY_START, QUALITY_MIN - 1, -QUALITY_STEP):
            data = encode_webp(current_img, quality)
            size = len(data)

            if size <= TARGET_BYTES:
                best_under_target = (data, quality, size)
                break

            if size <= MAX_BYTES and best_under_max is None:
                best_under_max = (data, quality, size)

        if best_under_target:
            data, quality, size = best_under_target
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(data)
            return True, quality, size, current_img.size

        if best_under_max:
            data, quality, size = best_under_max
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(data)
            return True, quality, size, current_img.size

        width, height = current_img.size

        new_width = int(width * RESIZE_STEP)
        new_height = int(height * RESIZE_STEP)

        if new_width < MIN_WIDTH or new_height < MIN_HEIGHT:
            data = encode_webp(current_img, QUALITY_MIN)
            size = len(data)

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(data)

            return size <= MAX_BYTES, QUALITY_MIN, size, current_img.size

        current_img = current_img.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )


def main():
    print("PNG to WebP batch converter")
    print("Input folder can contain spaces and special characters.")
    print()

    input_raw = input("Paste folder path: ")
    input_folder = clean_input_path(input_raw)

    if not input_folder.exists():
        print(f"\nERROR: Folder does not exist:\n{input_folder}")
        sys.exit(1)

    if not input_folder.is_dir():
        print(f"\nERROR: This is not a folder:\n{input_folder}")
        sys.exit(1)

    png_files = [
        file for file in input_folder.rglob("*")
        if file.is_file() and file.suffix.lower() == ".png"
    ]

    if not png_files:
        print("\nNo PNG files found.")
        sys.exit(0)

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    print(f"\nFound {len(png_files)} PNG files.")
    print(f"Output folder:\n{OUTPUT_FOLDER}")
    print()

    converted = 0
    failed = 0

    for index, src_path in enumerate(png_files, start=1):
        try:
            relative_path = src_path.relative_to(input_folder)
            dest_path = OUTPUT_FOLDER / relative_path.with_suffix(".webp")

            success, quality, size, final_dimensions = compress_to_webp(src_path, dest_path)

            size_kb = size / 1024
            status = "OK" if success else "OVER 1MB"

            print(
                f"[{index}/{len(png_files)}] {status} | "
                f"{size_kb:.1f} KB | "
                f"Q{quality} | "
                f"{final_dimensions[0]}x{final_dimensions[1]} | "
                f"{relative_path}"
            )

            converted += 1

        except Exception as e:
            failed += 1
            print(f"[{index}/{len(png_files)}] FAILED | {src_path}")
            print(f"Reason: {e}")

    print()
    print("Done.")
    print(f"Converted: {converted}")
    print(f"Failed: {failed}")
    print(f"Saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()