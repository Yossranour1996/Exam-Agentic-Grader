"""PDF page rasterization used before OCR."""
from pathlib import Path
from pdf2image import convert_from_path

from src.utils.logging import Logger


def pdf_to_images(pdf_path: str | Path, out_dir: str | Path, dpi: int = 300, logger: Logger | None = None) -> list[Path]:
    """Converts a PDF file into individual PNG images, one per page."""
    pdf_path = str(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        pages = convert_from_path(pdf_path, dpi=dpi)
        out_paths = []
        for i, img in enumerate(pages, start=1):
            p = out_dir / f"page_{i:02d}.png"
            img.save(p, "PNG")
            out_paths.append(p)
    except Exception as e:
        if logger:
            logger.log(
                entry=f"Error occurred while converting PDF to images: {e}",
                level='error',
                step_label='Extractor-pdf2imgs'
                )
        raise
    return out_paths
