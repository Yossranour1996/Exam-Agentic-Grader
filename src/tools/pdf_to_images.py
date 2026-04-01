from pathlib import Path
from pdf2image import convert_from_path

def pdf_to_images(pdf_path: str | Path, out_dir: str | Path, dpi: int = 300) -> list[str]:
    pdf_path = str(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = convert_from_path(pdf_path, dpi=dpi)
    out_paths = []
    for i, img in enumerate(pages, start=1):
        p = out_dir / f"page_{i:02d}.png"
        img.save(p, "PNG")
        out_paths.append(str(p))
    return out_paths
