"""Draw desktop/assets/fly_track.ico (and a 256 px png preview).

    python desktop/make_icon.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent / "assets"
S = 1024


def draw() -> Image.Image:
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((32, 32, S - 32, S - 32), radius=200, fill=(17, 21, 30, 255),
                        outline=(56, 189, 248, 255), width=28)

    # route on the floor plan
    d.line([(170, 820), (330, 640), (520, 700), (700, 470), (860, 300)],
           fill=(250, 204, 21, 255), width=46, joint="curve")
    for x, y in ((170, 820), (860, 300)):
        d.ellipse((x - 46, y - 46, x + 46, y + 46), fill=(250, 204, 21, 255))

    # fly: wings, body, head
    wing = (186, 230, 253, 215)
    d.ellipse((300, 250, 520, 560), fill=wing)
    d.ellipse((504, 250, 724, 560), fill=wing)
    d.ellipse((430, 330, 594, 690), fill=(12, 14, 20, 255), outline=(56, 189, 248, 255), width=18)
    d.ellipse((450, 230, 574, 350), fill=(12, 14, 20, 255), outline=(56, 189, 248, 255), width=18)
    d.ellipse((462, 262, 500, 300), fill=(244, 63, 94, 255))
    d.ellipse((524, 262, 562, 300), fill=(244, 63, 94, 255))
    return im


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    im = draw()
    im.save(OUT / "fly_track.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                                          (128, 128), (256, 256)])
    im.resize((256, 256), Image.LANCZOS).save(OUT / "fly_track.png")
    print(f"wrote {OUT / 'fly_track.ico'}")


if __name__ == "__main__":
    main()
