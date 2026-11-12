"""Make test images of known sizes (a gradient with a few shapes)."""
import sys

from PIL import Image, ImageDraw

SIZES = [(224, 224), (448, 448), (896, 896), (1280, 720), (1920, 1080)]
for w, h in SIZES:
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = (x * 255 // w, y * 255 // h, 128)
    d = ImageDraw.Draw(img)
    d.rectangle([w // 8, h // 8, w // 2, h // 2], outline="white", width=4)
    d.ellipse([w // 2, h // 2, w * 7 // 8, h * 7 // 8], fill="red")
    img.save(f"img_{w}x{h}.png")
    print("wrote", f"img_{w}x{h}.png")
