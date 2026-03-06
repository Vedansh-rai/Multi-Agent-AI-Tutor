from PIL import Image, ImageDraw, ImageFont
import urllib.request
import os

img = Image.new('RGB', (800, 400), color=(255, 255, 255))
d = ImageDraw.Draw(img)
# Use a default font
d.text((50, 150), "Find the maximum value of:\nf(x) = -x^2 + 4x + 1", fill=(0,0,0), font=None, font_size=50)
img.save('test_math.png')

os.system('say "Find the maximum value of f of x equals negative x squared plus four x plus one" -o test_math.m4a')
