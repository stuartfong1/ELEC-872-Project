from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[1]
image = Image.new('RGB', (1280, 570), '#f2f5f9')
draw = ImageDraw.Draw(image)
font_path = Path('C:/Windows/Fonts/segoeui.ttf')
bold_path = Path('C:/Windows/Fonts/segoeuib.ttf')
font = ImageFont.truetype(str(font_path), 16)
bold = ImageFont.truetype(str(bold_path), 18)
title = ImageFont.truetype(str(bold_path), 28)
draw.text((40, 25), 'ELEC872 - Phase 1 workflow', font=title, fill='#172a43')
draw.text((40, 66), 'Real source photos -> independent conditions -> model-specific features', font=font, fill='#536273')

boxes = [
    ((40, 145, 260, 275), 'Kaggle VGGFace2', ['Immutable original ZIP', 'Source / person / splits']),
    ((310, 145, 530, 275), 'RGB reference', ['Existing crop, fixed extent', 'Lossless PNG; no new JPEG']),
    ((580, 145, 830, 275), 'Independent conditions', ['IDENTITY + six JPEG settings', 'Optional supplied processors']),
    ((880, 125, 1230, 295), 'Model inputs + encoders', ['FaRL ep16 - projected CLS', 'LVFace-T - identity embedding', 'ResNet50 - trained avgpool']),
    ((310, 365, 530, 490), 'Own ResNet50', ['Identity loss + backprop', 'Validate / save / resume']),
    ((580, 365, 830, 490), 'Features + row mapping', ['raw.npy / l2.npy / rows.csv', 'Weight hash / layer / labels']),
    ((880, 365, 1230, 490), 'Integrity validation', ['Linkage / splits / labels / finite values', 'Phase 2 handoff; no probe training']),
]
for (x1, y1, x2, y2), heading, lines in boxes:
    draw.rounded_rectangle((x1, y1, x2, y2), radius=12, fill='white', outline='#c6d2df', width=2)
    draw.text((x1 + 16, y1 + 18), heading, font=bold, fill='#172a43')
    for i, line in enumerate(lines):
        draw.text((x1 + 16, y1 + 56 + i * 28), line, font=font, fill='#536273')

def arrow(points):
    draw.line(points, fill='#536273', width=3)
    x, y = points[-1]
    px, py = points[-2]
    if x > px:
        tip = [(x, y), (x - 10, y - 6), (x - 10, y + 6)]
    elif y > py:
        tip = [(x, y), (x - 6, y - 10), (x + 6, y - 10)]
    else:
        tip = [(x, y), (x - 6, y + 10), (x + 6, y + 10)]
    draw.polygon(tip, fill='#536273')

for points in [[(260, 210), (300, 210)], [(530, 210), (570, 210)], [(830, 210), (870, 210)],
               [(420, 275), (420, 355)], [(1050, 295), (1050, 325), (705, 325), (705, 355)],
               [(530, 400), (550, 400), (550, 310), (930, 310), (930, 298)], [(830, 425), (870, 425)]]:
    arrow(points)
draw.text((563, 286), 'Trained checkpoint', font=ImageFont.truetype(str(font_path), 14), fill='#536273')
draw.text((40, 529), 'Face-training photo splits and probe identity splits differ. All variants inherit source splits.', font=font, fill='#536273')
image.save(root / 'reports/workflow.png')
