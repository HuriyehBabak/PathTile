from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import fitz
from PIL import Image
import math, os

OUT='/mnt/data/cuda_kernel_designs_code_checked.pdf'
PREVIEW='/mnt/data/cuda_kernel_designs_code_checked_preview.png'
REG='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
pdfmetrics.registerFont(TTFont('DVS',REG))
pdfmetrics.registerFont(TTFont('DVS-Bold',BOLD))
W,H = 1120,655
SIDE=112
CW=252
Y_HEAD=62
Y_FWD=280
Y_DK=523
Y_KEY=585
COLORS={
 'naive':('#E9E9E9','#555555'),
 'gmc':('#CEE9FA','#277EB4'),
 'shared':('#D6F7DB','#368C43'),
 'warp':('#FADECC','#D2753A'),
 'label':'#DDECF8',
 'step_read':'#FCE4DF',
 'step_compute':'#DDEDFB',
 'step_write':'#EEE4F9',
 'step_reuse':'#DFF4E5',
 'ink':'#1A252F',
 'grid':'#C3CAD0',
 'line':'#B7C8D2',
}
c = canvas.Canvas(OUT,pagesize=(W,H),pageCompression=1)
c.setTitle('Figure 2: Code-checked GPU Execution Designs')

# Layout helpers (top-left coordinate system)
def rect(x,y,w,h,fill='#FFFFFF',stroke=None,r=0,lw=1):
    if fill: c.setFillColor(HexColor(fill))
    if stroke: c.setStrokeColor(HexColor(stroke)); c.setLineWidth(lw)
    if r:
        c.roundRect(x,H-y-h,w,h,r,stroke=int(stroke is not None),fill=int(fill is not None))
    else:
        c.rect(x,H-y-h,w,h,stroke=int(stroke is not None),fill=int(fill is not None))

def rule(x1,y1,x2,y2,color='#B7C8D2',lw=1):
    c.setStrokeColor(HexColor(color)); c.setLineWidth(lw)
    c.line(x1,H-y1,x2,H-y2)

def split_to_width(s,font,size,maxw):
    # Standard width-based word-wrapping with explicit newlines
    if not s: return ['']
    parts=[]
    for para in s.split('\n'):
        words=para.split(' ')
        line=''
        for word in words:
            new=f'{line} {word}' if line else word
            if pdfmetrics.stringWidth(new,font,size) <= maxw or not line:
                line=new
            else:
                parts.append(line)
                line=word
        parts.append(line)
    return parts

text_boxes=[]
def text(s,x,y,w,h,size=17,bold=False,color='#1A252F',align='left',max_lines=None,min_font=12,pad=0,leading_factor=1.16,ref=''):
    font='DVS-Bold' if bold else 'DVS'
    # Fit both width and height using bounding of reported lines, rather than guessing
    for attempt in range(40):
        lines=split_to_width(s,font,size,w-2*pad)
        leading=size*leading_factor
        if (max_lines is None or len(lines)<=max_lines) and (len(lines)*leading <= h-2*pad+0.05) and all(pdfmetrics.stringWidth(line,font,size)<=w-2*pad+.1 for line in lines):
            break
        size-=0.3
        if size<min_font: raise ValueError(f'Cannot fit {ref or s!r}: size {size} lines {lines}, {w}x{h}')
    yy=y+(h-len(lines)*leading)/2 + (leading-size)/2
    c.setFillColor(HexColor(color)); c.setFont(font,size)
    for idx,line in enumerate(lines):
        sw=pdfmetrics.stringWidth(line,font,size)
        if align=='center': xx=x+(w-sw)/2
        elif align=='right': xx=x+w-pad-sw
        else: xx=x+pad
        base_y=yy+idx*leading+size
        c.drawString(xx,H-base_y,line)
        text_boxes.append((ref,x,y,w,h,xx,yy+idx*leading,sw,leading,line))
    return len(lines),size

def gridicon(x,y,accent,selected,style='block'):
    S=10.2; GAP=2.0
    rgb=HexColor(accent)
    for r in range(5):
        for col in range(5):
            on=(r,col) in selected
            rect(x+col*(S+GAP),y+r*(S+GAP),S,S,fill=accent if on else '#FFFFFF',stroke='#C9D2D7',lw=.75)

cols=['Naive','GMC','Shared-Memory-Tiled','Warp-Tiled']
subheads=['Thread-per-output','Global-memory-coalesced','Cooperative staging','Warp-centric execution']
keys=['naive','gmc','shared','warp']

# Table colored strip and backgrounds
rect(0,0,SIDE,Y_HEAD,fill='#FFFFFF')
for i,k in enumerate(keys):
    x=SIDE+i*CW
    rect(x,0,CW,Y_HEAD,fill=COLORS[k][0],stroke='#AEC6D1',lw=1)
    headsize=[25,25,20.5,24.5][i]
    text(cols[i],x+8,5,CW-16,35,size=headsize,bold=True,align='center',ref=f'header{i}')
    text(subheads[i],x+8,38,CW-16,20,size=16,align='center',ref=f'sub{i}')

rect(0,Y_HEAD,SIDE,Y_FWD-Y_HEAD,fill=COLORS['label'],stroke='#AEC6D1')
rect(0,Y_FWD,SIDE,Y_DK-Y_FWD,fill=COLORS['label'],stroke='#AEC6D1')
rect(0,Y_DK,SIDE,Y_KEY-Y_DK,fill=COLORS['label'],stroke='#AEC6D1')
rect(0,Y_KEY,SIDE,H-Y_KEY,fill=COLORS['label'],stroke='#AEC6D1')
text('FWD / dX',5,Y_HEAD+42,SIDE-10,44,size=23,bold=True,align='center',ref='label fwd')
text('Output-local\nK-tap sums',5,Y_HEAD+100,SIDE-10,77,size=15.5,align='center',ref='label fwd subtitle')
text('dK',5,Y_FWD+52,SIDE-10,42,size=25,bold=True,align='center',ref='label dk')
text('Gradient over\nbatch and time',5,Y_FWD+104,SIDE-10,71,size=15,align='center',ref='label dk subtitle')
text('Key idea',4,Y_DK+6,SIDE-8,49,size=17,bold=True,align='center',ref='label key')
text('Launch\nparameter',4,Y_KEY+5,SIDE-8,60,size=16.6,bold=True,align='center',ref='label launch')

fwd_descriptions=[
 'One thread owns one (b, h, t) output. Neighboring threads access nearby time steps.',
 'A warp covers 32 time positions; a block groups 16 channels per batch.',
 'GMC mapping retained. Each warp stages a temporal tile plus halo in shared memory.',
 'One warp owns one (b, h) sequence; each lane handles t and t + 32 if valid.',
]
dk_descriptions=[
 'One thread owns one (h, j) coefficient; reduction over B x L is serial.',
 'Same per-coefficient mapping as Naive; no shared memory or parallel reduction.',
 'One thread per (h, j); cache x and dY in 64-batch tiles; retain serial sum.',
 'One block per (h, batch tile). B_TILE=64; four tap sums per loop.',
]
fwd_steps=[
 ['Global x, dY, k','K-tap sum','Write y / dX'],
 ['Coalesced x / dY','Local K-tap sum','Write y / dX'],
 ['Shared tile + halo','Warp sync + reuse','Write y / dX'],
 ['Shared x / dY','Warp sync; 1-2 outputs','Write y / dX'],
]
dk_steps=[
 ['Global x + dY','Serial B x L sum','Write dK once'],
 ['Global x + dY','Serial B x L sum','Write dK once'],
 ['Shared x + dY','Block sync; serial sum','Write dK once'],
 ['Shared x + dY','4 sums + shuffles','Combine partials','Atomic add dK'],
]
fwd_highlights=[
 {(2,2)}, {(2,0),(2,1),(2,2),(2,3),(2,4)}, {(2,0),(2,1),(2,2),(2,3),(2,4)}, {(r,col) for r in (1,2,3) for col in (1,2,3)},
]
dk_highlights=[
 {(2,2)}, {(2,2)}, {(2,2)}, {(r,col) for r in (1,2,3) for col in (1,2,3)},
]

for i,k in enumerate(keys):
    x=SIDE+i*CW
    colcolor=COLORS[k][1]
    rect(x,Y_HEAD,CW,Y_FWD-Y_HEAD,fill='#F9FBFC',stroke='#BDCAD1')
    rect(x,Y_FWD,CW,Y_DK-Y_FWD,fill='#F9FBFC',stroke='#BDCAD1')
    text(fwd_descriptions[i],x+13,Y_HEAD+8,CW-26,78,size=17.0,min_font=15,align='left',ref=f'fwd desc{i}')
    gridicon(x+15,Y_HEAD+126,colcolor,fwd_highlights[i])
    for j,step in enumerate(fwd_steps[i]):
        ys=Y_HEAD+91 + j*40
        rect(x+88,ys,151,40,fill=[COLORS['step_read'],COLORS['step_compute'],COLORS['step_write']][j],r=5)
        text(step,x+93,ys+1,141,38,size=17.5,min_font=15.5,align='center',ref=f'fwd step{i}-{j}')
    text(dk_descriptions[i],x+13,Y_FWD+9,CW-26,80,size=17,min_font=14.5,align='left',ref=f'dk desc{i}')
    gridicon(x+15,Y_FWD+151,colcolor,dk_highlights[i])
    # Four-stage pipelines have 4 shorter boxes; three-stage pipelines use 3 taller boxes
    n=len(dk_steps[i])
    if n==4: y0=Y_FWD+96; boxh=35; pitch=36.6
    else: y0=Y_FWD+107; boxh=38; pitch=44.5
    for j,step in enumerate(dk_steps[i]):
        ys=y0+j*pitch
        fill=COLORS['step_read'] if j==0 else COLORS['step_write'] if j==n-1 else COLORS['step_reuse'] if k=='shared' and j==2 else COLORS['step_compute']
        rect(x+88,ys,151,boxh,fill=fill,r=5)
        text(step,x+93,ys+1,141,boxh-2,size=16.9,min_font=15.0,align='center',ref=f'dk step{i}-{j}')

footers=[
 ['Thread-output / coefficient mapping','Thread-block size'],
 ['Coalesced FWD/dX time mapping','Thread-block size'],
 ['Shared operand reuse; serial dK','Thread-block size'],
 ['Warp-mapped FWD/dX; parallel dK','FWD/dX: warps per block\ndK: threads per block'],
]
for i,k in enumerate(keys):
    x=SIDE+i*CW
    rect(x,Y_DK,CW,Y_KEY-Y_DK,fill=COLORS[k][0],stroke='#BDCAD1')
    rect(x,Y_KEY,CW,H-Y_KEY,fill=COLORS[k][0],stroke='#BDCAD1')
    text(footers[i][0],x+12,Y_DK+5,CW-24,Y_KEY-Y_DK-10,size=17,min_font=14.8,align='center',ref=f'key{i}')
    text(footers[i][1],x+10,Y_KEY+5,CW-20,H-Y_KEY-10,size=17,min_font=14.7,align='center',ref=f'launch{i}')

# Unit test non-overlap for all actual text lines. Each line's rendered box must stay inside dedicated cell.
for ref,x,y,w,h,xx,yy,sw,leading,line in text_boxes:
    if not (xx >= x-0.15 and xx+sw <= x+w+0.15 and yy >= y-0.2 and yy+leading <= y+h+1.5):
        raise AssertionError((ref,line,(x,y,w,h),(xx,yy,sw,leading)))
# check no separate text lines from different assigned boxes intersect. Normal ink occupies <= leading; simplify using bbox
for a in range(len(text_boxes)):
    ar,ax,ay,aw,ah,axx,ayy,asw,al,at=text_boxes[a]
    a_box=(axx,ayy,axx+asw,ayy+al*.91)
    for b in range(a+1,len(text_boxes)):
        br,bx,by,bw,bh,bxx,byy,bsw,bl,bt=text_boxes[b]
        if ar==br: continue
        b_box=(bxx,byy,bxx+bsw,byy+bl*.91)
        if (min(a_box[2],b_box[2])-max(a_box[0],b_box[0])>1.1 and min(a_box[3],b_box[3])-max(a_box[1],b_box[1])>1.1):
            raise AssertionError(('TEXT OVERLAP',ar,br,at,bt,a_box,b_box))

c.showPage(); c.save()
doc=fitz.open(OUT)
page=doc[0]
pix=page.get_pixmap(matrix=fitz.Matrix(1.6,1.6),alpha=False)
pix.save(PREVIEW)
print('PDF:',OUT, os.path.getsize(OUT), 'bytes; preview:',PREVIEW, 'text lines:',len(text_boxes))
print('PDF text length:',len(page.get_text()), 'images count:',len(page.get_images()), 'page size:',page.rect)
