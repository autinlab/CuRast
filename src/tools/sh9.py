import sys, numpy as np

def read_hdr(path):
    f = open(path, 'rb')
    # --- header ---
    if not f.readline().startswith(b'#?'):
        raise RuntimeError('not a radiance file')
    while True:
        line = f.readline()
        if line in (b'\n', b'\r\n', b''):
            break
    res = f.readline().decode('ascii').strip().split()
    # expect "-Y H +X W"
    if res[0] != '-Y' or res[2] != '+X':
        raise RuntimeError('unsupported resolution line: %r' % res)
    H, W = int(res[1]), int(res[3])
    data = np.frombuffer(f.read(), dtype=np.uint8)
    f.close()

    rgbe = np.zeros((H, W, 4), dtype=np.uint8)
    pos = 0
    for y in range(H):
        if (W >= 8 and W < 32768 and data[pos] == 2 and data[pos+1] == 2
                and ((int(data[pos+2]) << 8) | int(data[pos+3])) == W):
            pos += 4
            for c in range(4):
                x = 0
                while x < W:
                    cnt = int(data[pos]); pos += 1
                    if cnt > 128:                      # run
                        n = cnt - 128
                        rgbe[y, x:x+n, c] = data[pos]; pos += 1
                        x += n
                    else:                              # literal
                        rgbe[y, x:x+cnt, c] = data[pos:pos+cnt]; pos += cnt
                        x += cnt
        else:                                          # flat RGBE scanline
            rgbe[y] = data[pos:pos+W*4].reshape(W, 4); pos += W*4

    e = rgbe[:, :, 3].astype(np.int32)
    scale = np.where(e == 0, 0.0, np.ldexp(1.0, e - 136))   # 2^(e-128) / 256
    img = (rgbe[:, :, :3].astype(np.float32) + 0.5) * scale[:, :, None]
    return img.astype(np.float32)

img = read_hdr(sys.argv[1])
H, W, _ = img.shape
print('# loaded %dx%d  min=%.4f max=%.2f mean=%.4f' %
      (W, H, img.min(), img.max(), img.mean()), file=sys.stderr)

# equirect -> direction, world up = +Z
v = (np.arange(H) + 0.5) / H
u = (np.arange(W) + 0.5) / W
theta = v * np.pi                    # 0 at top row -> +Z
phi = u * 2.0 * np.pi - np.pi
st, ct = np.sin(theta), np.cos(theta)
x = st[:, None] * np.cos(phi)[None, :]
y = st[:, None] * np.sin(phi)[None, :]
z = np.broadcast_to(ct[:, None], (H, W))

dw = (2.0 * np.pi / W) * (np.pi / H) * st[:, None]
dw = np.broadcast_to(dw, (H, W))

Y = [0.282095 * np.ones_like(x),
     0.488603 * y, 0.488603 * z, 0.488603 * x,
     1.092548 * x * y, 1.092548 * y * z,
     0.315392 * (3.0 * z * z - 1.0),
     1.092548 * x * z, 0.546274 * (x * x - y * y)]

L = [np.einsum('ij,ijc->c', Yi * dw, img) for Yi in Y]
A = [3.141593, 2.094395, 2.094395, 2.094395,
     0.785398, 0.785398, 0.785398, 0.785398, 0.785398]

names = ['Y00', 'Y1-1', 'Y10', 'Y11', 'Y2-2', 'Y2-1', 'Y20', 'Y21', 'Y22']
print('// SH9 irradiance coefficients (A_l * L_lm), world up = +Z')
print('__device__ const float3 c_envSH[9] = {')
for n, a, l in zip(names, A, L):
    c = a * l
    print('\t{%12.6ff, %12.6ff, %12.6ff},  // %s' % (c[0], c[1], c[2], n))
print('};')

# dominant light: intensity-weighted mean direction of the brightest 0.5% of energy
lum = img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
wgt = lum * dw
thr = np.quantile(lum, 0.995)
m = lum >= thr
wm = wgt * m
d = np.array([(x * wm).sum(), (y * wm).sum(), (z * wm).sum()])
d /= np.linalg.norm(d)
col = np.einsum('ij,ijc->c', wm, img) / wm.sum()
print('// dominant light dir=(%.4f, %.4f, %.4f)  colour=(%.3f, %.3f, %.3f)  frac=%.3f'
      % (d[0], d[1], d[2], col[0], col[1], col[2], wm.sum() / wgt.sum()))
print('__device__ const float3 c_envKeyDir = {%.6ff, %.6ff, %.6ff};' % (d[0], d[1], d[2]))
mx = max(col.max(), 1e-6)
print('__device__ const float3 c_envKeyColor = {%.6ff, %.6ff, %.6ff};'
      % (col[0] / mx, col[1] / mx, col[2] / mx))
