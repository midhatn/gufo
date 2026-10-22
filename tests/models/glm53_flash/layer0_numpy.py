# Independent reimplementation of the GLM-5.3-Flash leading-layer forward,
# written from src/models/glm53_flash/AGENTS.md and compared against the C++
# harness dumps (tests/models/glm53_flash/forward_layer0_test.cpp).
# Usage: layer0_numpy.py <model.gguf> <dump_dir>
import math
import os
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

H, HC, HD, HEADS, D_INNER = 4096, 4, 128, 64, 8192
HC_EPS = 1e-6
RMS_EPS = 1e-5
TOK = [9703, 11, 1879, 0, 101726, 98384]
LAYERS = 3


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def rms(x, w=None, eps=RMS_EPS):
    inv = 1.0 / np.sqrt(np.mean(x * x, dtype=np.float64) + eps)
    y = x * inv
    return (y * w) if w is not None else y


def l2(x, eps=1e-6):
    return x / np.sqrt(np.sum(x * x, dtype=np.float64) + eps)


class Model:
    def __init__(self, path):
        self.reader = GGUFReader(path)
        self.tensors = {t.name: t for t in self.reader.tensors}
        self.cache = {}

    def get(self, name):
        if name not in self.cache:
            t = self.tensors[name]
            if int(t.tensor_type) == 0:
                self.cache[name] = np.array(t.data, dtype=np.float32)
            else:
                self.cache[name] = np.array(dequantize(t.data, t.tensor_type),
                                           dtype=np.float32)
        return self.cache[name]

    def w(self, name):  # matrix [out, in] for GGUF dims (in, out)
        return self.get(name)

    def proj(self, name, x):
        return self.get(name) @ x


def hc_mixes(m, token):
    xn = rms(np.asarray(m, dtype=np.float64).ravel()).astype(np.float32)
    mixes = token['hc_fn'] @ xn
    scale, base = token['hc_scale'], token['hc_base']
    pre = sigmoid(mixes[0:HC] * scale[0] + base[0:HC]) + HC_EPS
    post = sigmoid(mixes[HC:2 * HC] * scale[1] + base[HC:2 * HC]) * 2.0
    comb = (mixes[2 * HC:2 * HC + HC * HC] * scale[2] +
            base[2 * HC:2 * HC + HC * HC]).reshape(HC, HC, order="F")  # [dst, src]
    comb = comb - comb.max(axis=0, keepdims=True)
    comb = np.exp(comb)
    comb /= (comb.sum(axis=0, keepdims=True) + HC_EPS)
    comb /= (comb.sum(axis=1, keepdims=True) + HC_EPS)  # over src
    for _ in range(19):
        comb /= (comb.sum(axis=0, keepdims=True) + HC_EPS)  # over dst
        comb /= (comb.sum(axis=1, keepdims=True) + HC_EPS)  # over src
    return xn, pre.astype(np.float32), post.astype(np.float32), comb.astype(
        np.float32)


def hc_post(x_stream, block, post, comb):
    out = block[None, :] * post[:, None]
    out += (comb @ x_stream).astype(np.float32)
    return out


def main():
    model = Model(sys.argv[1])
    dump = sys.argv[2]
    T = len(TOK)

    embd = model.get('token_embd.weight')  # [vocab, H]
    x = np.repeat(embd[np.array(TOK)][:, None, :], HC, axis=1)

    conv_state = np.zeros((LAYERS, 3, 3, D_INNER), dtype=np.float32)
    ssm_state = np.zeros((LAYERS, HEADS, HD, HD), dtype=np.float32)

    for layer in range(LAYERS):
        p = f'blk.{layer}.'
        wa = dict(hc_fn=model.get(p + 'hc_attn_fn.weight'),
                  hc_scale=model.get(p + 'hc_attn_scale.weight'),
                  hc_base=model.get(p + 'hc_attn_base.weight'))
        wf = dict(hc_fn=model.get(p + 'hc_ffn_fn.weight'),
                  hc_scale=model.get(p + 'hc_ffn_scale.weight'),
                  hc_base=model.get(p + 'hc_ffn_base.weight'))
        attn_norm = model.get(p + 'attn_norm.weight')
        ffn_norm = model.get(p + 'ffn_norm.weight')
        cq = model.get(p + 'ssm_conv1d_q.weight').reshape(D_INNER, 4)
        ck = model.get(p + 'ssm_conv1d_k.weight').reshape(D_INNER, 4)
        cv = model.get(p + 'ssm_conv1d_v.weight').reshape(D_INNER, 4)
        a_log = model.get(p + 'ssm_a')
        dt = model.get(p + 'ssm_dt.bias')

        for t in range(T):
            cur = x[t].astype(np.float32)
            _, pre, post, comb = hc_mixes(cur, wa)
            xin = (cur * pre[:, None]).sum(axis=0)
            xin = rms(xin, attn_norm)
            check(dump, f'L{layer}_attn_in', t, xin)

            q = proj(model, p + 'attn_q.weight', xin)
            k = proj(model, p + 'attn_k.weight', xin)
            v = proj(model, p + 'attn_v.weight', xin)
            q = conv_silu(conv_state[layer, 0], cq, q)
            k = conv_silu(conv_state[layer, 1], ck, k)
            v = conv_silu(conv_state[layer, 2], cv, v)

            g = proj(model, p + 'ssm_f_b.weight',
                     proj(model, p + 'ssm_f_a.weight', xin)) + dt
            beta = sigmoid(proj(model, p + 'ssm_beta.weight', xin))
            g2 = proj(model, p + 'ssm_g_b.weight',
                      proj(model, p + 'ssm_g_a.weight', xin))

            o = np.zeros(D_INNER, dtype=np.float32)
            for h in range(HEADS):
                s = ssm_state[layer, h]
                qh = l2(q[h * HD:(h + 1) * HD].astype(np.float64)).astype(
                    np.float32)
                kh = l2(k[h * HD:(h + 1) * HD].astype(np.float64)).astype(
                    np.float32)
                decay = np.exp(-5.0 * sigmoid(-(g[h * HD:(h + 1) * HD] *
                                              a_log[h]))).astype(np.float32)
                s *= decay[None, :]
                u = s @ kh
                delta = (v[h * HD:(h + 1) * HD] - u) * beta[h]
                s += delta[:, None] * kh[None, :]
                oh = (s @ qh).astype(np.float32) / math.sqrt(HD)
                oh = rms(oh, model.get(p + 'ssm_norm.weight'))
                o[h * HD:(h + 1) * HD] = oh * sigmoid(
                    g2[h * HD:(h + 1) * HD])
            block = proj(model, p + 'attn_output.weight', o)
            check(dump, f'L{layer}_attn_block', t, block)
            x[t] = hc_post(cur, block, post, comb)

            _, pre, post, comb = hc_mixes(x[t].astype(np.float32), wf)
            fin = (x[t].astype(np.float32) * pre[:, None]).sum(axis=0)
            fin = rms(fin, ffn_norm)
            gate = proj(model, p + 'ffn_gate.weight', fin)
            up = proj(model, p + 'ffn_up.weight', fin)
            gate = np.minimum(gate, 10.0)
            up = np.clip(up, -10.0, 10.0)
            ffn_out = proj(
                model, p + 'ffn_down.weight',
                (gate * sigmoid(gate) * up).astype(np.float32))
            check(dump, f'L{layer}_ffn_block', t, ffn_out)
            x[t] = hc_post(x[t].astype(np.float32), ffn_out, post, comb)
            x[t].astype(np.float32).tofile(
                os.path.join(dump, f'L{layer}_ffn_stream.{t}.py.f32'))

    h = x.mean(axis=1)
    h = np.stack([rms(h[t], model.get('output_norm.weight')) for t in range(T)])
    for t in range(T):
        check(dump, 'h_final', t, h[t])
    print('numpy forward complete')


_cache = {}


def proj(model, name, x):
    if name not in _cache:
        _cache[name] = model.get(name).astype(np.float64)
    return (_cache[name] @ x.astype(np.float64)).astype(np.float32)


def conv_silu(hist, w, x):
    y = hist[0] * w[:, 0] + hist[1] * w[:, 1] + hist[2] * w[:, 2] + x * w[:, 3]
    hist[0] = hist[1]
    hist[1] = hist[2]
    hist[2] = x
    return (y * sigmoid(y)).astype(np.float32)


def check(dump, stage, t, arr):
    path = os.path.join(dump, f'{stage}.{t}.f32')
    if not os.path.exists(path):
        return
    ref = np.fromfile(path, dtype=np.float32)
    a = np.asarray(arr, dtype=np.float32)
    scale = max(float(np.abs(ref).max()), 1.0)
    rel = np.abs(a - ref).max() / scale
    status = 'OK' if rel < 5e-3 else 'MISMATCH'
    print(f'{stage} t={t} max_rel={rel:.3e} {status}')
    if rel >= 5e-3:
        sys.exit(1)


if __name__ == '__main__':
    main()
