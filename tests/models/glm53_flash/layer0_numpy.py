# Independent reimplementation of the GLM-5.3-Flash leading-layer forward,
# written from src/models/glm53_flash/AGENTS.md and compared against the C++
# harness dumps (tests/models/glm53_flash/forward_layer0_test.cpp).
# Usage: layer0_numpy.py <model.gguf> <dump_dir>
import math
import os
import sys

import numpy as np
try:
    from gguf import GGUFReader
    from gguf.quants import dequantize
except ImportError:
    sys.exit(77)

H, HC, HD, HEADS, D_INNER = 4096, 4, 128, 64, 8192
HC_EPS = 1e-6
RMS_EPS = 1e-5
TOK = [9703, 11, 1879, 0, 101726, 98384]
LAYERS = int(os.environ.get('GUFO_GLM53_LAYERS', '3'))
KV, QK, IH, ID, KPOOL = 512, 256, 32, 128, 4


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def rms(x, w=None, eps=RMS_EPS):
    inv = 1.0 / np.sqrt(np.mean(x * x, dtype=np.float64) + eps)
    y = x * inv
    return (y * w) if w is not None else y


def lnorm(x, w, b, eps=1e-6):
    mu = x.mean()
    var = np.mean((x - mu) ** 2, dtype=np.float64)
    y = (x - mu) / np.sqrt(var + eps)
    return (y * w + b).astype(np.float32)


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

    def expert(self, name, e):
        """Dequantize one expert slab of a 3-D tensor without loading all."""
        key = f'{name}#{e}'
        if key not in self.cache:
            t = self.tensors[name]
            self.cache[key] = np.array(dequantize(t.data[e], t.tensor_type),
                                      dtype=np.float32)
        return self.cache[key]

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




def dsa_dump(dump, layer, t, st, model, p):
    check(dump, f'L{layer}_dsa_lat', t, np.asarray(st['lat'][t]))


def dsa_block(model, p, xin, st, t, sp_dump=None):
    """Sparse MLA (nope-only) with the kpool-select indexer."""
    qr = rms((model.get(p + 'attn_q_a.weight') @ xin).astype(np.float32),
             model.get(p + 'attn_q_a_norm.weight'))
    q = (model.get(p + 'attn_q_b.weight') @
         qr.astype(np.float64)).astype(np.float32).reshape(HEADS, QK)
    lat = rms((model.get(p + 'attn_kv_a_mqa.weight') @ xin).astype(np.float32),
              model.get(p + 'attn_kv_a_norm.weight'))

    iq = (model.get(p + 'indexer.attn_q_b.weight') @ qr).reshape(IH, ID)
    ik = model.get(p + 'indexer.attn_k.weight') @ xin
    ik = lnorm(ik.astype(np.float64), model.get(p + 'indexer.k_norm.weight'),
               model.get(p + 'indexer.k_norm.bias'))
    gate = model.get(p + 'indexer_compressor_gate.weight') @ xin

    n = len(st['lat'])
    st['lat'].append(lat)
    st['ik'].append(ik)
    st['gate'].append(gate)
    if (n + 1) % KPOOL == 0:
        ape = model.get(p + 'indexer_compressor_ape.weight')  # [slot, dim]
        b = n + 1 - KPOOL
        ks = np.stack(st['ik'][b:n + 1])          # [slot, dim]
        gs = np.stack(st['gate'][b:n + 1]) + ape  # [slot, dim]
        gs = gs - gs.max(axis=0, keepdims=True)
        ps = np.exp(gs)
        ps /= ps.sum(axis=0, keepdims=True)
        st['pooled'].append((ps * ks).sum(axis=0).astype(np.float32))

    w = (model.get(p + 'indexer.proj.weight') @ xin) / math.sqrt(IH * ID)
    npool = len(st['pooled'])
    n_top = min(npool, int(os.environ.get('GUFO_GLM53_TOPPOOL',
                        2048 // KPOOL)))
    vis = []
    if npool:
        pooled = np.stack(st['pooled'])  # [pool, dim]
        score = (np.maximum(pooled.astype(np.float64) @ iq.T.astype(
            np.float64), 0) * w[None, :].astype(np.float64)).sum(axis=1)
        if sp_dump is None:
            order = np.argsort(-score, kind='stable')[:n_top]
        else:
            check(sp_dump, 'DSASP_w0', t, w.astype(np.float32))
            check(sp_dump, 'DSASP_ikrow', t, ik.astype(np.float32))
            check(sp_dump, 'DSASP_grow', t, gate.astype(np.float32))
            check(sp_dump, 'DSASP_iq0', t, iq.reshape(-1).astype(np.float32))
            check(sp_dump, 'DSASP_pool0', t, pooled[0].astype(np.float32))
            check(sp_dump, 'DSASP_pool_score', t, score.astype(np.float32))
            cs = np.fromfile(
                os.path.join(sp_dump, f'DSASP_pool_score.{t}.f32'),
                dtype=np.float32)
            order = np.argsort(-cs, kind='stable')[:n_top]
        vis = [int(o) * KPOOL + k for o in order[:n_top]
               for k in range(KPOOL)]
    for k in range((n + 1) % KPOOL):
        vis.append(n - k)

    wk = model.get(p + 'attn_k_b.weight')   # [head, kv, qk]
    wv = model.get(p + 'attn_v_b.weight')   # [head, qk, kv]
    latm = np.stack(st['lat'])              # [tok, kv]
    qabs = (q[:, None, :].astype(np.float64) *
            wk.astype(np.float64)).sum(axis=2)  # [head, kv]
    sel = latm[vis].astype(np.float64)         # [vis, kv]
    s = (qabs @ sel.T) / math.sqrt(QK)
    s -= s.max(axis=1, keepdims=True)
    a = np.exp(s)
    a /= a.sum(axis=1, keepdims=True)
    ctx = (a @ sel).astype(np.float32)         # [head, kv]
    out = np.einsum('hc,hjc->hj', ctx.astype(np.float64),
                    wv.astype(np.float64)).astype(np.float32)
    return (model.get(p + 'attn_output.weight') @
            out.reshape(-1).astype(np.float64)).astype(np.float32)


def moe_ffn(model, p, fin):
    logits = model.get(p + 'ffn_gate_inp.weight') @ fin.astype(np.float64)
    eb = model.get(p + 'exp_probs_b.bias')
    probs = sigmoid(logits.astype(np.float32))
    sel = probs + eb
    idx = np.argsort(-sel)[:8]
    wsum = max(float(probs[idx].sum()), 6.103515625e-5)
    out = np.zeros(4096, dtype=np.float32)
    for e in idx:
        g = model.expert(p + 'ffn_gate_exps.weight', e) @ fin
        u = model.expert(p + 'ffn_up_exps.weight', e) @ fin
        g = np.minimum(g, 10.0)
        u = np.clip(u, -10.0, 10.0)
        act = g * sigmoid(g) * u
        out += probs[e] / wsum * 2.5 * (
            model.expert(p + 'ffn_down_exps.weight', e) @ act).astype(np.float32)
    g = np.minimum(model.get(p + 'ffn_gate_shexp.weight') @ fin, 10.0)
    u = np.clip(model.get(p + 'ffn_up_shexp.weight') @ fin, -10.0, 10.0)
    act = g * sigmoid(g) * u
    return (out + (model.get(p + 'ffn_down_shexp.weight') @
                   act).astype(np.float32))


def main():
    if not len(sys.argv) > 2 or not sys.argv[1]:
        sys.exit(77)
    model = Model(sys.argv[1])
    dump = sys.argv[2]

    if len(sys.argv) > 3 and sys.argv[3] == '--dsasp':
        st = dict(lat=[], ik=[], gate=[], pooled=[])
        idx = np.arange(4096, dtype=np.float32)
        for t in range(48):
            xn = (np.float32(0.1) * np.sin(
                np.float32(0.37) * idx +
                np.float32(0.11) * np.float32(t))).astype(np.float32)
            blk = dsa_block(model, 'blk.3.', xn, st, t, dump)
            check(dump, 'DSASP_dsa_lat', t, np.asarray(st['lat'][t]))
            check(dump, 'DSASP_attn_block', t, blk)
        print('numpy dsasp complete')
        return
    T = len(TOK)

    embd = model.get('token_embd.weight')  # [vocab, H]
    x = np.repeat(embd[np.array(TOK)][:, None, :], HC, axis=1)

    conv_state = np.zeros((LAYERS, 3, 3, D_INNER), dtype=np.float32)
    ssm_state = np.zeros((LAYERS, HEADS, HD, HD), dtype=np.float32)
    dsa_state = [dict(lat=[], ik=[], gate=[], pooled=[]) for _ in range(45)]

    for layer in range(LAYERS):
        p = f'blk.{layer}.'
        wa = dict(hc_fn=model.get(p + 'hc_attn_fn.weight'),
                  hc_scale=model.get(p + 'hc_attn_scale.weight'),
                  hc_base=model.get(p + 'hc_attn_base.weight'))
        wf = dict(hc_fn=model.get(p + 'hc_ffn_fn.weight'),
                  hc_scale=model.get(p + 'hc_ffn_scale.weight'),
                  hc_base=model.get(p + 'hc_ffn_base.weight'))
        dsa = (layer % 4) == 3
        moe = layer >= 3
        attn_norm = model.get(p + 'attn_norm.weight')
        ffn_norm = model.get(p + 'ffn_norm.weight')
        if not dsa:
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

            if dsa:
                block = dsa_block(model, p, xin, dsa_state[layer], t)
                dsa_dump(dump, layer, t, dsa_state[layer], model, p)
            else:
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
            if moe:
                sel = np.argsort(-(sigmoid(
                    (model.get(p + 'ffn_gate_inp.weight') @
                     fin.astype(np.float64)).astype(np.float32)
                    ) + model.get(p + 'exp_probs_b.bias')))[:8]
                check(dump, f'L{layer}_moe_sel', t, sel.astype(np.float32))
                ffn_out = moe_ffn(model, p, fin)
            else:
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
    if os.environ.get('GUFO_GLM53_LOGITS'):
        w = model.get('output.weight')
        for t in range(T - 1):
            lg = (w @ h[t].astype(np.float64)).astype(np.float32)
            top = np.argsort(-lg)[:5]
            ref = np.fromfile(os.path.join(dump, f'logits_top5.{t}.f32'),
                              dtype=np.float32)
            cpp_ids = ref[1::2].view(np.int32)
            cpp_log = ref[0::2]
            agree = int(top[0]) == int(cpp_ids[0])
            vrel = np.abs(lg[top] - cpp_log).max() / max(
                np.abs(cpp_log).max(), 1.0)
            print(f'logits t={t} top1 {"HIT" if agree else "MISS"} '
                  f'top5_logit_max_rel={vrel:.3e}')
            if not agree or vrel >= 5e-3:
                sys.exit(1)
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
