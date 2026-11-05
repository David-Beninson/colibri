/* Qwen3.6 checkpoint MTP head, included by qwen36.c.
 * Pair the backbone's output h_p (after its final norm, which drafts better than the
 * residual before it on the released checkpoint) with embed(token_{p+1}):
 * fc(cat(pre_fc_norm_embedding(embed), pre_fc_norm_hidden(h))), one gated
 * full-attention/MoE decoder layer, mtp.norm and the shared lm_head.
 * tools/qwen36_mtp_ref.py checks this against Qwen's transformers decoder.
 * The head owns its KV cache; verification alone commits backbone tokens. */

typedef struct {
    int on, E, K, I, ish;
    Layer L;                         /* q, k, v, o, qn, kn, in_ln (attn_norm), post_ln: the engine's (1 + w) norms */
    QW eh;                           /* [D x 2D] */
    float *enorm, *hnorm, *head_norm;
    float *router;                   /* [E][D] */
    float *sh_gate;                  /* [D] */
    QW sh_g, sh_u, sh_d;
    QW *eg, *eu, *ed;                /* [E] */
    QW dvoc; int nvoc;               /* the shared lm_head rows selected by vid */
    DraftVocab *voc;                 /* Q36_MTP_VOCAB: adaptive draft vocabulary (draft_vocab.h); NULL = full or vid */
    uint64_t voc_gen; int voc_reuse; /* one more at every rebuild of voc's list; the prompt tokens serve kept */
    float last_p;                    /* the newest draft's probability under the head */
    int pick_id, pick_ok;            /* q36_mg_row / q36_mc_row compact pick */
    int *vid;                        /* Q36_MTP_VOCAB_IDS: row j of dvoc scores id vid[j] (NULL: id j) */
    void *Kc, *Vc; int stride;       /* the head's KV rows [KV][stride][hd], q36_kv_es() bytes an element */
    void **Kp, **Vp;                 /* the model's per-layer pointers and the head's after them */
    /* where the head is: rows up to `len` read; `pend` the model's row at len (before the
     * final norm), waiting for the token at len + 1; `done`: the row at done_pos already
     * run by a draft with token done_tok */
    int len, has_pend, done_pos, done_tok;
    float *pend;
    float *last; int last_base, last_n, last_cap;   /* the newest forward's rows (a rewind lands in them) */
    unsigned long long rows, drafts, kv_rows;
    double ms, kv_ms;
} Q36Mtp;
static Q36Mtp g_q36_mtp;

/* Keep the container's expert quantization, without a second quantization.
 * QW's packed format is gs64 planar; other grouped layouts widen exactly. */
static void q36_mtp_expert_qw(QW *w, const int8_t *q, const float *sc,
                              int I, int O, int gs, int packed) {
    w->I = I; w->O = O;
    int ng = gs ? (I + gs - 1) / gs : 1;
    if (packed && gs == 64 && I % 64 == 0 && dense_idot_on()) {
        w->q4 = malloc((size_t)I * O / 2);
        w->sg = falloc((int64_t)O * ng); w->ng = ng;
        if (!w->q4) { fprintf(stderr, "OOM MTP expert\n"); exit(1); }
        for (int o = 0; o < O; o++) for (int j = 0; j < I; j += 64)
            for (int k = 0; k < 32; k++)
                w->q4[(size_t)o * I / 2 + j / 2 + k] =
                    (q[(size_t)o * I + j + k] + 8) | ((q[(size_t)o * I + j + k + 32] + 8) << 4);
        memcpy(w->sg, sc, (size_t)O * ng * sizeof(float));
    } else if (!gs) {
        w->q = q36_walloc((size_t)I * O); w->sc = falloc(O);
        if (!w->q) { fprintf(stderr, "OOM MTP expert\n"); exit(1); }
        memcpy(w->q, q, (size_t)I * O); memcpy(w->sc, sc, (size_t)O * sizeof(float));
    } else {
        float *f = falloc((int64_t)I * O);
        for (int o = 0; o < O; o++) for (int j = 0; j < I; j++)
            f[(size_t)o * I + j] = q[(size_t)o * I + j] * sc[o * ng + j / gs];
        w->w = f;
    }
}
static int q36_mtp_load(Model *m) {
    Cfg *c = &m->c; Q36Mtp *p = &g_q36_mtp;
    int D = c->hidden, E = c->n_experts, I = c->inter, Ish = c->shared_inter;
    if (!st_find(&m->S, "mtp.fc.weight")) return 0;
    if (E <= 0 || E > 1024 || c->topk > 16 || Ish <= 0) {
        fprintf(stderr, "[mtp] head requires a MoE model with shared expert\n"); return 0;
    }
    p->E = E; p->K = c->topk; p->I = I; p->ish = Ish;
#define MN(field, name, n) p->field = load_norm_n(m, "mtp." name ".weight", n)
    MN(enorm, "pre_fc_norm_embedding", D); MN(hnorm, "pre_fc_norm_hidden", D);
    MN(head_norm, "norm", D); MN(L.in_ln, "layers.0.input_layernorm", D);
    MN(L.post_ln, "layers.0.post_attention_layernorm", D);
    MN(L.qn, "layers.0.self_attn.q_norm", c->head_dim);
    MN(L.kn, "layers.0.self_attn.k_norm", c->k_head_dim);
#undef MN
#define MW(field, name, ni, no) load_tq(m, "mtp." name ".weight", ni, no, 1, "mtp", &p->field)
    MW(eh, "fc", 2 * D, D);
    MW(L.q, "layers.0.self_attn.q_proj", D, c->q_heads * c->q_head_dim);
    MW(L.k, "layers.0.self_attn.k_proj", D, c->kv_heads * c->k_head_dim);
    MW(L.v, "layers.0.self_attn.v_proj", D, c->kv_heads * c->v_head_dim);
    MW(L.o, "layers.0.self_attn.o_proj", c->q_heads * c->head_dim, D);
    MW(sh_g, "layers.0.mlp.shared_expert.gate_proj", D, Ish);
    MW(sh_u, "layers.0.mlp.shared_expert.up_proj", D, Ish);
    MW(sh_d, "layers.0.mlp.shared_expert.down_proj", Ish, D);
#undef MW
    p->router = load_t_n(m, "mtp.layers.0.mlp.gate.weight", (int64_t)E * D);
    p->sh_gate = load_t_n(m, "mtp.layers.0.mlp.shared_expert_gate.weight", D);
    p->eg = calloc(E, sizeof(QW)); p->eu = calloc(E, sizeof(QW)); p->ed = calloc(E, sizeof(QW));
    if (!p->eg || !p->eu || !p->ed) { fprintf(stderr, "OOM MTP experts\n"); exit(1); }
    for (int e = 0; e < E; e++) {
        char name[128], sn[128];
        snprintf(name, sizeof name, "mtp.layers.0.mlp.experts.%d.merged_weight", e);
        snprintf(sn, sizeof sn, "mtp.layers.0.mlp.experts.%d.qs", e);
        st_tensor *t = st_find(&m->S, name);
        int64_t n = (int64_t)I * D;
        if (!t || (t->nbytes != 3*n && t->nbytes != 3*n/2 && t->nbytes != 2*n)) {
            fprintf(stderr, "invalid MTP expert %s\n", name); exit(1);
        }
        int packed = t->nbytes == 3*n/2, mixed = t->nbytes == 2*n;
        int8_t *q = malloc((size_t)3*n); uint8_t *raw = malloc((size_t)t->nbytes);
        if (!q || !raw) { fprintf(stderr, "OOM MTP expert\n"); exit(1); }
        st_read_raw(&m->S, name, raw, 1);
        if (packed) unpack_int4_to_int8(q, raw, 3*n);
        else if (mixed) { unpack_int4_to_int8(q, raw, 2*n); memcpy(q + 2*n, raw + n, (size_t)n); }
        else memcpy(q, raw, (size_t)3*n);
        int64_t ns = scale_count_gu(c), ds = scale_count_d(c);
        float *sc = load_t_n(m, sn, 2*ns + ds);
        q36_mtp_expert_qw(&p->eg[e], q, sc, D, I, g_expert_gs, packed || mixed);
        q36_mtp_expert_qw(&p->eu[e], q+n, sc+ns, D, I, g_expert_gs, packed || mixed);
        q36_mtp_expert_qw(&p->ed[e], q+2*n, sc+2*ns, I, D, mixed ? g_expert_down_gs : g_expert_gs, packed);
        free(q); free(raw); free(sc);
    }
    const char *file = getenv("Q36_MTP_VOCAB_IDS");
    if (file && *file) {
        FILE *f = fopen(file, "r");
        if (!f) { fprintf(stderr, "cannot read Q36_MTP_VOCAB_IDS %s\n", file); exit(1); }
        int *ids = malloc((size_t)c->vocab * sizeof(int));
        unsigned char *seen = calloc(c->vocab, 1); int id, n = 0;
        if (!ids || !seen) { fprintf(stderr, "OOM MTP vocabulary\n"); exit(1); }
        while (fscanf(f, "%d", &id) == 1) {
            if (id < 0 || id >= c->vocab) { fprintf(stderr, "invalid MTP vocabulary id %d\n", id); exit(1); }
            if (!seen[id]) { ids[n++] = id; seen[id] = 1; }
        }
        if (!feof(f) || !n) { fprintf(stderr, "invalid/empty MTP vocabulary\n"); exit(1); }
        fclose(f); free(seen);
        /* Read the shared checkpoint matrix, not a separately trained output head. */
        char resolved[QW_DENSE_NAME_MAX];
        const char *lm = dense_resolve(m, "lm_head.weight", resolved, sizeof resolved);
        st_tensor *weight = st_find(&m->S, lm);
        if (!weight || weight->numel != (int64_t)c->vocab * D || weight->dtype > 2) {
            fprintf(stderr, "MTP vocabulary needs a float shared lm_head\n"); exit(1);
        }
        float *rows = falloc((int64_t)n * D);
        for (int j = 0; j < n; j++)
            st_read_slice_f32(&m->S, lm, (int64_t)ids[j]*D, D, rows + (int64_t)j*D, 0);
        p->dvoc.I = D; p->dvoc.O = n; p->dvoc.w = rows;
        if (dense_i8_on()) { qw_quantize(rows, D, n, "lm_head", &p->dvoc); free(rows); p->dvoc.w = NULL; }
        p->nvoc = n; p->vid = ids;
    } else {
        /* Q36_MTP_VOCAB: the head scores the most used tokens only (draft_vocab.h, as Qwen3.8's).
         * Unset, a list of DRAFT_VOCAB_DEFAULT_TOP where the head drafts on the CUDA chain (measured
         * there), the whole vocabulary elsewhere; 0 is always the whole vocabulary. */
        const char *ve = getenv("Q36_MTP_VOCAB"), *dmp = getenv("Q36_MTP_DUMP");
        int top = 0;
        if (ve && *ve) {
            char *end = NULL; long parsed = strtol(ve, &end, 10);
            if (end == ve || *end || parsed < 0 || parsed > (long)c->vocab) {
                fprintf(stderr, "Q36_MTP_VOCAB must be an integer in 0..%d\n", c->vocab); exit(1);
            }
            top = (int)parsed;
        }
        int can = !(dmp && *dmp) && 0 < top && top < c->vocab && (m->lm_head.q || m->lm_head.w);
        if (can) {
            const char *snap = getenv("SNAP"), *vp = getenv("COLI_VOCAB_USAGE"), *up = getenv("COLI_USAGE");
            if (!snap || !*snap) snap = ".";
            size_t cl = (vp && *vp ? strlen(vp) : strlen(snap) + 13) + 1;
            size_t ul = (up && *up ? strlen(up) : strlen(snap) + 13) + 1;
            char *counts = malloc(cl), *usage = malloc(ul);
            if (!counts || !usage) { fprintf(stderr, "OOM MTP vocabulary\n"); exit(1); }
            if (vp && *vp) memcpy(counts, vp, cl); else snprintf(counts, cl, "%s/.coli_vocab", snap);
            if (up && *up) memcpy(usage, up, ul); else snprintf(usage, ul, "%s/.coli_usage", snap);
            int same = coli_own_same_path(counts, usage);
            free(usage);
            if (same) {
                fprintf(stderr, "[mtp] draft vocabulary off, the head scores the whole vocabulary: "
                                "COLI_VOCAB_USAGE and COLI_USAGE are the same file, %s. Set COLI_VOCAB_USAGE "
                                "to another file.\n", counts);
                free(counts);
            } else {
                p->voc = draft_vocab_new(c->vocab, top);
                if (!p->voc) { fprintf(stderr, "OOM MTP vocabulary\n"); exit(1); }
                fprintf(stderr, "[mtp] draft vocabulary: top %d + %d/%d rule + conversation, ",
                        p->voc->top, DRAFT_VOCAB_NEAR_NUM, DRAFT_VOCAB_NEAR_DEN);
                double tokens; int distinct;
                switch (draft_vocab_load(p->voc, counts)) {
                case DRAFT_VOCAB_LOADED:
                    draft_vocab_totals(p->voc, &tokens, &distinct);
                    fprintf(stderr, "counts from %s (%.0f tokens, %d distinct)\n", p->voc->path, tokens, distinct);
                    break;
                case DRAFT_VOCAB_NO_FILE:
                    fprintf(stderr, "no counts file at %s yet\n", p->voc->path);
                    break;
                case DRAFT_VOCAB_NOT_COUNTS:
                    fprintf(stderr, "%s cannot be read as a counts file\n", p->voc->path);
                    break;
                case DRAFT_VOCAB_OTHER_SIZE:
                    fprintf(stderr, "%s has counts for another vocabulary size\n", p->voc->path);
                    break;
                case DRAFT_VOCAB_DAMAGED:
                    draft_vocab_totals(p->voc, &tokens, &distinct);
                    fprintf(stderr, "counts from %s up to a damaged record, after %d records (%.0f tokens, %d distinct)\n",
                            p->voc->path, p->voc->records, tokens, distinct);
                    break;
                }
            }
        }
    }
    p->pend = falloc(D); p->on = 1;
    fprintf(stderr, "[mtp] Qwen3.6 checkpoint head: %d experts\n", E);
    return 1;
}

/* the head's KV rows at the model's row stride (attention() indexes K[layer] by max_t),
 * and the pointer tables that put them after the model's layers */
static int q36_mtp_kv(Model *m) {
    Cfg *c = &m->c; Q36Mtp *p = &g_q36_mtp;
    if (p->stride != m->max_t || !p->Kc) {
        size_t n = (size_t)c->kv_heads * m->max_t * c->k_head_dim, es = q36_kv_es();
        void *kc = calloc(n, es), *vc = calloc(n, es);
        if (!kc || !vc) { free(kc); free(vc); p->on = 0; return 0; }
        if (p->Kc && p->Vc) {
            int keep = p->stride < m->max_t ? p->stride : m->max_t;
            for (int h = 0; h < c->kv_heads; h++) {
                memcpy(q36_kv_ptr(kc, (int64_t)h*m->max_t*c->k_head_dim), q36_kv_ptr(p->Kc, (int64_t)h*p->stride*c->k_head_dim),
                       (size_t)keep*c->k_head_dim*es);
                memcpy(q36_kv_ptr(vc, (int64_t)h*m->max_t*c->k_head_dim), q36_kv_ptr(p->Vc, (int64_t)h*p->stride*c->k_head_dim),
                       (size_t)keep*c->k_head_dim*es);
            }
        }
        free(p->Kc); free(p->Vc); p->Kc = kc; p->Vc = vc; p->stride = m->max_t;
    }
    if (!p->Kp) {
        p->Kp = malloc(sizeof(void *) * (size_t)(c->n_layers + 1));
        p->Vp = malloc(sizeof(void *) * (size_t)(c->n_layers + 1));
        if (!p->Kp || !p->Vp) { p->on = 0; return 0; }
    }
    for (int i = 0; i < c->n_layers; i++) { p->Kp[i] = m->K[i]; p->Vp[i] = m->V[i]; }
    p->Kp[c->n_layers] = p->Kc; p->Vp[c->n_layers] = p->Vc;
    return 1;
}

/* one row's routing from its router logits l[E] (softmax in place): top-K by the CPU's
 * rule, renormalised */
static void q36_mtp_route(float *l, int E, int K, int *idx, float *val) {
    softmax_row(l, E);
    for (int k = 0; k < K; k++) {
        int best = -1; float bv = -1e30f;
        for (int e = 0; e < E; e++) {
            int taken = 0; for (int j = 0; j < k; j++) taken |= idx[j] == e;
            if (!taken && l[e] > bv) { bv = l[e]; best = e; }
        }
        idx[k] = best < 0 ? k : best; val[k] = best < 0 ? 0.f : bv;
    }
    float sm = 0; for (int k = 0; k < K; k++) sm += val[k];
    if (sm > 0) for (int k = 0; k < K; k++) val[k] /= sm;
}
/* routed experts (softmax top-K, renormalised) + the sigmoid-gated shared expert;
 * a prompt's rows grouped per expert, so its catch-up is a few batched matmuls */
static void q36_mtp_moe(const float *x, int R, int D, float *out) {
    Q36Mtp *p = &g_q36_mtp; int E = p->E, K = p->K, I = p->I, Ish = p->ish;
    int *idx = malloc(sizeof(int) * (size_t)R * K); float *val = falloc((int64_t)R * K), *lg = falloc((int64_t)R * E);
    /* the router's logits: every (row, expert) dot apart */
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < R * E; j++) {
        const float *xr = x + (int64_t)(j / E) * D, *w = p->router + (int64_t)(j % E) * D;
        float a = 0; for (int i = 0; i < D; i++) a += xr[i] * w[i];
        lg[j] = a;
    }
    #pragma omp parallel for schedule(static) if (R > 1)
    for (int r = 0; r < R; r++) q36_mtp_route(lg + (int64_t)r * E, E, K, idx + r * K, val + r * K);
    memset(out, 0, sizeof(float) * (size_t)R * D);
    {   /* the shared expert, every row */
        float *g = falloc((int64_t)R * Ish), *u = falloc((int64_t)R * Ish), *y = falloc((int64_t)R * D);
        matmul_d(g, x, &p->sh_g, R, D, Ish); matmul_d(u, x, &p->sh_u, R, D, Ish);
        for (int64_t i = 0; i < (int64_t)R * Ish; i++) g[i] = g[i] / (1.f + expf(-g[i])) * u[i];
        matmul_d(y, g, &p->sh_d, R, Ish, D);
        for (int r = 0; r < R; r++) {
            const float *xr = x + (int64_t)r * D; float a = 0; for (int i = 0; i < D; i++) a += xr[i] * p->sh_gate[i];
            float sg = 1.f / (1.f + expf(-a));
            for (int i = 0; i < D; i++) out[(int64_t)r * D + i] += sg * y[(int64_t)r * D + i];
        }
        free(g); free(u); free(y);
    }
    if (R * K <= 64) {   /* a decode row's experts: one per thread (the matmuls inside run serially) */
        float *ye = falloc((int64_t)R * K * D);
        #pragma omp parallel for schedule(dynamic, 1)
        for (int j = 0; j < R * K; j++) {
            int e = idx[j];
            float *g = falloc(I), *u = falloc(I);
            const float *xr = x + (int64_t)(j / K) * D;
            matmul_d(g, xr, &p->eg[e], 1, D, I); matmul_d(u, xr, &p->eu[e], 1, D, I);
            for (int i = 0; i < I; i++) g[i] = g[i] / (1.f + expf(-g[i])) * u[i];
            matmul_d(ye + (int64_t)j * D, g, &p->ed[e], 1, I, D);
            free(g); free(u);
        }
        for (int j = 0; j < R * K; j++) {   /* rank order */
            float w = val[j], *o = out + (int64_t)(j / K) * D; const float *yr = ye + (int64_t)j * D;
            for (int i = 0; i < D; i++) o[i] += w * yr[i];
        }
        free(ye); free(idx); free(val); free(lg);
        return;
    }
    int *rows = malloc(sizeof(int) * (size_t)R * K);
    float *xe = falloc((int64_t)R * K * D), *ge = falloc((int64_t)R * K * I), *ue = falloc((int64_t)R * K * I), *ye = falloc((int64_t)R * K * D);
    for (int e = 0; e < E; e++) {   /* the routed experts, each over its rows */
        int n = 0;
        for (int i = 0; i < R * K; i++) if (idx[i] == e) rows[n++] = i;
        if (!n) continue;
        for (int j = 0; j < n; j++) memcpy(xe + (int64_t)j * D, x + (int64_t)(rows[j] / K) * D, sizeof(float) * D);
        matmul_d(ge, xe, &p->eg[e], n, D, I); matmul_d(ue, xe, &p->eu[e], n, D, I);
        for (int64_t i = 0; i < (int64_t)n * I; i++) ge[i] = ge[i] / (1.f + expf(-ge[i])) * ue[i];
        matmul_d(ye, ge, &p->ed[e], n, I, D);
        for (int j = 0; j < n; j++) {
            float w = val[rows[j]]; float *o = out + (int64_t)(rows[j] / K) * D; const float *yr = ye + (int64_t)j * D;
            for (int i = 0; i < D; i++) o[i] += w * yr[i];
        }
    }
    free(idx); free(val); free(lg); free(rows); free(xe); free(ge); free(ue); free(ye);
}

/* The model's lm_head on one normed row: on the device when the chain holds it */
static void q36_mtp_lm_head(Model *m, const float *row, float *logits);
static void q36_mtp_lm_rows(Model *m, const float *row, const int *ids, int n, float *scores);

/* A fed row's only lasting effect is its K/V row (the head reads nothing else of it
 * later): fc, the input norm, k and v, the k norm and RoPE, into the head's cache,
 * as attention() stores them. */
static void q36_mtp_kv_rows(Model *m, const int *tok, const float *hid, int R, int pos0) {
    Cfg *c = &m->c; Q36Mtp *p = &g_q36_mtp; int D = c->hidden, KV = c->kv_heads, kvd = c->k_head_dim, kvo = KV * kvd;
    double t0 = now_s();
    float *cat = falloc((int64_t)R * 2 * D), *x = falloc((int64_t)R * D), *nrm = falloc((int64_t)R * D);
    float *k = falloc((int64_t)R * kvo), *v = falloc((int64_t)R * kvo);
    for (int r = 0; r < R; r++) {
        float *em = cat + (int64_t)r * 2 * D;
        q36_embed_row(m, tok[r], pos0 + r, em);
        rmsnorm_row(em, em, p->enorm, D, c->eps);
        rmsnorm_row(cat + (int64_t)r * 2 * D + D, hid + (int64_t)r * D, p->hnorm, D, c->eps);
    }
    matmul_d(x, cat, &p->eh, R, 2 * D, D);
    for (int r = 0; r < R; r++) rmsnorm_row(nrm + (int64_t)r * D, x + (int64_t)r * D, p->L.in_ln, D, c->eps);
    matmul_d(k, nrm, &p->L.k, R, D, kvo); matmul_d(v, nrm, &p->L.v, R, D, kvo);
    for (int r = 0; r < R; r++) for (int h = 0; h < KV; h++) {
        float *kh = k + (int64_t)r * kvo + h * kvd;
        if (p->L.kn) rmsnorm_row(kh, kh, p->L.kn, kvd, c->eps);
        if (m->mpos || m->rope_delta) { int p3[3]; mrope_at(m, pos0 + r, p3); rope_head_mrope(kh, p3, c->mrope_section, c->rotary_dim, c->theta); }
        else rope_head_partial(kh, pos0 + r, c->rotary_dim, kvd, c->theta);
        size_t dst = ((size_t)h * m->max_t + pos0 + r) * kvd;
        q36_kv_put(p->Kc, (int64_t)dst, kh, kvd);
        q36_kv_put(p->Vc, (int64_t)dst, v + (int64_t)r * kvo + h * kvd, kvd);
    }
    free(cat); free(x); free(nrm); free(k); free(v);
    p->kv_rows += (unsigned long long)R; p->kv_ms += (now_s() - t0) * 1e3;
}


/* R head rows at positions pos0 .. pos0+R-1: token tok[r] with the model's row hid[r]
 * (its output at pos0+r-1, after the final norm); the last row's logits when logits,
 * and with hid_out that row's output after mtp.norm (a deeper draft's input).
 * Without logits only the rows' K/V matter (q36_mtp_kv_rows). */
static void q36_mtp_rows(Model *m, const int *tok, const float *hid, int R, int pos0, float *logits, float *hid_out) {
    Cfg *c = &m->c; Q36Mtp *p = &g_q36_mtp; int D = c->hidden;
    if (!q36_mtp_kv(m)) return;
    pos0--; /* pair h_p/token_{p+1} occupies head cache row p; no dummy row */
    if (!logits) { q36_mtp_kv_rows(m, tok, hid, R, pos0); return; }
    double t0 = now_s();
    float *cat = falloc((int64_t)R * 2 * D), *x = falloc((int64_t)R * D), *nrm = falloc((int64_t)R * D), *tmp = falloc((int64_t)R * D);
    for (int r = 0; r < R; r++) {
        float *em = cat + (int64_t)r * 2 * D;
        q36_embed_row(m, tok[r], pos0 + r, em);
        rmsnorm_row(em, em, p->enorm, D, c->eps);
        rmsnorm_row(cat + (int64_t)r * 2 * D + D, hid + (int64_t)r * D, p->hnorm, D, c->eps);
    }
    matmul_d(x, cat, &p->eh, R, 2 * D, D);
    for (int r = 0; r < R; r++) rmsnorm_row(nrm + (int64_t)r * D, x + (int64_t)r * D, p->L.in_ln, D, c->eps);
    void **K0 = m->K, **V0 = m->V;
    m->K = p->Kp; m->V = p->Vp;
    attention(m, &p->L, c->n_layers, nrm, R, pos0, tmp);
    m->K = K0; m->V = V0;
    for (int64_t i = 0; i < (int64_t)R * D; i++) x[i] += tmp[i];
    for (int r = 0; r < R; r++) rmsnorm_row(nrm + (int64_t)r * D, x + (int64_t)r * D, p->L.post_ln, D, c->eps);
    q36_mtp_moe(nrm, R, D, tmp);
    for (int64_t i = 0; i < (int64_t)R * D; i++) x[i] += tmp[i];
    if (logits) {
        float *last = falloc(D);
        rmsnorm_row(last, x + (int64_t)(R - 1) * D, p->head_norm, D, c->eps);
        if (p->nvoc) {
            float *scores = falloc(p->nvoc);
            matmul_d(scores, last, &p->dvoc, 1, D, p->nvoc);
            for (int i = 0; i < c->vocab; i++) logits[i] = -INFINITY;
            for (int j = 0; j < p->nvoc; j++) logits[p->vid[j]] = scores[j];
            free(scores);
        } else if (p->voc) {
            const int *ids = p->voc->list; int n = p->voc->list_n;
            float *scores = falloc(n);
            q36_mtp_lm_rows(m, last, ids, n, scores);
            int b = 0;
            for (int j = 1; j < n; j++) if (scores[j] > scores[b]) b = j;
            float z = 0.f, sb = scores[b];
            for (int j = 0; j < n; j++) if (scores[j] - sb > -20.f) z += expf(scores[j] - sb);
            p->pick_id = ids[b]; p->last_p = z > 0.f ? (float)(1.0 / z) : 0.f; p->pick_ok = 1;
            free(scores);
        } else q36_mtp_lm_head(m, last, logits);
        if (hid_out) memcpy(hid_out, last, sizeof(float) * D);   /* a deeper draft's: after mtp.norm */
        free(last);
    }
    free(cat); free(x); free(nrm); free(tmp);
    p->rows += (unsigned long long)R; p->ms += (now_s() - t0) * 1e3;
}

/* After a forward fed ids[0..S) at pos_base with its residual rows res[S] (normalized here):
 * the head reads the rows it completes and keeps the last one pending. A forward that
 * starts behind the head (a verify's rejected rows undone) continues from the forward
 * before's row there; one that leaves a gap stops the head until the next prompt. */
static void q36_mtp_feed_rows(Model *m, const int *ids, int S, int pos_base, const float *hid);
static void q36_mtp_feed(Model *m, const int *ids, int S, int pos_base, const float *res) {
    Q36Mtp *p = &g_q36_mtp; int D = m->c.hidden;
    if (!p->on || S < 1 || !q36_mtp_kv(m)) return;
    float *hid = falloc((int64_t)S * D);   /* h_p: the model's output, after its final norm */
    #pragma omp parallel for schedule(static) if(S > 8)
    for (int s = 0; s < S; s++) rmsnorm_row(hid + (int64_t)s * D, res + (int64_t)s * D, m->final_norm, D, m->c.eps);
    q36_mtp_feed_rows(m, ids, S, pos_base, hid);
    free(hid);
}
static void q36_mtp_feed_rows(Model *m, const int *ids, int S, int pos_base, const float *hid) {
    Q36Mtp *p = &g_q36_mtp; int D = m->c.hidden;
    if (pos_base == 0) { p->len = 0; p->has_pend = 0; p->done_pos = -1; }
    else {
        /* the model's row at pos_base - 1: pending, or in the newest forward's rows */
        int prev = pos_base - 1;
        if (!(p->has_pend && p->len == prev)) {
            if (p->last && prev >= p->last_base && prev < p->last_base + p->last_n) {
                memcpy(p->pend, p->last + (int64_t)(prev - p->last_base) * D, sizeof(float) * D);
                p->len = prev; p->has_pend = 1;
            } else { p->has_pend = 0; p->len = -1; }
        }
        if (p->has_pend) {
            if (!(p->done_pos == pos_base && p->done_tok == ids[0]))
                q36_mtp_rows(m, ids, p->pend, 1, pos_base, NULL, NULL);
        }
    }
    p->done_pos = -1;
    if (pos_base == 0 || p->has_pend) {
        if (S > 1) q36_mtp_rows(m, ids + 1, hid, S - 1, pos_base + 1, NULL, NULL);
        memcpy(p->pend, hid + (int64_t)(S - 1) * D, sizeof(float) * D);
        p->len = pos_base + S - 1; p->has_pend = 1;
    }
    if (S > p->last_cap) { free(p->last); p->last = falloc((int64_t)S * D); p->last_cap = S; }
    memcpy(p->last, hid, sizeof(float) * (size_t)S * D);
    p->last_base = pos_base; p->last_n = S;
}

/* Whether the head can draft for a token about to be fed at pos */
static int q36_mtp_ready(int pos) {
    const Q36Mtp *p = &g_q36_mtp;
    if (!p->on) return 0;
    if (p->has_pend && p->len == pos - 1) return 1;
    /* after a verify's rejected rows: the model's row at pos - 1 is one the newest
     * forward computed (the head reads it from there, as q36_mtp_feed does) */
    return p->last && pos - 1 >= p->last_base && pos - 1 < p->last_base + p->last_n;
}
/* The pending row made the model's row at pos - 1 (q36_mtp_ready said it is there). */
static void q36_mtp_pend_at(Model *m, int pos) {
    Q36Mtp *p = &g_q36_mtp; int D = m->c.hidden;
    if (p->has_pend && p->len == pos - 1) return;
    memcpy(p->pend, p->last + (int64_t)(pos - 1 - p->last_base) * D, sizeof(float) * D);
    p->len = pos - 1; p->has_pend = 1;
}
/* The head's draft for the token after `tok` (about to be fed at pos): its row at pos;
 * `hidden` (D floats, may be NULL) gets the row's state for a deeper draft */
static int q36_mtp_draft(Model *m, int tok, int pos, float *hidden) {
    Q36Mtp *p = &g_q36_mtp; int V = m->c.vocab;
    float *lg = falloc(V);
    q36_mtp_pend_at(m, pos);
    q36_mtp_rows(m, &tok, p->pend, 1, pos, lg, hidden);
    const char *dump = getenv("Q36_MTP_DUMP");
    if (dump && *dump) {
        FILE *f = fopen(dump, "ab"); int row = pos - 1;
        if (!f || fwrite(&row, 4, 1, f) != 1 || fwrite(&tok, 4, 1, f) != 1 ||
            fwrite(lg, 4, V, f) != (size_t)V) { fprintf(stderr, "Q36_MTP_DUMP write failed\n"); exit(1); }
        fclose(f);
    }
    p->done_pos = pos; p->done_tok = tok; p->drafts++;
    int best = 0;
    if (p->pick_ok) { best = p->pick_id; p->pick_ok = 0; }   /* last_p set there */
    else {
        for (int i = 1; i < V; i++) if (lg[i] > lg[best]) best = i;
        /* the draft's softmax probability: how deep a verify should go (Q36_MTP_PMIN) */
        float z = 0, mx = lg[best];
        for (int i = 0; i < V; i++) if (lg[i] - mx > -20.f) z += expf(lg[i] - mx);
        p->last_p = z > 0 ? (float)(1.0 / z) : 0.f;
    }
    free(lg);
    return best;
}

/* A deeper draft: the head's row at `pos` on its own state of the row before (`hidden`,
 * becomes this row's) with the draft just proposed. Its K/V row there is a tail: the
 * fed row the verify's standing rows bring overwrites it. */
static int q36_mtp_draft_more(Model *m, float *hidden, int tok, int pos) {
    Q36Mtp *p = &g_q36_mtp; int V = m->c.vocab;
    float *lg = falloc(V), *h = falloc(m->c.hidden);
    memcpy(h, hidden, sizeof(float) * (size_t)m->c.hidden);
    q36_mtp_rows(m, &tok, h, 1, pos, lg, hidden);
    p->drafts++;
    int best = 0;
    if (p->pick_ok) { best = p->pick_id; p->pick_ok = 0; }   /* last_p set there */
    else {
        for (int i = 1; i < V; i++) if (lg[i] > lg[best]) best = i;
        /* the draft's softmax probability: how deep a verify should go (Q36_MTP_PMIN) */
        float z = 0, mx = lg[best];
        for (int i = 0; i < V; i++) if (lg[i] - mx > -20.f) z += expf(lg[i] - mx);
        p->last_p = z > 0 ? (float)(1.0 / z) : 0.f;
    }
    free(lg); free(h);
    return best;
}

/* Q36_MTP=1: the head, after the model and the device are up */
static void q36_mtp_attach(Model *m) {
    const char *path = getenv("Q36_MTP");
    if (!path || strcmp(path, "1") || g_q36_mtp.on) return;
    /* where a verify would not reproduce plain decoding there are no drafts (q36_spec_begin):
     * a head there would only be fed, every row brought back from the chain for nothing.
     * Under the CUDA chain with the whole tier there are, and the head runs on its device
     * (qwen36_mtp_cuda.h). */
    if (qt_ready() || m->dn_dev) {
        fprintf(stderr, "[mtp] Q36_MTP=1: the head stays off under a CUDA tier holding part of the experts, "
                        "where there are no speculative steps\n");
        return;
    }
    if (g_cache_route || qq_active()) {
        fprintf(stderr, "[mtp] Q36_MTP=1: the head stays off under %s, where there are no speculative steps\n",
                g_cache_route ? "CACHE_ROUTE" : "a qpack container");
        return;
    }
    if (!q36_mtp_load(m)) { fprintf(stderr, "[mtp] Q36_MTP=%s: the head stays off\n", path); return; }
}
/* A request starts: its prompt joins the list and counts, but the tokens serve kept from the
 * last request (counted then) do not count again. draft_vocab_begin rebuilds the list. */
static void q36_mtp_vocab_begin(const int *prompt, int n) {
    Q36Mtp *p = &g_q36_mtp;
    int reuse = p->voc_reuse; p->voc_reuse = 0;
    if (!p->on || !p->voc || !prompt || n <= 0) return;
    if (reuse < 0 || reuse > n) reuse = 0;
    draft_vocab_begin(p->voc, prompt, n, reuse);
    p->voc_gen++;
}
/* A token the request feeds: counted, and appended to the list when it is not in it. */
static void q36_mtp_vocab_step(int tok) {
    Q36Mtp *p = &g_q36_mtp;
    if (!p->on || !p->voc) return;
    uint64_t r = p->voc->run.rebuilds;
    draft_vocab_count_token(p->voc, tok);
    if (p->voc->run.rebuilds != r) p->voc_gen++;   /* a new list, not ids at its tail: the device's copy is stale */
}
/* The counts beside the model (.coli_vocab, or COLI_VOCAB_USAGE) at exit; USAGE_SAVE=0 keeps them. */
static void q36_mtp_vocab_save(void) {
    Q36Mtp *p = &g_q36_mtp;
    if (!p->on || !p->voc || !rt_save_enabled()) return;
    double tokens; int distinct;
    draft_vocab_totals(p->voc, &tokens, &distinct);
    switch (draft_vocab_save(p->voc)) {
    case DRAFT_VOCAB_SAVED:
        fprintf(stderr, "[STATS] %.0f tokens across %d distinct -> %s\n", tokens, distinct, p->voc->path);
        break;
    case DRAFT_VOCAB_LEFT_ALONE:
        fprintf(stderr, "[STATS] no counts are written to %s. Set COLI_VOCAB_USAGE to use another file.\n", p->voc->path);
        break;
    case DRAFT_VOCAB_NOT_SAVED:
        fprintf(stderr, "[STATS] the draft vocabulary counts were not written to %s. Make the directory "
                        "writable, or point COLI_VOCAB_USAGE at a file that is.\n", p->voc->path);
        break;
    }
}
static void q36_mtp_report(void) {
    const Q36Mtp *p = &g_q36_mtp;
    if (!p->on) return;
    fprintf(stderr, "[mtp] head rows %llu (%.2f ms each), fed rows %llu (K/V only, %.2f ms each), drafts %llu\n", p->rows,
            p->rows ? p->ms / (double)p->rows : 0.0, p->kv_rows, p->kv_rows ? p->kv_ms / (double)p->kv_rows : 0.0, p->drafts);
    if (p->voc) {
        DraftVocabSummary s = draft_vocab_summary(p->voc);
        fprintf(stderr, "[mtp] draft list %.0f tokens on average, %.1f%% of picks inside, %llu ids appended, %llu rebuilds\n",
                s.average_list_tokens, s.inside_percent, (unsigned long long)s.appended, (unsigned long long)s.rebuilds);
    }
}
