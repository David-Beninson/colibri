/* omp_tune.h — dimensionamento della squadra OpenMP sui CORE FISICI.
 *
 * PERCHE' SOLO IL DIMENSIONAMENTO, E NON LO SPIN-WAIT.
 * Il blocco di tuning in colibri.c fa due cose che hanno profili di rischio
 * OPPOSTI, e vanno tenute separate:
 *
 *   dimensionamento  OMP_NUM_THREADS = core fisici (niente SMT)
 *                    #718: +2.3x su Zen3 (5950X, 16C/32T) solo cambiando
 *                    il numero di thread. Il guadagno e' cosi' grande che
 *                    "sommerge la maggior parte dei delta che si citano qui".
 *
 *   spin-wait        OMP_WAIT_POLICY=active, GOMP_SPINCOUNT, KMP_BLOCKTIME
 *                    #707: -2.2x sul decode di un host a bassa residenza
 *                          (M1 Max 32 GB, ~10% di expert residenti)
 *                    #116: -39% su Metal      #159: ~3x su x86+CUDA
 *                    #341: 3000% di CPU su FreeBSD con la squadra ferma
 *                    Meccanismo: dove il token e' fatto di byte dal disco,
 *                    una squadra che gira a vuoto ruba i core al pool di I/O
 *                    che sta facendo il lavoro vero.
 *
 * Kimi K3 e OLMoE non avevano NESSUNA delle due. Qui prendono solo la prima:
 * Kimi e' il motore piu' disk-bound del progetto (misurato: 6.7% di hit,
 * 891 GB letti per 32 token), cioe' esattamente il regime in cui la seconda
 * meta' fa danno. Aggiungergliela sarebbe stato un peggioramento misurabile.
 * GLM now calls this same helper after its optional hot-team re-exec, so it
 * receives the physical-core sizing without changing the independent spin-wait
 * policy that its existing block controls.
 *
 * PERCHE' QUI NON SERVE IL RE-EXEC.
 * colibri.c si ri-esegue perche' OMP_WAIT_POLICY & co. le legge il COSTRUTTORE
 * di libgomp, prima di main(): un setenv() dentro main() arriva tardi. Il numero
 * di thread no: omp_set_num_threads() e' una API di runtime e ha effetto subito.
 * La meta' sicura e' anche la meta' semplice.
 *
 * REGOLA SUI FALLIMENTI: se il conteggio dei core fisici non e' determinabile,
 * NON si indovina — si lascia il default di OpenMP. Un conteggio sbagliato e'
 * peggio di nessun conteggio (cfr. #325, dove un fallback silenzioso a 1
 * inchiodava il decode su un core solo).
 */
#ifndef COLI_OMP_TUNE_H
#define COLI_OMP_TUNE_H

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>

/* Numero di core FISICI, o 0 se non determinabile. Mai un valore inventato. */
static int coli_physical_cores(void)
{
    /* Apple Silicon: every core but the efficiency tier's. hw.physicalcpu counts
     * the E-cores too (10 on an M1 Max), and with a barrier per matmul the slowest
     * thread sets the pace: the E-cores slowed the team (#707, -4.2% decode). */
    int n = 0; size_t sz = sizeof(n);
    if (sysctlbyname("hw.nperflevels", &n, &sz, NULL, 0) == 0 && n >= 1) {
        int tot = 0, last = n > 1 ? n - 1 : 1;
        for (int i = 0; i < last && i < 16; i++) {
            char nm[48]; snprintf(nm, sizeof nm, "hw.perflevel%d.logicalcpu", i);
            int c = 0; sz = sizeof(c);
            if (sysctlbyname(nm, &c, &sz, NULL, 0) == 0 && c > 0) tot += c;
        }
        if (tot > 0) return tot;
    }
    int v = 0; sz = sizeof(v);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &v, &sz, NULL, 0) == 0 && v > 0) return v;
    v = 0; sz = sizeof(v);
    if (sysctlbyname("hw.physicalcpu", &v, &sz, NULL, 0) == 0 && v > 0) return v;
    return 0;
}

static int coli_performance_cores(void)
{
    return 0;   /* macOS: coli_physical_cores counts the performance level already */
}

/* Dimensiona la squadra OpenMP sui core fisici. Rispetta OMP_NUM_THREADS se
 * l'utente l'ha impostata, e non fa nulla se il conteggio non e' affidabile.
 * `engine` finisce solo nella riga di log. */
static void coli_omp_tune_threads(const char *engine)
{
#ifdef _OPENMP
    const char *off = getenv("COLI_NO_OMP_TUNE");
    if (off) return;                       /* stesso kill-switch degli altri motori */
    if (getenv("OMP_NUM_THREADS")) return; /* l'utente comanda */

    int phys = coli_physical_cores();
    if (phys <= 0) return;                 /* sconosciuto -> default di OpenMP */
    int perf = coli_performance_cores();   /* a hybrid CPU: its P-cores pace the team */
    int team = perf > 0 && perf < phys ? perf : phys;
    int logical = omp_get_max_threads();
    if (team >= logical) return;           /* niente SMT da evitare: silenzio */

    omp_set_num_threads(team);
    if (team < phys)
        fprintf(stderr, "[OMP] %s: %d performance-core threads instead of %d logical CPUs "
                        "(%d physical cores, the efficiency cores left out: the slowest thread "
                        "sets the pace of every matmul); set OMP_NUM_THREADS=<n> to override\n",
                engine, team, logical, phys);
    else
        fprintf(stderr, "[OMP] %s: %d physical-core threads instead of %d logical CPUs; "
                        "SMT can halve decode throughput on some CPUs (#718); "
                        "set OMP_NUM_THREADS=<n> to override\n",
                engine, phys, logical);
#else
    (void)engine;
#endif
}

/* 1 when an engine switch such as COLI_CUDA is on (set, and not 0/false/off/no) */
static inline int coli_omp_env_on(const char *name)
{
    const char *v = getenv(name);
    if (!v || !*v) return 0;
    return !(strcmp(v, "0") == 0 || strcmp(v, "false") == 0 || strcmp(v, "off") == 0 || strcmp(v, "no") == 0);
}

static void coli_omp_hot_team(char **argv, const char *engine, int gpu_on)
{
    (void)argv; (void)engine; (void)gpu_on;
}

#endif /* COLI_OMP_TUNE_H */
