/* compat.h — shim di portabilita' per piattaforme non-Linux (oggi: macOS / Apple Silicon,
 * Windows 11 x86-64 via MinGW-w64).
 * Regola: ogni differenza di piattaforma vive QUI; i .c restano puliti.
 *
 * Storicamente su Linux questo header era un NO-OP totale (solo shim per le altre
 * piattaforme). Non lo e' piu': coli_stdin_readable() definisce anche il ramo POSIX,
 * perche' un helper *portabile* deve esistere su tutte le piattaforme -- altrimenti i
 * .c dovrebbero avere il proprio #ifdef, che e' esattamente cio' che la regola vieta.
 * Resta vero che il percorso Linux non e' alterato: nulla viene ridefinito, e la
 * funzione e' static inline, quindi un TU che non la chiama non paga nulla. */
#ifndef COMPAT_H
#define COMPAT_H

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/select.h>
#include <mach/mach.h>
#include <sys/sysctl.h>

/* --- posix_fadvise: assente su macOS ---
 * WILLNEED -> F_RDADVISE (readahead esplicito: stessa semantica).
 * DONTNEED -> no-op: XNU non espone un drop mirato per-range; la sua unified
 *             buffer cache si autoregola sotto pressione. Il motore usa DONTNEED
 *             solo come consiglio, quindi ignorarlo e' corretto (e su una macchina
 *             con molta RAM tenere le pagine e' proprio cio' che si vuole). */
#ifndef POSIX_FADV_NORMAL
#define POSIX_FADV_NORMAL      0
#define POSIX_FADV_RANDOM      1
#define POSIX_FADV_SEQUENTIAL  2
#define POSIX_FADV_WILLNEED    3
#define POSIX_FADV_DONTNEED    4
#define POSIX_FADV_NOREUSE     5
#endif
static inline int compat_fadvise(int fd, off_t off, off_t len, int advice){
    if(advice==POSIX_FADV_WILLNEED){
        struct radvisory ra;
        ra.ra_offset = off;
        ra.ra_count  = (int)(len>0x7FFFFFFF ? 0x7FFFFFFF : len);
        return fcntl(fd, F_RDADVISE, &ra)<0 ? -1 : 0;
    }
    return 0;
}
#define posix_fadvise compat_fadvise

/* --- O_DIRECT: assente su macOS ---
 * L'equivalente e' F_NOCACHE sul fd (bypass della unified buffer cache). */
static inline int compat_open_direct(const char *path){
    int fd = open(path, O_RDONLY);
    if(fd>=0) fcntl(fd, F_NOCACHE, 1);
    return fd;
}

#ifndef getenv_utf8
#define getenv_utf8(name) getenv(name)
#endif

#ifndef compat_aligned_free
#define compat_aligned_free free
#endif

#ifndef COMPAT_O_RDONLY
#define COMPAT_O_RDONLY O_RDONLY
#endif
#ifndef COMPAT_O_BINARY
#define COMPAT_O_BINARY 0
#endif

#ifndef compat_open_overlapped
#define compat_open_overlapped(path) (-1)
#endif

/* --- read-only file mapping ------------------------------------------------- */
typedef struct {
    void *base;
    size_t len;
} compat_ro_map;

static inline int compat_map_readonly(int fd, int64_t off, size_t len,
                                      compat_ro_map *map, const void **data)
{
    if (!map || !data || off < 0 || len == 0) { errno = EINVAL; return -1; }
    memset(map, 0, sizeof(*map));
    long pg = sysconf(_SC_PAGESIZE);
    if (pg <= 0) pg = 16384;
    uint64_t gran = (uint64_t)pg;
    uint64_t aligned = (uint64_t)off - ((uint64_t)off % gran);
    uint64_t delta = (uint64_t)off - aligned;
    if (delta > SIZE_MAX || len > SIZE_MAX - (size_t)delta) { errno = EOVERFLOW; return -1; }
    size_t view_len = (size_t)delta + len;
    void *base = mmap(NULL, view_len, PROT_READ, MAP_SHARED, fd, (off_t)aligned);
    if (base == MAP_FAILED) return -1;
    map->base = base;
    map->len = view_len;
    *data = (const char*)base + (size_t)delta;
    return 0;
}

static inline void compat_unmap_readonly(compat_ro_map *map)
{
    if (!map || !map->base) return;
    munmap(map->base, map->len);
    memset(map, 0, sizeof(*map));
}

static inline int coli_stdin_readable(void)
{
    fd_set r; struct timeval tv = {0, 0};
    FD_ZERO(&r); FD_SET(0, &r);
    return select(1, &r, NULL, NULL, &tv) > 0 && FD_ISSET(0, &r);
}

static inline void coli_serve_binary_mode(void)
{
}

static inline int coli_console_is_own(void)
{
    return 0;
}

static inline void coli_hold_console(void)
{
}

static inline void coli_print_launcher_help(const char *engine, const char *by_hand)
{
    const char *run = "./coli";
    fprintf(stderr,
        "colibri: this is the %s engine, and it was started without a model.\n"
        "The engine is not the program you run directly -- the launcher is:\n"
        "\n"
        "    %s chat  --model <model directory>    interactive chat\n"
        "    %s serve --model <model directory>    OpenAI-compatible API\n"
        "    %s web   --model <model directory>    API plus the dashboard\n"
        "    %s doctor --model <model directory>   check a model is usable\n"
        "\n"
        "The launcher needs Python 3 and picks the right engine for the model.\n"
        "(Running the engine by hand: %s)\n"
        "Getting a model, step by step: https://github.com/JustVugg/colibri"
        "/blob/main/docs/quickstart.md\n",
        engine, run, run, run, run, by_hand);
}

static inline void compat_meminfo_gb(double *total_gb, double *avail_gb){
    double total = 0, avail = 0;
    uint64_t memsize = 0; size_t len = sizeof memsize;
    if(sysctlbyname("hw.memsize", &memsize, &len, NULL, 0) == 0) total = (double)memsize / 1e9;
    mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vm;
    if(host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm, &cnt) == KERN_SUCCESS)
        avail = ((double)vm.free_count + (double)vm.inactive_count + (double)vm.purgeable_count)
                * (double)sysconf(_SC_PAGESIZE) / 1e9;
    if(total_gb) *total_gb = total;
    if(avail_gb) *avail_gb = avail;
}

static inline double compat_mem_available_gb(void){
    double avail = 0;
    compat_meminfo_gb(NULL, &avail);
    return avail;
}

#endif /* COMPAT_H */
