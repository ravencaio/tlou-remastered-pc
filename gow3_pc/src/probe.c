/* Minimal x86-64 native loader experiment. Not a PS4 emulator or game port. */
#define _GNU_SOURCE
#include <stdint.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "runtime.h"
#include "cpu_check.h"
#include "red_zone.h"
#include "orb_hook.h"
#include "menu_exit_hook.h"
#include "runtime_menu_exit.h"
#include "runtime_menu_pc.h"
#include "runtime_menu_saves.h"
#include "cheat_hook.h"
#include "actor_hook.h"
#include "damage_hook.h"
#include "near_gap.h"
#include "runtime_mouselook.h"
#include "runtime_gamesettings.h"
#include "gpu/gow3gpu.h"
#if !defined(__x86_64__) || !defined(__GNUC__)
#error This prototype requires x86-64 GCC or Clang (including MinGW).
#endif
#include <x86intrin.h>
#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#else
#include <sys/mman.h>
#include <malloc.h>
#include <unistd.h>
#include <signal.h>
#include <ucontext.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <time.h>
#endif

typedef struct { uint64_t address, size, flags; } Segment;
typedef struct { uint64_t target, kind, value, addend; } Reloc;
static char (*names)[128];
static uint64_t import_count;
static unsigned char *image;
static size_t page_size;
typedef struct { uint64_t base, size, init, tls_address, tls_memsz, tls_filesz, tls_module; } LinkedModule;
static LinkedModule modules[16];
static uint64_t module_count;
static int entered_game;
static int gpu_enabled;
int vulkan_smoke(void);

static void fail(const char *message) { fprintf(stderr, "ERROR: %s\n", message); exit(1); }
static uint64_t read64(FILE *f) {
    unsigned char b[8];
    if (fread(b, 1, 8, f) != 8) fail("truncated boot file");
    uint64_t n = 0;
    for (int i = 7; i >= 0; --i) n = (n << 8) | b[i];
    return n;
}
/* Red-zone protection (scripts/red_zone.py): guest code keeps data in the 128 bytes below rsp and
 * Windows writes its exception frame there on every GPU page-tracking fault. */
static RedZoneTable *red_zone;
static size_t round_page(size_t size) { return (size + page_size - 1) & ~(page_size - 1); }
static void *allocate(size_t size) {
    void *low=runtime_low_map(size,PROT_READ|PROT_WRITE);
    if (low) return low;
#ifdef _WIN32
    void *p = VirtualAlloc(NULL, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!p) fail("VirtualAlloc failed");
#else
    void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) fail("mmap failed");
#endif
    return p;
}
/* Some games encode image pointers as offsets in a fixed 2 GiB static range.
 * Keep the image there without replacing any existing host mapping. */
static void *allocate_low_image(size_t size) {
    for (uintptr_t address=UINT64_C(0x400000); address+size<UINT64_C(0x80000000); address+=UINT64_C(0x1000000)) {
#ifdef _WIN32
        void *p=VirtualAlloc((void *)address,size,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE);
        if (p) return p;
#else
        void *p=mmap((void *)address,size,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED_NOREPLACE,-1,0);
        if (p!=MAP_FAILED) return p;
#endif
    }
    fail("cannot allocate guest image below 2 GiB");
    return NULL;
}
static void protect(void *p, size_t size, unsigned flags) {
#ifdef _WIN32
    DWORD old, mode = PAGE_NOACCESS;
    if (flags & 1) mode = (flags & 2) ? PAGE_EXECUTE_READWRITE : PAGE_EXECUTE_READ;
    else if (flags & 2) mode = PAGE_READWRITE;
    else if (flags & 4) mode = PAGE_READONLY;
    if (!VirtualProtect(p, size, mode, &old)) fail("VirtualProtect failed");
    FlushInstructionCache(GetCurrentProcess(), p, size);
#else
    int mode = ((flags & 4) ? PROT_READ : 0) | ((flags & 2) ? PROT_WRITE : 0) | ((flags & 1) ? PROT_EXEC : 0);
    if (mprotect(p, size, mode)) fail("mprotect failed");
#endif
}
static ABI __attribute__((noreturn)) void unresolved(uint32_t id, uintptr_t argument) {
    if (id >= import_count) fail("bad import trap index");
    printf("STOP: first unsupported PS4 import: %s (index %u)\n", names[id], id);
    printf("API: %s\n", runtime_import_name(names[id]));
    printf("Caller return offset: 0x%" PRIxPTR "; first argument: 0x%" PRIxPTR "\n",
           (uintptr_t)__builtin_return_address(0) - (uintptr_t)image, argument);
    uintptr_t caller=(uintptr_t)__builtin_return_address(0)-(uintptr_t)image;
    for (uint64_t m=0;m<module_count;++m)
        if (caller>=modules[m].base && caller-modules[m].base<modules[m].size)
            printf("Caller in linked module %" PRIu64 " (%s): +0x%" PRIxPTR "\n",m,m==0 ? "libc.prx" : "system module",caller-modules[m].base);
    runtime_report();
    puts(entered_game ? "Original guest entry instructions executed; game initialization is incomplete." :
                        "Native libc initialization is incomplete; game entry has not run.");
    fflush(NULL);
    _exit(20); /* no destructors: GPU, audio and guest threads are still running */
}
#ifndef _WIN32
/* enter_on_stack(entry, arg0, arg1, stack_top): call entry(arg0,arg1) on a new stack. */
void enter_on_stack(void *entry, void *arg0, void *arg1, void *top);
__asm__(".text\n.globl enter_on_stack\nenter_on_stack:\n"
        " push %rbp\n mov %rsp,%rbp\n and $-16,%rcx\n mov %rcx,%rsp\n"
        " mov %rdi,%rax\n mov %rsi,%rdi\n mov %rdx,%rsi\n call *%rax\n"
        " mov %rbp,%rsp\n pop %rbp\n ret\n");
#endif
static ABI void guest_exit(void) { puts("Runtime: process finalizer callback reached"); }
/* GOW3_RED_ZONE_STATS=1: of the faults guest code took, how many were protected instructions. */
static void note_red_zone_fault(uintptr_t rip) {
    static int on=-1;
    if (on<0) { const char *env=getenv("GOW3_RED_ZONE_STATS"); on=env && env[0]=='1'; }
    if (!on || !red_zone || rip-(uintptr_t)image>=0x20000000) return;
    static uint64_t total, in_stubs;
    const uint64_t n=__atomic_add_fetch(&total,1,__ATOMIC_RELAXED);
    const int stub=rip-(uintptr_t)red_zone->rz_area<red_zone->rz_code_bytes;
    const uint64_t s=stub ? __atomic_add_fetch(&in_stubs,1,__ATOMIC_RELAXED) : __atomic_load_n(&in_stubs,__ATOMIC_RELAXED);
    if (!(n&8191)) printf("Red zone: %" PRIu64 " of %" PRIu64 " guest faults came from protected instructions\n",s,n);
}
#ifdef _WIN32
static void describe_address(char *out, size_t size, uintptr_t address) {
    HMODULE module=NULL;
    char name[MAX_PATH]="?";
    uint64_t protected_site;
    if (red_zone_site_of_stub(red_zone,address,&protected_site)) {
        snprintf(out,size,"guest offset 0x%" PRIx64 " (red-zone stub)",protected_site);
        return;
    }
    if (image && address-(uintptr_t)image<0x20000000) { snprintf(out,size,"guest offset 0x%" PRIxPTR,address-(uintptr_t)image); return; }
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,(LPCSTR)address,&module) && module) {
        GetModuleFileNameA(module,name,sizeof(name));
        const char *base=strrchr(name,'\\');
        snprintf(out,size,"%s+0x%" PRIxPTR,base ? base+1 : name,address-(uintptr_t)module);
        return;
    }
    snprintf(out,size,"%p",(void *)address);
}
/* Host code: unwound with the executable's unwind tables (optimized host code keeps no frame
 * pointers); stops at guest code, which has none. */
static void print_host_stack(const CONTEXT *c, int max_depth) {
    char where[MAX_PATH+64];
    CONTEXT ctx=*c;
    for (int depth=0; depth<max_depth; ++depth) {
        DWORD64 image_base=0;
        PRUNTIME_FUNCTION fn=RtlLookupFunctionEntry(ctx.Rip,&image_base,NULL);
        describe_address(where,sizeof(where),(uintptr_t)ctx.Rip);
        fprintf(stderr,"  host #%d %s\n",depth,where);
        if (!fn) {
            if (depth) break;
            /* A leaf function: its return address is on top of the stack. */
            DWORD64 ret=0; SIZE_T got=0;
            if (!ReadProcessMemory(GetCurrentProcess(),(void *)(uintptr_t)ctx.Rsp,&ret,sizeof(ret),&got) || got!=sizeof(ret)) break;
            ctx.Rip=ret; ctx.Rsp+=8;
            continue;
        }
        PVOID handler_data=NULL; DWORD64 establisher=0;
        RtlVirtualUnwind(UNW_FLAG_NHANDLER,image_base,ctx.Rip,fn,&ctx,&handler_data,&establisher,NULL);
        if (!ctx.Rip) break;
    }
}
/* gow3 21 Nov 2026: a C++ exception thrown in host code on a guest thread and not caught there
 * unwinds into guest frames, which have no unwind tables: the process ended with STATUS_GCC_THROW
 * (exit code 541541187), only "libunwind: pc not in table" in the log and no report (the frame
 * walk fails before any unhandled-exception filter runs). Vectored handlers see the throw before
 * the unwinding, with the thrower's stack intact: the first throws are logged with their type and
 * host stack (an exception caught later is harmless; the log says so). */
#define STATUS_GCC_THROW_CODE 0x20474343u
static volatile LONG cxx_throws;
static void report_cxx_throw(const EXCEPTION_POINTERS *info) {
    const LONG n=InterlockedIncrement(&cxx_throws);
    if (n>8) return;
    const EXCEPTION_RECORD *record=info->ExceptionRecord;
    char type[256]="?";
    if (record->NumberParameters>=1 &&
        !gow3gpu_describe_cxx_exception((const void *)record->ExceptionInformation[0],type,sizeof(type)))
        snprintf(type,sizeof(type),"unknown type");
    fprintf(stderr,"C++ exception thrown (%s) on thread %lu (fine when caught; a crash right after "
            "this ends with exit code 541541187)%s\n",type,GetCurrentThreadId(),
            n==8 ? "; further throws not logged" : "");
    print_host_stack(info->ContextRecord,24);
    fflush(stderr);
}
static int fatal_reported;
static LONG fatal_exception(EXCEPTION_POINTERS *info) {
    if (__atomic_exchange_n(&fatal_reported,1,__ATOMIC_ACQ_REL)) { Sleep(INFINITE); }
    const EXCEPTION_RECORD *record=info->ExceptionRecord;
    const CONTEXT *c=info->ContextRecord;
    char where[MAX_PATH+64];
    describe_address(where,sizeof(where),(uintptr_t)c->Rip);
    if (record->ExceptionCode==EXCEPTION_ACCESS_VIOLATION && record->NumberParameters>=2)
        fprintf(stderr,"Fault: %s access to %p at %s (thread %lu)\n",
                record->ExceptionInformation[0]==1 ? "write" : record->ExceptionInformation[0]==8 ? "execute" : "read",
                (void *)record->ExceptionInformation[1],where,GetCurrentThreadId());
    else
        fprintf(stderr,"Fault: exception 0x%08lx at %s (thread %lu)\n",record->ExceptionCode,where,GetCurrentThreadId());
    fprintf(stderr,"  rax=%016llx rbx=%016llx rcx=%016llx rdx=%016llx\n  rsi=%016llx rdi=%016llx rbp=%016llx rsp=%016llx\n",
            c->Rax,c->Rbx,c->Rcx,c->Rdx,c->Rsi,c->Rdi,c->Rbp,c->Rsp);
    if (gpu_enabled) gow3gpu_dump_guest_writes(info);
    print_host_stack(c,40);
    /* rbp chain (guest and host code keep frame pointers). */
    uintptr_t rbp=(uintptr_t)c->Rbp;
    for (int depth=0; depth<24 && rbp; ++depth) {
        uintptr_t frame[2];
        SIZE_T got=0;
        if (!ReadProcessMemory(GetCurrentProcess(),(void *)rbp,frame,sizeof(frame),&got) || got!=sizeof(frame) || frame[0]<=rbp) break;
        describe_address(where,sizeof(where),frame[1]);
        fprintf(stderr,"  #%d %s\n",depth,where);
        rbp=frame[0];
    }
    fflush(NULL);
    TerminateProcess(GetCurrentProcess(),3);
    return EXCEPTION_CONTINUE_SEARCH;
}
/* Data watchpoints (diagnostics): tools/watch_write.py arms DR0-DR3 on the guest threads; each hit
 * is logged with the guest offset of the writing code (trap: RIP is after the store) and guest
 * return addresses near the stack top, then execution continues. */
/* Up to 63 printable bytes at a guest pointer (diagnostics; never faults). */
static void watch_string(char *out, size_t size, uint64_t address) {
    unsigned char buf[64]={0}; SIZE_T got=0;
    out[0]=0;
    if (address<0x10000 || !ReadProcessMemory(GetCurrentProcess(),(LPCVOID)(uintptr_t)address,buf,sizeof(buf)-1,&got)) return;
    size_t n=0;
    while (n<got && n+1<size && buf[n]>=0x20 && buf[n]<0x7f) { out[n]=(char)buf[n]; ++n; }
    out[n]=0;
}
static volatile LONG watch_hits;
static LONG watch_hit(EXCEPTION_POINTERS *info) {
#define hits watch_hits
    CONTEXT *c=info->ContextRecord;
    DWORD64 dr6=c->Dr6;
    c->Dr6=0;
    /* Instruction breakpoints (GOW3_EXEC_WATCH) fault before the instruction: resume past them. */
    c->EFlags|=0x10000;
    /* Only guest code is logged: runtime copies of the watched memory would fill the 400 lines. */
    /* GOW3_WATCH_HITS: lines to log (default 400). */
    static LONG max_hits;
    if (!max_hits) { const char *e=getenv("GOW3_WATCH_HITS"); LONG v=e ? atol(e) : 0; max_hits=v>0 ? v : 400; }
    if (image && c->Rip-(uintptr_t)image<0x2000000 && InterlockedIncrement(&hits)<=max_hits) {
        char line[1400], s1[64], s2[64];
        watch_string(s1,sizeof(s1),c->Rsi); watch_string(s2,sizeof(s2),c->Rdx);
        int n=snprintf(line,sizeof(line),"Watch: dr6 %#llx rip guest %#llx rax %#llx rcx %#llx rdx %#llx rsi %#llx rdi %#llx r8 %#llx r9 %#llx rbx %#llx r12 %#llx r13 %#llx r14 %#llx r15 %#llx [rsi '%s'] [rdx '%s'] callers",
                       (unsigned long long)dr6,(unsigned long long)(c->Rip-(uintptr_t)image),(unsigned long long)c->Rax,
                       (unsigned long long)c->Rcx,(unsigned long long)c->Rdx,(unsigned long long)c->Rsi,(unsigned long long)c->Rdi,
                       (unsigned long long)c->R8,(unsigned long long)c->R9,(unsigned long long)c->Rbx,(unsigned long long)c->R12,
                       (unsigned long long)c->R13,(unsigned long long)c->R14,(unsigned long long)c->R15,s1,s2);
        const uint64_t *stack=(const uint64_t *)(uintptr_t)c->Rsp;
        for (int i=0,found=0;i<64 && found<6 && n<(int)sizeof(line)-24;++i) {
            uint64_t v=stack[i];
            if (v-(uintptr_t)image<0x2000000) { n+=snprintf(line+n,sizeof(line)-n," %#llx",(unsigned long long)(v-(uintptr_t)image)); ++found; }
        }
        puts(line); fflush(stdout);
    } else if (image && c->Rip-(uintptr_t)image>=0x2000000) {
        /* GOW3_WATCH_HOST=1: host code too (HLE library calls such as memcpy made by the game:
         * guest return addresses on the stack), as "Watch host:" lines. */
        static int host=-1;
        if (host<0) { const char *e=getenv("GOW3_WATCH_HOST"); host=e && e[0]=='1'; }
        static volatile LONG host_hits;
        if (host && InterlockedIncrement(&host_hits)<=max_hits) {
            char line[600];
            HMODULE mod=NULL; char name[MAX_PATH]="?";
            if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,(LPCSTR)(uintptr_t)c->Rip,&mod))
                GetModuleFileNameA(mod,name,sizeof(name));
            const char *base_name=strrchr(name,'\\'); base_name=base_name ? base_name+1 : name;
            int n=snprintf(line,sizeof(line),"Watch host: dr6 %#llx thread %lu rip %s+%#llx guest callers",
                           (unsigned long long)dr6,(unsigned long)GetCurrentThreadId(),base_name,
                           (unsigned long long)(c->Rip-(uintptr_t)mod));
            const uint64_t *stack=(const uint64_t *)(uintptr_t)c->Rsp;
            for (int i=0,found=0;i<512 && found<6 && n<(int)sizeof(line)-24;++i) {
                uint64_t v=stack[i];
                if (v-(uintptr_t)image<0x2000000) { n+=snprintf(line+n,sizeof(line)-n," %#llx",(unsigned long long)(v-(uintptr_t)image)); ++found; }
            }
            puts(line); fflush(stdout);
        }
    }
    return EXCEPTION_CONTINUE_EXECUTION;
#undef hits
}
/* GOW3_EXEC_WATCH=<off>[,<off>...] (hex guest image offsets, up to 4): instruction breakpoints
 * on every thread of the process (GOW3_EXEC_WATCH_SECONDS, default 600 s); a helper arms threads as they
 * appear (the guest main thread before guest code starts). Hits are logged by watch_hit. */
static uint64_t exec_watch[4];
static int exec_watch_count;
static HANDLE exec_watch_ready;
static DWORD exec_watch_armed[8192];
static volatile LONG exec_watch_armed_count;
/* Diagnostics: points data watch 0 at `address` (8-byte writes) and re-arms every thread
 * (needs GOW3_EXEC_WATCH set at start, e.g. w:0). Called by the GPU library's checks. */
void runtime_exec_watch_retarget(uint64_t address) {
    exec_watch[0]=address|(1ull<<63);
    if (exec_watch_count<1) exec_watch_count=1;
    InterlockedExchange(&exec_watch_armed_count,0);
    printf("Exec watch: data watch moved to %#llx\n",(unsigned long long)address);
    fflush(stdout);
}
/* Diagnostics: up to 4 read/write data watches (4 bytes each, absolute addresses) in place of
 * the current ones, every thread re-armed (needs GOW3_EXEC_WATCH set at start, e.g. r:0, and a
 * long GOW3_EXEC_WATCH_SECONDS). Called by the GPU library (GOW3_RESULT_WATCH). */
void runtime_exec_watch_set_reads(const uint64_t *addresses, int count) {
    if (count>4) count=4;
    for (int i=0;i<4;++i) exec_watch[i]=i<count ? (addresses[i]&~3ull)|(1ull<<62) : 0;
    exec_watch_count=count;
    InterlockedExchange(&exec_watch_armed_count,0);
}
static const char *exec_watch_file;
static void poll_exec_watch_file(void);
static DWORD WINAPI exec_watch_thread(LPVOID unused) {
    (void)unused;
    DWORD *armed=exec_watch_armed;
    DWORD self=GetCurrentThreadId(), pid=GetCurrentProcessId();
    ULONGLONG start=GetTickCount64();
    const char *secs=getenv("GOW3_EXEC_WATCH_SECONDS");
    const ULONGLONG limit=(secs && atoi(secs)>0 ? (ULONGLONG)atoi(secs) : 600)*1000;
    for (int round=0; exec_watch_file || GetTickCount64()-start<limit; ++round) {
        if (round%50==0) poll_exec_watch_file();
        HANDLE snap=CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD,0);
        if (snap!=INVALID_HANDLE_VALUE) {
            THREADENTRY32 te={.dwSize=sizeof(te)};
            for (BOOL ok=Thread32First(snap,&te); ok; ok=Thread32Next(snap,&te)) {
                if (te.th32OwnerProcessID!=pid || te.th32ThreadID==self) continue;
                int known=0;
                const LONG armed_count=exec_watch_armed_count;
                for (int i=0;i<armed_count;++i) if (armed[i]==te.th32ThreadID) { known=1; break; }
                if (known || armed_count==8192) continue;
                HANDLE t=OpenThread(THREAD_SUSPEND_RESUME|THREAD_GET_CONTEXT|THREAD_SET_CONTEXT,FALSE,te.th32ThreadID);
                if (!t) continue;
                if (SuspendThread(t)!=(DWORD)-1) {
                    CONTEXT ctx={.ContextFlags=CONTEXT_DEBUG_REGISTERS};
                    if (GetThreadContext(t,&ctx)) {
                        DWORD64 *dr[4]={&ctx.Dr0,&ctx.Dr1,&ctx.Dr2,&ctx.Dr3};
                        ctx.Dr7=0;
                        for (int i=0;i<exec_watch_count;++i) {
                            /* Bit 63 marks a data watch (w:<absolute address>): 8-byte writes. */
                            if ((exec_watch[i]>>62)==1) {
                                *dr[i]=(DWORD64)(exec_watch[i]&~(3ull<<62)&~3ull);
                                ctx.Dr7|=(1ull<<(2*i))|(3ull<<(16+4*i))|(3ull<<(18+4*i));
                            } else if (exec_watch[i]>>63) {
                                *dr[i]=(DWORD64)(exec_watch[i]&~(1ull<<63)&~7ull);
                                ctx.Dr7|=(1ull<<(2*i))|(1ull<<(16+4*i))|(2ull<<(18+4*i));
                            } else {
                                *dr[i]=(DWORD64)(uintptr_t)(image+exec_watch[i]);
                                ctx.Dr7|=1ull<<(2*i);
                            }
                        }
                        if (SetThreadContext(t,&ctx)) armed[InterlockedIncrement(&exec_watch_armed_count)-1]=te.th32ThreadID;
                    }
                    ResumeThread(t);
                }
                CloseHandle(t);
            }
            CloseHandle(snap);
        }
        if (round==0) SetEvent(exec_watch_ready);
        Sleep(round<200 ? 1 : 20);
    }
    printf("Exec watch: done (%ld threads armed)\n",(long)exec_watch_armed_count);
    return 0;
}
/* Bit 62 marks a 4-byte read/write data watch (r:<absolute address>). */
static int parse_exec_watch(const char *text, uint64_t out[4]) {
    int count=0;
    for (const char *p=text; *p && count<4;) {
        while (*p==' ' || *p=='\n' || *p=='\r') ++p;
        int data=0, read=0;
        if ((p[0]=='w' || p[0]=='W') && p[1]==':') { data=1; p+=2; }
        else if ((p[0]=='r' || p[0]=='R') && p[1]==':') { read=1; p+=2; }
        char *end=NULL;
        uint64_t v=strtoull(p,&end,16);
        if (end==p) break;
        out[count++]=data ? v|(1ull<<63) : read ? v|(1ull<<62) : v;
        p=*end==',' ? end+1 : end;
    }
    return count;
}
/* GOW3_EXEC_WATCH_FILE=<path>: the watch list (same syntax, plus r:<address>) is read from the
 * file every second while the game runs; a changed list re-arms every thread (diagnostics: data
 * watches on heap addresses found at runtime). */
static const char *exec_watch_file;
static char exec_watch_file_text[256];
static void poll_exec_watch_file(void) {
    if (!exec_watch_file) return;
    FILE *f=fopen(exec_watch_file,"rb");
    if (!f) return;
    char text[256]={0};
    size_t n=fread(text,1,sizeof(text)-1,f);
    fclose(f);
    text[n]=0;
    if (!strcmp(text,exec_watch_file_text)) return;
    strcpy(exec_watch_file_text,text);
    uint64_t list[4]={0};
    const int count=parse_exec_watch(text,list);
    for (int i=0;i<4;++i) exec_watch[i]=list[i];
    exec_watch_count=count;
    InterlockedExchange(&watch_hits,0);
    InterlockedExchange(&exec_watch_armed_count,0);
    printf("Exec watch: %d watches from %s: %s\n",count,exec_watch_file,text);
    fflush(stdout);
}
/* GOW3_POKE=<entry>[,<entry>...]: bytes the runtime keeps set in guest memory, rewritten every
 * 250 ms (cheats: the game's own debug-menu switches).
 * An entry is <offset>[=<value>] (hex guest image offset, value 1 by default) or
 * *<offset>+<add>[=<value>] (the byte <add> past the pointer stored at <offset>). Reads and
 * writes go through Read/WriteProcessMemory: an address not mapped (yet) is skipped. */
typedef struct { int deref; uint64_t offset, add; unsigned char value; } Poke;
static Poke pokes[16];
static int poke_count;
static DWORD WINAPI poke_thread(LPVOID unused) {
    (void)unused;
    const HANDLE self=GetCurrentProcess();
    for (;;) {
        for (int i=0;i<poke_count;++i) {
            uintptr_t address=(uintptr_t)image+(uintptr_t)pokes[i].offset;
            SIZE_T got=0;
            if (pokes[i].deref) {
                uint64_t pointer=0;
                if (!ReadProcessMemory(self,(LPCVOID)address,&pointer,sizeof(pointer),&got) || got!=sizeof(pointer) || !pointer) continue;
                address=(uintptr_t)pointer+(uintptr_t)pokes[i].add;
            }
            unsigned char current=0;
            if (ReadProcessMemory(self,(LPCVOID)address,&current,1,&got) && got==1 && current!=pokes[i].value)
                WriteProcessMemory(self,(LPVOID)address,&pokes[i].value,1,NULL);
        }
        Sleep(250);
    }
    return 0;
}
static void start_pokes(void) {
    const char *env=getenv("GOW3_POKE");
    if (!env || !*env || !image) return;
    const char *p=env;
    while (*p && poke_count<16) {
        Poke k={0};
        char *end;
        if (*p=='*') { k.deref=1; ++p; }
        k.offset=strtoull(p,&end,16);
        p=end;
        if (k.deref && *p=='+') { k.add=strtoull(p+1,&end,16); p=end; }
        k.value=1;
        if (*p=='=') { k.value=(unsigned char)strtoul(p+1,&end,16); p=end; }
        pokes[poke_count++]=k;
        while (*p && *p!=',') ++p;
        if (*p==',') ++p;
    }
    printf("Runtime: %d memory switches kept set (GOW3_POKE: %s)\n",poke_count,env);
    CloseHandle(CreateThread(NULL,0,poke_thread,NULL,0,NULL));
}
static void start_exec_watch(void) {
    const char *env=getenv("GOW3_EXEC_WATCH");
    exec_watch_file=getenv("GOW3_EXEC_WATCH_FILE");
    if ((!env || !*env) && !exec_watch_file) return;
    if (env) exec_watch_count=parse_exec_watch(env,exec_watch);
    exec_watch_ready=CreateEventA(NULL,TRUE,FALSE,NULL);
    CloseHandle(CreateThread(NULL,0,exec_watch_thread,NULL,0,NULL));
    WaitForSingleObject(exec_watch_ready,5000);
    printf("Exec watch: %d breakpoints armed from %s\n",exec_watch_count,env ? env : "(file)");
}
/* GitHub #22 (18 Oct 2026, Windows 10 19041, at start): the games run their job
 * code on PS4 fibers whose stacks lie in guest memory (gow3_fiber.cpp), outside the stack the
 * thread's TEB names. Resuming from any exception there, ntdll's RtlGuardRestoreContext ended the
 * process (FAST_FAIL_INVALID_SET_OF_CONTEXT: the context's rsp outside the TEB stack). The TEB now
 * names the stack the thread is on before an exception is handled: the guest allocation holding
 * rsp, or the thread's own stack again once it is back there. */
int runtime_memory_region(uintptr_t address, uintptr_t *start, uintptr_t *end, int *mapped);
static void cover_current_stack(const CONTEXT *context) {
    NT_TIB *tib=(NT_TIB *)NtCurrentTeb();
    const uintptr_t sp=(uintptr_t)context->Rsp;
    if (sp>=(uintptr_t)tib->StackLimit && sp<(uintptr_t)tib->StackBase) return;
    static _Thread_local uintptr_t own_limit, own_base;
    if (!own_base) { own_limit=(uintptr_t)tib->StackLimit; own_base=(uintptr_t)tib->StackBase; }
    if (sp>=own_limit && sp<own_base) {
        tib->StackLimit=(PVOID)own_limit; tib->StackBase=(PVOID)own_base;
        return;
    }
    /* gow3 19 Oct 2026: a stack in guest memory is bounded by its guest mapping. runtime_memory.c
     * maps large ranges as several views of at most 2 MiB, so a Windows allocation no longer spans
     * a mapping: limits taken from it cut fiber stacks that cross a view, and a game stopped
     * at start ("fiber stack overflow"). Other stacks: their Windows allocation, as before. */
    uintptr_t guest_start, guest_end;
    int guest_mapped;
    if (sp>=UINT64_C(0x1000000000) && sp<UINT64_C(0xfc00000000) &&
        runtime_memory_region(sp,&guest_start,&guest_end,&guest_mapped) && guest_mapped &&
        guest_start<=sp && sp<guest_end) {
        tib->StackLimit=(PVOID)guest_start; tib->StackBase=(PVOID)guest_end;
        return;
    }
    MEMORY_BASIC_INFORMATION region;
    if (!VirtualQuery((LPCVOID)sp,&region,sizeof(region)) || region.State!=MEM_COMMIT) return;
    uintptr_t low=(uintptr_t)region.AllocationBase, high=(uintptr_t)region.BaseAddress+region.RegionSize;
    /* The rest of the allocation above rsp (protection changes split it into regions). */
    for (int i=0;i<256;++i) {
        MEMORY_BASIC_INFORMATION next;
        if (!VirtualQuery((LPCVOID)high,&next,sizeof(next)) || next.AllocationBase!=region.AllocationBase) break;
        high=(uintptr_t)next.BaseAddress+next.RegionSize;
    }
    tib->StackLimit=(PVOID)low; tib->StackBase=(PVOID)high;
}
static LONG CALLBACK vectored_fault(EXCEPTION_POINTERS *info) {
    const EXCEPTION_RECORD *record=info->ExceptionRecord;
    DWORD code=record->ExceptionCode;
    cover_current_stack(info->ContextRecord);
    if (code==STATUS_GCC_THROW_CODE) { report_cxx_throw(info); return EXCEPTION_CONTINUE_SEARCH; }
    if (code==EXCEPTION_SINGLE_STEP && (info->ContextRecord->Dr6&0xf)) return watch_hit(info);
    if (code==EXCEPTION_ACCESS_VIOLATION && record->NumberParameters>=2) {
        /* GOW3_FAULT_TRACE=1: the first 64 access violations (diagnostics), written directly so
         * that a fault the process doesn't survive still shows: kind, address, instruction. */
        static int trace=-1;
        if (trace<0) { const char *env=getenv("GOW3_FAULT_TRACE"); trace=env && env[0]=='1' ? 64 : 0; }
        if (trace>0) {
            --trace;
            char line[256];
            uintptr_t rip=(uintptr_t)info->ContextRecord->Rip;
            HMODULE module=NULL; char name[MAX_PATH]="?";
            if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,(LPCSTR)rip,&module))
                GetModuleFileNameA(module,name,sizeof(name));
            const char *base=strrchr(name,'\\');
            int n=snprintf(line,sizeof(line),"AV %s %p rip %p (%s+%#llx, guest %#llx) thread %lu\n",
                           record->ExceptionInformation[0]==1 ? "write" : record->ExceptionInformation[0]==8 ? "exec" : "read",
                           (void *)record->ExceptionInformation[1],(void *)rip,base ? base+1 : name,
                           module ? (unsigned long long)(rip-(uintptr_t)module) : 0ULL,
                           image && rip-(uintptr_t)image<0x20000000 ? (unsigned long long)(rip-(uintptr_t)image) : 0ULL,
                           GetCurrentThreadId());
            DWORD written; WriteFile(GetStdHandle(STD_OUTPUT_HANDLE),line,(DWORD)n,&written,NULL);
        }
        /* GPU page tracking (write-protected guest pages) is resolved first, except during a
         * speculative read (draw preparation workers): resolving a GPU-modified page there waits
         * for the GPU thread, which may be waiting for a lock that worker holds (seen in a
         * level load). Such a fault takes the recovery path; the GPU thread does that draw. */
        if (gpu_enabled && !runtime_fault_recover) {
            /* A fault inside the GPU fault handlers is legitimate a level or two deep (a handler
             * touching another tracked page). A bug there recursed until the thread's stack ran
             * out and the game froze for good (18 Oct 2026: the page table indexed
             * with a garbage address, thousands of levels). Deeper than 8: a report, and stop. */
            static _Thread_local int handler_depth;
            if (handler_depth>=8) {
                fprintf(stderr,"Fault handler: faulted inside itself %d levels deep\n",handler_depth);
                return fatal_exception(info);
            }
            ++handler_depth;
            const int handled=gow3gpu_handle_fault(info,(void *)record->ExceptionInformation[1]);
            --handler_depth;
            if (handled) {
                note_red_zone_fault((uintptr_t)info->ContextRecord->Rip);
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
        if (runtime_fault_recover) {
            /* A speculative guest memory read (GPU draw preparation) faulted: the thread resumes
             * in gow3_longjmp, outside the exception dispatcher, back at its recovery point. */
            unsigned long long *buffer=*runtime_fault_recover;
            runtime_fault_recover=NULL;
            CONTEXT *c=info->ContextRecord;
            c->Rsp=((c->Rsp-128)&~(DWORD64)15)-40;
            c->Rcx=(DWORD64)(uintptr_t)buffer;
            c->Rdx=1;
            c->Rip=(DWORD64)(uintptr_t)gow3_longjmp;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }
    if (code==EXCEPTION_ACCESS_VIOLATION || code==EXCEPTION_ILLEGAL_INSTRUCTION || code==EXCEPTION_PRIV_INSTRUCTION ||
        code==EXCEPTION_STACK_OVERFLOW || code==EXCEPTION_INT_DIVIDE_BY_ZERO || code==EXCEPTION_DATATYPE_MISALIGNMENT) {
        /* Faults in guest code, the loader or the runtime end the process with a report; drivers
         * and system libraries may handle their own exceptions. */
        uintptr_t rip=(uintptr_t)info->ContextRecord->Rip;
        HMODULE self=GetModuleHandleA(NULL), module=NULL;
        int ours=(image && rip-(uintptr_t)image<0x20000000) ||
                 (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,(LPCSTR)rip,&module) && module==self) ||
                 !module;
        if (ours) return fatal_exception(info);
    }
    /* GOW3_STRICT_HANDLES=1: a closed or invalid handle used anywhere is reported (diagnostics). */
    if (code==0xC0000008 /* STATUS_INVALID_HANDLE */) return fatal_exception(info);
    return EXCEPTION_CONTINUE_SEARCH;
}
static LONG WINAPI unhandled_fault(EXCEPTION_POINTERS *info) { return fatal_exception(info); }
/* Guest code reads its TCB with `mov rax, gs:[0]` (link_*.py rewrote fs:[0]); on Windows
 * the displacement becomes the TEB TLS slot that holds the guest TCB (runtime_thread.c). */
uint32_t runtime_thread_tcb_offset(void);
static uint64_t patch_tcb_loads(const Segment *segments, uint64_t count) {
    static const unsigned char load[]={0x65,0x48,0x8b,0x04,0x25,0,0,0,0};
    uint32_t displacement=runtime_thread_tcb_offset();
    uint64_t patched=0;
    for (uint64_t s=0;s<count;++s) {
        if (!(segments[s].flags&1) || segments[s].size<sizeof(load)) continue;
        unsigned char *at=image+segments[s].address, *end=at+segments[s].size-sizeof(load);
        for (; at<=end; ++at)
            if (at[0]==0x65 && !memcmp(at,load,sizeof(load))) { memcpy(at+5,&displacement,4); ++patched; at+=sizeof(load)-1; }
    }
    return patched;
}
#endif
#ifndef _WIN32
/* gow3 (Linux): the Windows CRT (ucrtbase) defines __ImageBase; the GPU library uses
 * &__ImageBase as an anchor inside the loader image (buffer_cache, vk_scheduler). */
char __ImageBase;
/* gow3 (Linux): instruction/data watches use the Win32 debug registers and exception
 * handlers (the Windows-only block above); the GPU library's diagnostic hooks (GOW3_SRT_WATCH,
 * GOW3_TRACE_WATCH_NODE, GOW3_RESULT_WATCH) are no-ops here. */
void runtime_exec_watch_retarget(uint64_t address) {
    static int noted;
    if (!noted++) printf("Exec watch: Windows-only diagnostics (debug registers), ignored on POSIX\n");
    (void)address;
}
void runtime_exec_watch_set_reads(const uint64_t *addresses, int count) {
    static int noted;
    if (!noted++) printf("Exec watch: Windows-only diagnostics (debug registers), ignored on POSIX\n");
    (void)addresses; (void)count;
}
#endif
#ifndef _WIN32
static void fault(int sig, siginfo_t *info, void *context) {
    ucontext_t *uc = context;
    const uintptr_t rip = (uintptr_t)uc->uc_mcontext.gregs[REG_RIP];
    /* GPU page tracking (write-protected guest pages) is resolved first, except during a
     * speculative read: see vectored_fault. (Linux puts its signal frame below the 128-byte red
     * zone already; the red-zone table is loaded here for its ISA emulation, MOVBE/AVX2/F16C on
     * CPUs without them, and the stats count its stubs the same way.) */
    if (gpu_enabled && sig == SIGSEGV && !runtime_fault_recover) {
        if (gow3gpu_handle_fault(context, info->si_addr)) {
            note_red_zone_fault(rip);
            return;
        }
    }
    /* A speculative guest memory read (runtime_memory.c) failed: resume its recovery point. */
    if ((sig == SIGSEGV || sig == SIGBUS) && runtime_fault_recover) {
        sigjmp_buf *recover = runtime_fault_recover;
        runtime_fault_recover = NULL;
        sigset_t unblock;
        sigemptyset(&unblock);
        sigaddset(&unblock, sig);
        pthread_sigmask(SIG_UNBLOCK, &unblock, NULL);
        siglongjmp(*recover, 1);
    }
    /* The process is terminating: dladdr/snprintf are acceptable here. */
    char line[512];
    Dl_info where;
    uint64_t site = 0;
    if (red_zone && red_zone_site_of_stub(red_zone, rip, &site))
        snprintf(line, sizeof(line), "Guest fault (signal %d) at guest offset 0x%llx (red-zone stub), address %p\n",
                 sig, (unsigned long long)site, info->si_addr);
    else if (rip - (uintptr_t)image < 0x10000000)
        snprintf(line, sizeof(line), "Guest fault (signal %d) at guest offset 0x%lx, address %p\n",
                 sig, (unsigned long)(rip - (uintptr_t)image), info->si_addr);
    else if (dladdr((void *)rip, &where) && where.dli_fname)
        snprintf(line, sizeof(line), "Host fault (signal %d) in %s+0x%lx (%s), address %p\n", sig, where.dli_fname,
                 (unsigned long)(rip - (uintptr_t)where.dli_fbase), where.dli_sname ? where.dli_sname : "?", info->si_addr);
    else
        snprintf(line, sizeof(line), "Fault (signal %d) at RIP %p, address %p\n", sig, (void *)rip, info->si_addr);
    { ssize_t written_=write(2, line, strlen(line)); (void)written_; }
    if (gpu_enabled) gow3gpu_dump_guest_writes(context);
    /* Host call chain (frames with unwind info; guest frames end it). */
    void *frames[32];
    int depth = backtrace(frames, 32);
    for (int i = 2; i < depth; ++i) {
        if (dladdr(frames[i], &where) && where.dli_fname)
            snprintf(line, sizeof(line), "  #%d %s+0x%lx (%s)\n", i, where.dli_fname,
                     (unsigned long)((uintptr_t)frames[i] - (uintptr_t)where.dli_fbase), where.dli_sname ? where.dli_sname : "?");
        else
            snprintf(line, sizeof(line), "  #%d %p\n", i, frames[i]);
        ssize_t written_=write(2, line, strlen(line)); (void)written_;
    }
    _exit(128 + sig);
}
#endif
#ifndef _WIN32
/* Watchdog: dump RIP and the rbp frame chain of every thread (guest offsets
 * when inside the image). Reads use process_vm_readv so bad frames cannot fault. */
static uintptr_t exe_base;
static void write_hex(char *out, uint64_t v) {
    const char digits[] = "0123456789abcdef";
    for (int i = 0; i < 16; ++i) out[i] = digits[(v >> (60 - i * 4)) & 15];
}
static void dump_frames(ucontext_t *uc) {
    char line[] = "  tid=0000000000000000 rip=0000000000000000 image-relative=0000000000000000 host-relative=0000000000000000\n";
    uintptr_t rip=(uintptr_t)uc->uc_mcontext.gregs[REG_RIP], rbp=(uintptr_t)uc->uc_mcontext.gregs[REG_RBP];
    uint64_t tid=(uint64_t)gettid();
    /* First argument register: the lock address when a thread waits on a futex. */
    char arg[]="  tid=0000000000000000 rdi=0000000000000000\n";
    write_hex(arg+6,tid); write_hex(arg+27,(uint64_t)uc->uc_mcontext.gregs[REG_RDI]);
    { ssize_t written_=write(2,arg,sizeof(arg)-1); (void)written_; }
    for (int depth=0; depth<24; ++depth) {
        write_hex(line+6,tid); write_hex(line+27,rip); write_hex(line+59,rip-(uintptr_t)image); write_hex(line+90,rip-exe_base);
        { ssize_t written_=write(2,line,sizeof(line)-1); (void)written_; }
        uintptr_t frame[2];
        struct iovec local={frame,sizeof(frame)}, remote={(void *)rbp,sizeof(frame)};
        if (!rbp || process_vm_readv(getpid(),&local,1,&remote,1,0)!=(ssize_t)sizeof(frame)) break;
        if (frame[0]<=rbp) break;
        rbp=frame[0]; rip=frame[1];
    }
}
static void thread_dump(int sig, siginfo_t *info, void *context) { (void)sig; (void)info; dump_frames(context); }
static void watchdog(int sig, siginfo_t *info, void *context) {
    (void)info;
    const char head[]="STOP: watchdog timeout; thread stacks:\n";
    { ssize_t written_=write(2,head,sizeof(head)-1); (void)written_; }
    dump_frames(context);
    int dir=open("/proc/self/task",O_RDONLY|O_DIRECTORY);
    char buffer[4096];
    long n;
    pid_t self=gettid();
    while (dir>=0 && (n=syscall(SYS_getdents64,dir,buffer,sizeof(buffer)))>0)
        for (long at=0; at<n;) {
            struct { uint64_t ino; int64_t off; unsigned short reclen; unsigned char type; char name[]; } *d=(void *)(buffer+at);
            pid_t tid=(pid_t)strtol(d->name,NULL,10);
            if (tid>0 && tid!=self) { syscall(SYS_tgkill,getpid(),tid,SIGUSR2); usleep(20000); }
            at+=d->reclen;
        }
    usleep(100000);
    _exit(128 + sig);
}
#endif
/* param.sfo lookup: string or integer value of key, 0 when absent. */
static int sfo_value(const char *path, const char *key, char *text, size_t text_size, uint32_t *number) {
    FILE *f=fopen(path,"rb");
    if (!f) return 0;
    unsigned char data[65536];
    size_t n=fread(data,1,sizeof(data),f); fclose(f);
    if (n<20 || memcmp(data,"\0PSF",4)) return 0;
    uint32_t keys, values, count;
    memcpy(&keys,data+8,4); memcpy(&values,data+12,4); memcpy(&count,data+16,4);
    for (uint32_t i=0;i<count && 20+i*16+16<=n;++i) {
        const unsigned char *e=data+20+i*16;
        uint16_t key_offset, format; uint32_t length, offset;
        memcpy(&key_offset,e,2); memcpy(&format,e+2,2); memcpy(&length,e+4,4); memcpy(&offset,e+12,4);
        if (keys+key_offset>=n || values+offset+length>n || strcmp((const char *)data+keys+key_offset,key)) continue;
        if (format==0x0404 && number && length>=4) { memcpy(number,data+values+offset,4); return 1; }
        if (text && text_size) {
            size_t copy=length<text_size-1 ? length : text_size-1;
            memcpy(text,data+values+offset,copy); text[copy]=0;
        }
        return 1;
    }
    return 0;
}
static int mapped(Segment *segments, uint64_t count, uint64_t address, uint64_t bytes) {
    for (uint64_t i = 0; i < count; ++i)
        if (address >= segments[i].address && bytes <= segments[i].size &&
            address - segments[i].address <= segments[i].size - bytes) return 1;
    return 0;
}
extern ABI float gow3_orbs_gain(void *player, float requested);
extern void gow3_orbs_status(int supported);
extern uintptr_t gow3_cheat_flag(unsigned id);
extern uintptr_t gow3_actor_slots_address(void);
extern void gow3_actor_status(int supported);
extern void gow3_cheat_status(unsigned id);
extern uintptr_t gow3_damage_multiplier(int taken);
extern uintptr_t gow3_orb_pickup_multiplier(unsigned kind);
extern void gow3_stat_multipliers_status(int damage, int orb_pickup);

/* The hooks below find their sites by signature (src/signature.h), so they work in any region
 * and version of the game whose code matches; a site that is not found leaves its feature off. */
static unsigned char *allocate_near_image(uint64_t size);
static void install_cheats(Segment *segments, uint64_t count, uint64_t size) {
    const char *names[] = {"health", "magic", "item meter", "Rage of Sparta", "red orbs",
                           "Very Hard unlock", "infinite double jumps"};
    uint64_t at[GOW3_CHEAT_HOOK_COUNT];
    for (unsigned i = 0; i < GOW3_CHEAT_HOOK_COUNT; ++i)
        at[i] = gow3_signature_find(image, size, &gow3_cheat_hooks[i].site);
    for (unsigned id = 0; id < GOW3_CHEAT_COUNT; ++id) {
        int valid = 1;
        for (unsigned i = 0; i < GOW3_CHEAT_HOOK_COUNT; ++i) {
            const Gow3CheatHook *hook = &gow3_cheat_hooks[i];
            if (hook->id != id) continue;
            if (at[i] == UINT64_MAX || !mapped(segments, count, at[i], hook->length) ||
                !gow3_cheat_signature_match(hook, image + at[i], size - at[i])) valid = 0;
        }
        if (!valid) {
            printf("Cheats: %s unavailable (code not found in this eboot.bin)\n", names[id]);
            continue;
        }
        unsigned char *stubs = allocate_near_image(size);
        if (!stubs) {
            printf("Cheats: %s unavailable (no stub page in range)\n", names[id]);
            continue;
        }
        int32_t relative[GOW3_CHEAT_HOOK_COUNT] = {0};
        for (unsigned i = 0; i < GOW3_CHEAT_HOOK_COUNT; ++i) {
            const Gow3CheatHook *hook = &gow3_cheat_hooks[i];
            if (hook->id != id) continue;
            unsigned char *stub = stubs + i * 256;
            int64_t delta = (int64_t)(uintptr_t)stub - (int64_t)(uintptr_t)(image + at[i] + 5);
            if (delta < INT32_MIN || delta > INT32_MAX) { valid = 0; break; }
            relative[i] = (int32_t)delta;
            gow3_cheat_build_stub(stub, hook, gow3_cheat_flag(id),
                                 (uintptr_t)(image + at[i] + hook->length));
        }
        if (!valid) {
            printf("Cheats: %s unavailable (hook outside relative-jump range)\n", names[id]);
            continue;
        }
        protect(stubs, page_size, 5);
        for (unsigned i = 0; i < GOW3_CHEAT_HOOK_COUNT; ++i) {
            const Gow3CheatHook *hook = &gow3_cheat_hooks[i];
            if (hook->id != id) continue;
            image[at[i]] = 0xe9;
            memcpy(image + at[i] + 1, &relative[i], 4);
            memset(image + at[i] + 5, 0x90, hook->length - 5);
        }
        gow3_cheat_status(id);
        printf("Cheats: %s ready (live toggle, initially disabled unless saved)\n", names[id]);
    }
}

/* Damage hook for the enemy health bar (src/actor_hook.h). Always
 * installed when the code matches: it only records values; the overlay decides what to show. */
static void install_actor_watch(Segment *segments, uint64_t count, uint64_t size) {
    const uint64_t at = gow3_signature_find(image, size, &gow3_actor_site);
    if (at == UINT64_MAX || !mapped(segments, count, at, GOW3_ACTOR_HOOK_LENGTH) ||
        !gow3_actor_signature_match(image + at, size - at)) {
        printf("Health display: unavailable (code not found in this eboot.bin)\n");
        return;
    }
    unsigned char *stub = allocate_near_image(size);
    if (!stub) {
        printf("Health display: unavailable (no stub page in range)\n");
        return;
    }
    const int64_t delta = (int64_t)(uintptr_t)stub - (int64_t)(uintptr_t)(image + at + 5);
    if (delta < INT32_MIN || delta > INT32_MAX) {
        printf("Health display: unavailable (hook outside relative-jump range)\n");
        return;
    }
    gow3_actor_build_stub(stub, gow3_actor_slots_address(), (uintptr_t)(image + at + GOW3_ACTOR_HOOK_LENGTH));
    protect(stub, page_size, 5);
    const int32_t relative = (int32_t)delta;
    image[at] = 0xe9;
    memcpy(image + at + 1, &relative, 4);
    memset(image + at + 5, 0x90, GOW3_ACTOR_HOOK_LENGTH - 5);
    gow3_actor_status(1);
    printf("Health display: damage hook ready (enemy health bar in the menu)\n");
}

static void install_orb_multiplier(Segment *segments, uint64_t count, uint64_t size) {
    uint64_t calls[3];
    const uint64_t setter_at = gow3_orb_find(image, size, calls);
    if (setter_at == UINT64_MAX) {
        printf("Red orbs: multiplier unavailable (code not found in this eboot.bin)\n");
        return;
    }
    for (unsigned i = 0; i < 3; ++i) {
        if (!mapped(segments, count, calls[i], 5)) return;
    }
    /* SysV call sites: preserve the receiver across the helper, then tail-call the original
     * setter. Its cap and upgrade notifications remain in the game's own code. */
    unsigned char *stub = allocate_near_image(size);
    if (!stub) {
        printf("Red orbs: multiplier unavailable (no stub page in range)\n");
        return;
    }
    uintptr_t helper = (uintptr_t)gow3_orbs_gain, setter = (uintptr_t)(image + setter_at);
    unsigned char code[26];
    gow3_orb_build_stub(code, helper, setter);
    int32_t displacements[3];
    for (unsigned i = 0; i < 3; ++i) {
        int64_t delta = (int64_t)(uintptr_t)stub - (int64_t)(uintptr_t)(image + calls[i] + 5);
        if (delta < INT32_MIN || delta > INT32_MAX) {
            printf("Red orbs: multiplier unavailable (hook outside relative-call range)\n");
            return;
        }
        displacements[i] = (int32_t)delta;
    }
    memcpy(stub, code, sizeof(code));
    protect(stub, page_size, 5);
    for (unsigned i = 0; i < 3; ++i)
        memcpy(image + calls[i] + 1, &displacements[i], 4);
    gow3_orbs_status(1);
    printf("Red orbs: live multiplier ready (3 validated gain paths, 0.1x to 100x)\n");
}
/* Writes a jump from image+at to `stub`; the rest of the replaced bytes become nops (the stub
 * replays them). 0 when the stub is out of relative-jump range. */
static int install_jump(uint64_t at, unsigned length, unsigned char *stub) {
    const int64_t delta = (int64_t)(uintptr_t)stub - (int64_t)(uintptr_t)(image + at + 5);
    if (delta < INT32_MIN || delta > INT32_MAX) return 0;
    const int32_t relative = (int32_t)delta;
    image[at] = 0xe9;
    memcpy(image + at + 1, &relative, 4);
    memset(image + at + 5, 0x90, length - 5);
    return 1;
}

/* The persistent Molotov point light uses a fixed linear RGB ratio while its
 * intensity flickers. Its pool of fire has a larger radius than the flame on
 * the bottle. Burning barrels use a separate flickering point light above
 * the rim and a short-range fill inside the barrel. */
static int trace_fire_shadow_selection;
static int trace_dynamic_barrels_only;
static int anchor_views_installed;
static _Atomic uint32_t fire_shadow_atlas_assignments;

static int is_barrel_palette(const float color[3]) {
    return color[0] > 0.0f && color[2] == 0.0f &&
           color[1] >= color[0] * 0.49f && color[1] <= color[0] * 0.65f;
}

/* A barrel's enlarged point light can reach the view after the original
 * emitter has been culled. Let the point-light builder test the new bounds. */
static ABI int barrel_needs_view_test(const unsigned char *source) {
    if (!gow3gpu_pc_setting(GOW3_PC_FIRE_SHADOWS) || source[0x10d] != 1) return 0;
    float color[3], radius;
    memcpy(color, source + 0xc8, sizeof(color));
    memcpy(&radius, source + 0xc4, sizeof(radius));
    return radius >= 0.8f && is_barrel_palette(color);
}

static int install_barrel_view_test(unsigned char *stub, Segment *segments, uint64_t count) {
    const uint64_t at = 0xA97020;
    static const unsigned char original[] = {0x41,0x8b,0x47,0x0c,0x44,0x0f,0xa3,0xe8,0x73,0x34};
    if (!mapped(segments, count, at, sizeof(original)) ||
        memcmp(image + at, original, sizeof(original))) return 0;

    static const unsigned char code[] = {
        0x41,0x8b,0x47,0x0c, 0x44,0x0f,0xa3,0xe8, /* original view mask */
        0x72,0x17,                               /* jc submit */
        0x56, 0x48,0x83,0xec,0x08,              /* save rsi, align stack */
        0x4c,0x89,0xff,                         /* mov rdi, r15 */
        0x48,0xb8,0,0,0,0,0,0,0,0,             /* movabs rax, classifier */
        0xff,0xd0,                               /* call rax */
        0x48,0x83,0xc4,0x08, 0x5e,              /* restore stack and rsi */
        0x85,0xc0, 0x74,0x0e,                  /* test eax, eax; jz skip */
        0xff,0x25,0,0,0,0, 0,0,0,0,0,0,0,0,    /* submit */
        0xff,0x25,0,0,0,0, 0,0,0,0,0,0,0,0,    /* skip */
    };
    const uintptr_t classify = (uintptr_t)barrel_needs_view_test;
    const uintptr_t submit = (uintptr_t)(image + at + sizeof(original));
    const uintptr_t skip = (uintptr_t)(image + 0xA9705E);
    memcpy(stub, code, sizeof(code));
    /* The original mask's taken path goes directly to the submit relay. */
    stub[9] = 0x1d;
    memcpy(stub + 20, &classify, sizeof(classify));
    memcpy(stub + 45, &submit, sizeof(submit));
    memcpy(stub + 59, &skip, sizeof(skip));
    return install_jump(at, sizeof(original), stub);
}

static void configure_lantern_shadow(unsigned char *point, uint32_t flags) {
    static const float color[] = {7.5f, 7.2894f, 4.9725f};
    static const float radius_and_intensity[] = {4.5f, 0.60f};
    static const float shadow_clip[] = {0.0f, 0.0f};
    static const float shadow_fade[] = {16.0f, 24.0f};
    flags |= UINT32_C(0x203);
    memcpy(point + 0x10, color, sizeof(color));
    memcpy(point + 0x30, radius_and_intensity, sizeof(radius_and_intensity));
    memcpy(point + 0x3c, &flags, sizeof(flags));
    memcpy(point + 0x44, shadow_clip, sizeof(shadow_clip));
    memcpy(point + 0x54, shadow_fade, sizeof(shadow_fade));
}

/* Exterior emitter positions measured in the nighttime escape scene. */
static const float escape_barrel_emitters[][3] = {
    {-300.644f, 0.093f, 270.697f},
    {-299.792f, 0.093f, 263.561f},
    {-273.310f, 0.207f, 271.983f},
    {-253.963f, 0.201f, 279.681f},
};

static int is_escape_barrel_emitter(const float origin[4]) {
    for (unsigned i = 0; i < sizeof(escape_barrel_emitters) / sizeof(*escape_barrel_emitters); ++i) {
        const float *known = escape_barrel_emitters[i];
        if (fabsf(origin[0] - known[0]) < 0.03f && fabsf(origin[1] - known[1]) < 0.03f &&
            fabsf(origin[2] - known[2]) < 0.03f) return 1;
    }
    return 0;
}

/* Gather live barrel emitters while the scene builds its light list.
 * Keep nearby emitters across camera turns, even when the game's smaller
 * authored volume temporarily stops contributing to the view. */
static _Thread_local struct {
    float positions[64][3];
    unsigned count;
} nearby_barrels;

static float light_distance_squared(const float a[3], const float b[3]) {
    const float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

static int register_barrel_for_view(const float origin[4]) {
    if (!anchor_views_installed) return 0;
    if (!trace_dynamic_barrels_only && is_escape_barrel_emitter(origin)) return 1;
    for (unsigned i = 0; i < nearby_barrels.count; ++i) {
        if (light_distance_squared(nearby_barrels.positions[i], origin) < 0.03f * 0.03f) return 1;
    }
    if (nearby_barrels.count == sizeof(nearby_barrels.positions) / sizeof(*nearby_barrels.positions)) {
        return 0;
    }
    memcpy(nearby_barrels.positions[nearby_barrels.count++], origin, 3 * sizeof(float));
    return 1;
}

static ABI void submit_fire_point_light(void *lights, void *view, const unsigned char *point, int mask) {
    typedef void (ABI *SubmitPointLight)(void *, void *, const unsigned char *, int);
    SubmitPointLight submit = (SubmitPointLight)(image + 0xA638B0);
    if (gow3gpu_pc_setting(GOW3_PC_FIRE_SHADOWS)) {
        float color[3], radius;
        uint32_t flags;
        memcpy(color, point + 0x10, sizeof(color));
        memcpy(&radius, point + 0x30, sizeof(radius));
        memcpy(&flags, point + 0x3c, sizeof(flags));
        const uint32_t light_type = flags & UINT32_C(0x00ffffff);
        /* The upper flag byte can change while the same emitter stays active. */
        /* The thrown bottle stays below a one-metre radius in the trace. Leave
         * room for the fire pool's radius and normalized color to vary. */
        const int molotov = light_type == 3 && radius > 1.5f && color[0] > 0.0f &&
            fabsf(color[1] - color[0] * 0.55068f) < color[0] * 0.003f &&
            fabsf(color[2] - color[0] * 0.3260f) < color[0] * 0.003f;
        /* The barrel's exterior light has no blue channel. */
        const int barrel_palette = is_barrel_palette(color);
        const int barrel = barrel_palette && light_type == 1 && radius >= 0.8f;
        const int barrel_fill = barrel_palette && light_type == 3 &&
            radius >= 0.30f && radius <= 0.55f;
        /* Tommy's Dam uses two authored palettes for the same kerosene prop. */
        const int kerosene_lantern = light_type == 1 && radius == 2.0f &&
            ((fabsf(color[0] - 1.3f) < 0.0001f && fabsf(color[1] - 0.8717f) < 0.0001f &&
              fabsf(color[2] - 0.4f) < 0.0001f) ||
             (fabsf(color[0] - 1.0f) < 0.0001f && fabsf(color[1] - 0.758f) < 0.0001f &&
              fabsf(color[2] - 0.5f) < 0.0001f));
        const int promote = molotov || barrel || kerosene_lantern;
        if (trace_fire_shadow_selection && color[0] > 0.0f &&
            color[1] > color[0] * 0.3f && color[1] < color[0] * 0.99f &&
            color[2] < color[0] * 0.78f) {
            float position[3];
            memcpy(position, point, sizeof(position));
            fprintf(stderr, "FireShadow input selected=%d flags=%08x view=%p mask=%x pos=%.3f %.3f %.3f rgb=%.6f %.6f %.6f radius=%.4f atlas=%u\n",
                    promote, flags, view, (unsigned)mask, position[0], position[1], position[2],
                    color[0], color[1], color[2], radius,
                    atomic_load_explicit(&fire_shadow_atlas_assignments, memory_order_relaxed));
        }
        if (promote || barrel_fill) {
            _Alignas(16) unsigned char adjusted[0xa0];
            memcpy(adjusted, point, sizeof(adjusted));
            if (barrel_fill) {
                /* The unshadowed glow can wash out the exterior light's
                 * shadows around the rim. Keep a small glow for the embers. */
                float intensity;
                memcpy(&intensity, point + 0x34, sizeof(intensity));
                intensity *= 0.25f;
                memcpy(adjusted + 0x34, &intensity, sizeof(intensity));
                submit(lights, view, adjusted, mask);
                return;
            }
            if (barrel) {
                /* Use the barrel emitter's world position with the same
                 * descriptor as a static oil lamp. The authored emitter's
                 * auxiliary visibility data describes its much smaller light. */
                float origin[4];
                memcpy(origin, point, sizeof(origin));
                /* These scene lights are inserted in the per-view list below. */
                if (register_barrel_for_view(origin)) return;
                origin[1] += 0.45f;
                memset(adjusted, 0, sizeof(adjusted));
                memcpy(adjusted, origin, sizeof(origin));
                configure_lantern_shadow(adjusted, 0);
                submit(lights, view, adjusted, mask);
                return;
            }
            if (kerosene_lantern) {
                configure_lantern_shadow(adjusted, flags);
                submit(lights, view, adjusted, mask);
                return;
            }
            /* Use a fixed view-distance fade for the shadow map. */
            const float shadow_fade[2] = {16.0f, 24.0f};
            memcpy(adjusted + 0x54, shadow_fade, sizeof(shadow_fade));
            flags |= UINT32_C(0x200); /* The point-light builder's omnidirectional shadow mode. */
            memcpy(adjusted + 0x3c, &flags, sizeof(flags));
            submit(lights, view, adjusted, mask);
            return;
        }
    }
    submit(lights, view, point, mask);
}

static ABI void *submit_anchored_shadows_for_view(void *view) {
    typedef void *(ABI *GetLights)(void *);
    typedef const unsigned char *(ABI *GetView)(void *);
    typedef void (ABI *SubmitPointLight)(void *, void *, const unsigned char *, int);
    void *lights = ((GetLights)(image + 0xABB420))(view);
    if (!lights || !gow3gpu_pc_setting(GOW3_PC_FIRE_SHADOWS)) {
        nearby_barrels.count = 0;
        return lights;
    }

    static const float placements[][3] = {
        {-228.74730f, 10.61145f, 100.39883f},
        {-144.90437f, 18.40289f, 14.48217f},
        {-14.77375f, 14.00229f, 135.57812f},
        {-14.76446f, 14.00497f, 147.36974f},
        {-14.60017f, 14.00497f, 159.20061f},
        {-14.13782f, 14.00100f, 108.41077f},
        /* Tommy's Dam kerosene placements without a matching authored point light.
         * The live emitters at the other lanterns are handled by the classifier. */
        {-3.89210f,  6.31069f,  -4.48367f}, /* pump room, lantern 5 */
        {-74.06360f, 4.62890f, -27.98560f}, /* turbine hall, lantern 8 */
        {-3.99498f, 14.24042f, 103.55309f},
        {12.35914f, 14.00100f, 97.76421f},
        {15.77726f, 14.00562f, 102.62310f},
        {17.22167f, 14.00562f, 100.24822f},
        {53.16716f, 13.53120f, 231.87029f},
    };
    const unsigned char *camera = ((GetView)(image + 0xABB180))(view);
    float eye[3];
    memcpy(eye, camera + 0x230, sizeof(eye));
    SubmitPointLight submit = (SubmitPointLight)(image + 0xA638B0);
    /* Drop positions from streamed-out areas. A camera turn without
     * translation keeps its nearby lights in the same list. */
    unsigned kept = 0;
    for (unsigned i = 0; i < nearby_barrels.count; ++i) {
        if (light_distance_squared(nearby_barrels.positions[i], eye) < 32.0f * 32.0f) {
            if (kept != i) memcpy(nearby_barrels.positions[kept], nearby_barrels.positions[i],
                                  sizeof(nearby_barrels.positions[0]));
            ++kept;
        }
    }
    nearby_barrels.count = kept;
    const struct {
        const float (*positions)[3];
        unsigned count;
        float height;
    } groups[] = {
        {placements, sizeof(placements) / sizeof(*placements), 0.22f},
        {escape_barrel_emitters, trace_dynamic_barrels_only ? 0 : sizeof(escape_barrel_emitters) / sizeof(*escape_barrel_emitters), 0.45f},
        {nearby_barrels.positions, nearby_barrels.count, 0.45f},
    };
    for (unsigned group = 0; group < sizeof(groups) / sizeof(*groups); ++group) {
        for (unsigned i = 0; i < groups[group].count; ++i) {
            const float *position = groups[group].positions[i];
            if (light_distance_squared(position, eye) >= 24.0f * 24.0f) continue;
            _Alignas(16) unsigned char point[0xa0] = {0};
            const float origin[] = {position[0], position[1] + groups[group].height, position[2], 1.0f};
            memcpy(point, origin, sizeof(origin));
            configure_lantern_shadow(point, 0);
            uint32_t before = 0, after;
            if (group != 0 && trace_fire_shadow_selection) memcpy(&before, lights, sizeof(before));
            submit(lights, view, point, 0);
            if (group != 0 && trace_fire_shadow_selection) {
                memcpy(&after, lights, sizeof(after));
                fprintf(stderr, "Barrel anchor %u:%u lights=%u->%u eye=%.2f %.2f %.2f\n",
                        group, i + 1, before, after, eye[0], eye[1], eye[2]);
            }
        }
    }
    return lights;
}

static ABI void submit_workbench_spotlight(void *lights, void *view,
                                           const unsigned char *spot, int mask) {
    typedef void (ABI *SubmitSpotlight)(void *, void *, const unsigned char *, int);
    SubmitSpotlight submit = (SubmitSpotlight)(image + 0xA63FF0);
    if (gow3gpu_pc_setting(GOW3_PC_FIRE_SHADOWS)) {
        float color[3], projector[5];
        uint32_t flags;
        memcpy(color, spot + 0x10, sizeof(color));
        memcpy(projector, spot + 0x40, sizeof(projector));
        memcpy(&flags, spot + 0x54, sizeof(flags));
        const int workbench = (flags & UINT32_C(0x00ffffff)) == 1 &&
            fabsf(color[0] - 7.5f) < 0.001f && fabsf(color[1] - 7.289377f) < 0.001f &&
            fabsf(color[2] - 4.9725f) < 0.001f && fabsf(projector[0] - 1.5f) < 0.001f &&
            fabsf(projector[3] - 1.04720f) < 0.001f && fabsf(projector[4] - 0.01745f) < 0.001f;
        if (workbench) {
            _Alignas(16) unsigned char adjusted[0xc0];
            memcpy(adjusted, spot, sizeof(adjusted));
            const float range = 2.6f;
            const float intensity = projector[1] * 2.05f;
            const float maximum_clip = 0.0f;
            const float fade[] = {16.0f, 24.0f, 16.0f, 24.0f};
            flags |= UINT32_C(0x100);
            memcpy(adjusted + 0x40, &range, sizeof(range));
            memcpy(adjusted + 0x44, &intensity, sizeof(intensity));
            memcpy(adjusted + 0x54, &flags, sizeof(flags));
            memcpy(adjusted + 0x60, &maximum_clip, sizeof(maximum_clip));
            memcpy(adjusted + 0x6c, fade, sizeof(fade));
            submit(lights, view, adjusted, mask);
            return;
        }
    }
    submit(lights, view, spot, mask);
}

/* A page for hook stubs within relative-jump range of the image (the image sits below 2 GiB with
 * --low-image; a page anywhere may be out of range). */
static unsigned char *allocate_near_image(uint64_t size) {
#ifdef _WIN32
    /* Free regions within +-1.5 GiB of the image (below it too: the linked modules and runtime
     * reservations can fill the space right above it). */
    const uintptr_t lo = (uintptr_t)image > UINT64_C(0x60000000) ? (uintptr_t)image - UINT64_C(0x60000000) : 0x10000;
    const uintptr_t hi = (uintptr_t)image + (uintptr_t)size + UINT64_C(0x60000000);
    MEMORY_BASIC_INFORMATION info;
    for (uintptr_t address = lo; address < hi && VirtualQuery((void *)address, &info, sizeof(info));
         address = (uintptr_t)info.BaseAddress + info.RegionSize) {
        if (info.State != MEM_FREE) continue;
        const uintptr_t start = ((uintptr_t)info.BaseAddress + 0xffff) & ~(uintptr_t)0xffff;
        if (start + page_size > (uintptr_t)info.BaseAddress + info.RegionSize) continue;
        void *p = VirtualAlloc((void *)start, page_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (p) return p;
    }
    return NULL;
#else
    /* Free gaps within +-1.5 GiB of the image, as on Windows. The guest's own memory lives at
     * 32 GiB and up (LOW_MIN/USER_MIN in runtime_memory.c), so the low 2 GiB around the image
     * holds only the host executable, the image and the glibc heap; the gaps between them stay
     * free for this process. MAP_FIXED_NOREPLACE only takes a gap that is still unmapped when
     * called, so nothing in use is ever replaced. */
    const uint64_t span = UINT64_C(0x60000000), min_addr = 0x10000;   /* vm.mmap_min_addr */
    const uint64_t lo = (uint64_t)(uintptr_t)image > span + min_addr ? (uint64_t)(uintptr_t)image - span : min_addr;
    const uint64_t hi = (uint64_t)(uintptr_t)image + size + span;
    FILE *maps = fopen("/proc/self/maps", "r");
    if (!maps) return NULL;
    unsigned char *found = NULL;
    char line[256];
    uint64_t prev_end = lo;   /* the free space below the first mapping counts too */
    int line_start = 1;
    while (!found && prev_end < hi && fgets(line, sizeof line, maps)) {
        /* A mapped file's path can be longer than the buffer: only a line's own start holds
         * its range, the rest of a long line is skipped. */
        const int at_start = line_start;
        line_start = strchr(line, '\n') != NULL;
        uint64_t a, b;
        if (!at_start || sscanf(line, "%" SCNx64 "-%" SCNx64, &a, &b) != 2) continue;
        if (a > prev_end) {
            const uint64_t start = near_gap_page(prev_end, a, lo, hi, page_size);
            if (start) {
                void *p = mmap((void *)(uintptr_t)start, page_size, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
                if (p != MAP_FAILED) found = p;
            }
        }
        if (b > prev_end) prev_end = b;
    }
    fclose(maps);
    if (!found && prev_end < hi) {   /* free up to the window's top, past the last mapping read */
        const uint64_t start = near_gap_page(prev_end, hi, lo, hi, page_size);
        if (start) {
            void *p = mmap((void *)(uintptr_t)start, page_size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
            if (p != MAP_FAILED) found = p;
        }
    }
    return found;
#endif
}
/* Mouse look stub (src/runtime_mouselook.h): saves rax, rcx, rdx, rsi, rdi, r8-r11 and xmm0-7
 * (the entry's argument and scratch registers; rsp stays 16-byte aligned at the call), calls the
 * handler with rdi/rsi/rdx unchanged, stores a non-NULL result over the saved rdx, restores them,
 * replays the replaced bytes and jumps back. */
static unsigned build_mouse_look_stub(unsigned char *p, const MouseLookSite *site, uintptr_t resume) {
    static const unsigned char save[] = {
        0x50, 0x51, 0x52, 0x56, 0x57, 0x41, 0x50, 0x41, 0x51, 0x41, 0x52, 0x41, 0x53,
        0x48, 0x81, 0xec, 0x80, 0x00, 0x00, 0x00,
        0xf3, 0x0f, 0x7f, 0x04, 0x24,       0xf3, 0x0f, 0x7f, 0x4c, 0x24, 0x10,
        0xf3, 0x0f, 0x7f, 0x54, 0x24, 0x20, 0xf3, 0x0f, 0x7f, 0x5c, 0x24, 0x30,
        0xf3, 0x0f, 0x7f, 0x64, 0x24, 0x40, 0xf3, 0x0f, 0x7f, 0x6c, 0x24, 0x50,
        0xf3, 0x0f, 0x7f, 0x74, 0x24, 0x60, 0xf3, 0x0f, 0x7f, 0x7c, 0x24, 0x70,
    };
    static const unsigned char restore[] = {
        0xf3, 0x0f, 0x6f, 0x04, 0x24,       0xf3, 0x0f, 0x6f, 0x4c, 0x24, 0x10,
        0xf3, 0x0f, 0x6f, 0x54, 0x24, 0x20, 0xf3, 0x0f, 0x6f, 0x5c, 0x24, 0x30,
        0xf3, 0x0f, 0x6f, 0x64, 0x24, 0x40, 0xf3, 0x0f, 0x6f, 0x6c, 0x24, 0x50,
        0xf3, 0x0f, 0x6f, 0x74, 0x24, 0x60, 0xf3, 0x0f, 0x6f, 0x7c, 0x24, 0x70,
        0x48, 0x81, 0xc4, 0x80, 0x00, 0x00, 0x00,
        0x41, 0x5b, 0x41, 0x5a, 0x41, 0x59, 0x41, 0x58, 0x5f, 0x5e, 0x5a, 0x59, 0x58,
    };
    unsigned n = 0;
    memcpy(p + n, save, sizeof(save)); n += sizeof(save);
    p[n++] = 0x48; p[n++] = 0xb8; /* mov rax, handler */
    const uint64_t handler = (uint64_t)(uintptr_t)site->handler;
    memcpy(p + n, &handler, 8); n += 8;
    p[n++] = 0xff; p[n++] = 0xd0; /* call rax */
    /* test rax, rax; jz +8; mov [rsp + 0xb0], rax (the saved rdx: 0x80 of xmm, then r11 r10 r9
     * r8 rdi rsi above it) */
    static const unsigned char replace_rdx[] = {0x48, 0x85, 0xc0, 0x74, 0x08,
                                                0x48, 0x89, 0x84, 0x24, 0xb0, 0x00, 0x00, 0x00};
    memcpy(p + n, replace_rdx, sizeof(replace_rdx)); n += sizeof(replace_rdx);
    memcpy(p + n, restore, sizeof(restore)); n += sizeof(restore);
    memcpy(p + n, site->expected, site->length); n += site->length;
    static const unsigned char jump[] = {0xff, 0x25, 0x00, 0x00, 0x00, 0x00}; /* jmp [rip+0] */
    memcpy(p + n, jump, sizeof(jump)); n += sizeof(jump);
    const uint64_t target = (uint64_t)resume;
    memcpy(p + n, &target, 8); n += 8;
    return n;
}
static void install_mouse_look(Segment *segments, uint64_t count, uint64_t size) {
    unsigned n = 0, used = 0, installed = 0;
    const MouseLookSite *sites = runtime_mouselook_sites(&n);
    unsigned char *stubs = NULL;
    for (unsigned i = 0; i < n; ++i) {
        const MouseLookSite *site = &sites[i];
        if (site->offset + site->expected_length > size || !mapped(segments, count, site->offset, site->expected_length) ||
            memcmp(image + site->offset, site->expected, site->expected_length)) continue;
        if (!stubs && !(stubs = allocate_near_image(size))) {
            puts("Mouse look: unavailable (no stub page in range)");
            return;
        }
        if (used + 256 > page_size) break;
        build_mouse_look_stub(stubs + used, site, (uintptr_t)(image + site->offset + site->length));
        if (!install_jump(site->offset, site->length, stubs + used)) {
            printf("Mouse look: %s unavailable (stub out of range)\n", site->game);
            continue;
        }
        runtime_mouselook_set_image(image);
        printf("Entry hook: %s at image offset %#llx\n", site->game, (unsigned long long)site->offset);
        used += 256;
        ++installed;
    }
    if (stubs) protect(stubs, page_size, 5);
    if (!installed && runtime_mouselook_enabled())
        puts("Mouse look: no known camera code in this executable (mouse acts as the right stick)");
}
/* Right stick boost (src/runtime_mouselook.h): each site's clamp jumps to a stub that replays it,
 * multiplies xmm4 (the clamped axis) by the boost and jumps back. The boost lives in the stub
 * page (executable and writable: the pad writes it per read, away from the code). */
static void install_stick_boost(Segment *segments, uint64_t count, uint64_t size) {
    unsigned n = 0, installed = 0;
    const StickBoostSite *sites = runtime_mouselook_boost_sites(&n);
    unsigned char *stubs = NULL;
    enum { STUB = 64, SLOT = 0x800, CLAMP = 9 };
    for (unsigned i = 0; i < n && (i + 1) * STUB <= SLOT; ++i) {
        const StickBoostSite *site = &sites[i];
        if (site->offset + sizeof(site->expected) > size || !mapped(segments, count, site->offset, sizeof(site->expected)) ||
            memcmp(image + site->offset, site->expected, sizeof(site->expected))) continue;
        if (!stubs && !(stubs = allocate_near_image(size))) {
            puts("Stick boost: unavailable (no stub page in range)");
            return;
        }
        unsigned char *p = stubs + i * STUB;
        unsigned k = 0;
        memcpy(p, site->expected, CLAMP); k += CLAMP;
        const int32_t slot = (int32_t)((stubs + SLOT) - (p + k + 8));
        p[k++] = 0xc5; p[k++] = 0xda; p[k++] = 0x59; p[k++] = 0x25; /* vmulss xmm4, xmm4, [rip + slot] */
        memcpy(p + k, &slot, 4); k += 4;
        static const unsigned char jump[] = {0xff, 0x25, 0x00, 0x00, 0x00, 0x00}; /* jmp [rip+0] */
        memcpy(p + k, jump, sizeof(jump)); k += sizeof(jump);
        const uint64_t resume = (uint64_t)(uintptr_t)(image + site->offset + CLAMP);
        memcpy(p + k, &resume, 8);
        if (!install_jump(site->offset, CLAMP, p)) {
            printf("Stick boost: %s unavailable (stub out of range)\n", site->game);
            continue;
        }
        printf("Stick boost: %s at image offset %#llx\n", site->game, (unsigned long long)site->offset);
        ++installed;
    }
    if (!stubs) return;
    const float one = 1.0f;
    memcpy(stubs + SLOT, &one, sizeof(one));
    protect(stubs, page_size, 7);
    if (installed) runtime_mouselook_set_boost_slot((float *)(void *)(stubs + SLOT));
}
/* Damage dealt/taken and green/blue/gold orb multipliers (src/damage_hook.h, src/orb_hook.h):
 * the stubs read the menu's floats directly, so changes apply on the next hit or pickup. */
static void install_stat_multipliers(Segment *segments, uint64_t count, uint64_t size) {
    const uint64_t damage_at = gow3_signature_find(image, size, &gow3_damage_site);
    const uint64_t pickup_at = gow3_signature_find(image, size, &gow3_orb_pickup_site);
    int damage = damage_at != UINT64_MAX && mapped(segments, count, damage_at, GOW3_DAMAGE_HOOK_LENGTH) &&
                 gow3_damage_signature_match(image + damage_at, size - damage_at);
    int pickup = pickup_at != UINT64_MAX && mapped(segments, count, pickup_at, GOW3_ORB_PICKUP_LENGTH) &&
                 gow3_orb_pickup_signature_match(image + pickup_at, size - pickup_at);
    unsigned char *stubs = damage || pickup ? allocate_near_image(size) : NULL;
    if ((damage || pickup) && !stubs) {
        puts("Multipliers: no stub page in range");
        damage = pickup = 0;
    }
    if (damage) {
        gow3_damage_build_stub(stubs, gow3_damage_multiplier(1), gow3_damage_multiplier(0),
                               (uintptr_t)(image + damage_at + GOW3_DAMAGE_HOOK_LENGTH));
    }
    if (pickup) {
        gow3_orb_pickup_build_stub(stubs + 256, gow3_orb_pickup_multiplier(0), gow3_orb_pickup_multiplier(1),
                                   gow3_orb_pickup_multiplier(2),
                                   (uintptr_t)(image + pickup_at + GOW3_ORB_PICKUP_LENGTH));
    }
    if (stubs) protect(stubs, page_size, 5);
    damage = damage && install_jump(damage_at, GOW3_DAMAGE_HOOK_LENGTH, stubs);
    pickup = pickup && install_jump(pickup_at, GOW3_ORB_PICKUP_LENGTH, stubs + 256);
    gow3_stat_multipliers_status(damage, pickup);
    printf("Multipliers: damage %s, green/blue/gold orbs %s\n", damage ? "ready" : "unavailable",
           pickup ? "ready" : "unavailable");
}
/* QUIT in the lobby and the pause menu (src/menu_exit_hook.h): the menu data gets the button as it
 * loads, and the dispatcher tells the runtime when it is pressed. GOW3_MENU_EXIT=0 installs none of the menu
 * hooks (QUIT, the PC SETTINGS page, the native save list). */
static void install_menu_exit(Segment *segments, uint64_t count, uint64_t size) {
    const char *env = getenv("GOW3_MENU_EXIT");
    if (env && env[0] == '0') {
        puts("Menu: the game's menu hooks are off (GOW3_MENU_EXIT=0): no QUIT, PC SETTINGS or native save list");
        return;
    }
    const uint64_t load_at = gow3_signature_find(image, size, &gow3_menu_load_site);
    const uint64_t click_at = gow3_signature_find(image, size, &gow3_menu_click_site);
    if (load_at == UINT64_MAX || click_at == UINT64_MAX || !mapped(segments, count, load_at, GOW3_MENU_LOAD_LENGTH) ||
        !mapped(segments, count, click_at, GOW3_MENU_CLICK_LENGTH)) {
        puts("Menu: QUIT button unavailable (code not found in this eboot.bin)");
        return;
    }
    /* The arena: the exit button's lists and the PC settings page (src/menu_pc_hook.h: the page
     * table is copied, ~14 KiB per menu data load), 32 slots of 32 KiB reused per load address
     * (src/runtime_menu_exit.c). */
    enum { CLICK_STUB = 512, TEXT_STUB = 1024, OPTION_STUB = 1536, UPDATE_STUB = 2048, COMMENTARY_STUB = 3072,
           FIRE_SHADOW_STUB = 2560, FIRE_ATLAS_STUB = 2816, ARENA_PAGES = 256 };
    uint64_t update_at = UINT64_MAX, commentary_at = UINT64_MAX;
    unsigned char *stubs = allocate_near_image(size);
    unsigned char *arena = allocate(page_size * ARENA_PAGES);
    if (!stubs || !arena) {
        puts("Menu: QUIT button unavailable (no memory in range)");
        return;
    }
    gow3_menu_build_click_stub(stubs + CLICK_STUB, (uintptr_t)runtime_menu_exit_button,
                               (uintptr_t)(image + click_at + GOW3_MENU_CLICK_LENGTH), image + click_at);
    gow3_menu_build_load_stub(stubs, (uintptr_t)runtime_menu_exit_data_loaded,
                              (uintptr_t)(image + load_at + GOW3_MENU_LOAD_LENGTH), image + load_at);
    /* The button's own label (QUIT TO DESKTOP, translated); the game's QUIT without this hook. */
    const uint64_t text_at = gow3_signature_find(image, size, &gow3_menu_text_lookup);
    const int text = text_at != UINT64_MAX && mapped(segments, count, text_at, GOW3_MENU_TEXT_LENGTH);
    unsigned text_original = 0;
    if (text)
        gow3_menu_build_text_stub(stubs + TEXT_STUB, (uintptr_t)runtime_menu_exit_text,
                                  (uintptr_t)(image + text_at + GOW3_MENU_TEXT_LENGTH), image + text_at, &text_original);
    /* The PC settings page's toggles and sliders: the binding factory's two calls go to a stub
     * that hands our option sids to the runtime (src/menu_pc_hook.h). */
    const char *pc_menu = getenv("GOW3_PC_MENU");
    const int pc_enabled = text && !(pc_menu && pc_menu[0] == '0');
    const uint64_t factory_at = pc_enabled ? gow3_signature_find(image, size, &gow3_menu_binding_factory) : UINT64_MAX;
    uint64_t factory_calls[GOW3_MENU_BINDING_CALLS];
    const unsigned factory_call_count = factory_at == UINT64_MAX ? 0
        : gow3_menu_find_calls(image, size, factory_at, factory_calls, GOW3_MENU_BINDING_CALLS);
    const int options = factory_call_count == GOW3_MENU_BINDING_CALLS;
    if (options) {
        gow3_menu_build_option_stub(stubs + OPTION_STUB, (uintptr_t)runtime_menu_pc_binding, (uintptr_t)(image + factory_at));
        runtime_menu_pc_set_factory(image + factory_at);
        /* The save list page (src/runtime_menu_saves.c) opens and closes with the game's own
         * page navigation. */
        const uint64_t open_at = gow3_signature_find(image, size, &gow3_menu_page_open);
        const uint64_t back_at = gow3_signature_find(image, size, &gow3_menu_page_back);
        if (open_at != UINT64_MAX && back_at != UINT64_MAX) runtime_menu_saves_set_navigation(image + open_at, image + back_at, image + click_at);
        update_at = gow3_signature_find(image, size, &gow3_menu_update_site);
        if (update_at != UINT64_MAX && mapped(segments, count, update_at, GOW3_MENU_UPDATE_LENGTH))
            gow3_menu_build_click_stub(stubs + UPDATE_STUB, (uintptr_t)runtime_menu_saves_update,
                                       (uintptr_t)(image + update_at + GOW3_MENU_UPDATE_LENGTH), image + update_at);
        else
            update_at = UINT64_MAX;
        /* The save list's footer: the Square prompt without the Triangle one (menu_pc_hook.h). */
        commentary_at = gow3_signature_find(image, size, &gow3_menu_footer_commentary);
        if (commentary_at != UINT64_MAX && mapped(segments, count, commentary_at, 64)) {
            commentary_at += GOW3_MENU_COMMENTARY_SKIP;
            gow3_menu_build_commentary_stub(stubs + COMMENTARY_STUB, (uintptr_t)runtime_menu_saves_commentary_flag(),
                                            image + commentary_at);
        } else {
            commentary_at = UINT64_MAX;
        }
    }
    static const unsigned char original_point_call[] = {0xe8, 0xc7, 0x02, 0x00, 0x00};
    const uint64_t point_call = 0xA635E4;
    const char *trace_selection = getenv("TLOU_TRACE_FIRE_SHADOW_SELECTION");
    trace_fire_shadow_selection = trace_selection && strcmp(trace_selection, "1") == 0;
    const char *dynamic_trace = getenv("TLOU_TRACE_DYNAMIC_BARRELS_ONLY");
    trace_dynamic_barrels_only = trace_fire_shadow_selection && dynamic_trace && !strcmp(dynamic_trace, "1");
    /* Count actual omni atlas assignments before the renderer's optional
     * logging branch. The retail log channel is unavailable in this build. */
    const char *trace_atlas = getenv("TLOU_TRACE_FIRE_SHADOW_ATLAS");
    const uint64_t atlas_assign_at = 0xA90084;
    static const unsigned char atlas_assign[] = {0x45, 0x88, 0x4f, 0x62, 0x41, 0xff, 0xc1};
    if (trace_fire_shadow_selection && trace_atlas && !strcmp(trace_atlas, "1") &&
        mapped(segments, count, atlas_assign_at, sizeof(atlas_assign)) &&
        !memcmp(image + atlas_assign_at, atlas_assign, sizeof(atlas_assign))) {
        unsigned char *stub = stubs + FIRE_ATLAS_STUB;
        unsigned char thunk[] = {
            0x50,                         /* push rax */
            0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, /* movabs rax, counter */
            0xf0, 0xff, 0x00,             /* lock inc dword ptr [rax] */
            0x58,                         /* pop rax */
            0x45, 0x88, 0x4f, 0x62, 0x41, 0xff, 0xc1, /* displaced assignment and increment */
            0xe9, 0, 0, 0, 0,            /* resume the retail code */
        };
        const uintptr_t counter = (uintptr_t)&fire_shadow_atlas_assignments;
        const int64_t relative = (int64_t)(uintptr_t)(image + atlas_assign_at + sizeof(atlas_assign)) -
                                 (int64_t)(uintptr_t)(stub + sizeof(thunk));
        if (relative >= INT32_MIN && relative <= INT32_MAX) {
            const int32_t displacement = (int32_t)relative;
            memcpy(thunk + 3, &counter, sizeof(counter));
            memcpy(thunk + sizeof(thunk) - sizeof(displacement), &displacement, sizeof(displacement));
            memcpy(stub, thunk, sizeof(thunk));
            if (install_jump(atlas_assign_at, sizeof(atlas_assign), stub))
                puts("FireShadow: omni atlas assignment counter installed");
        }
    }
    if (options &&
        mapped(segments, count, point_call, sizeof(original_point_call)) &&
        mapped(segments, count, 0xA638B0, 1) &&
        !memcmp(image + point_call, original_point_call, sizeof(original_point_call))) {
        unsigned char *stub = stubs + FIRE_SHADOW_STUB;
        const uint64_t target = (uintptr_t)submit_fire_point_light;
        stub[0] = 0x48;
        stub[1] = 0xb8;
        memcpy(stub + 2, &target, sizeof(target));
        stub[10] = 0xff;
        stub[11] = 0xe0;
        if (install_jump(point_call, sizeof(original_point_call), stub)) {
            image[point_call] = 0xe8;
            puts("TLOU_FIRE_SHADOWS_V6");
        }
    }

    /* Add the oil lamps before the light-list bounds are copied. The
     * workbench hook replaces only the authored spotlight submission. */
    const struct {
        uint64_t call, original;
        uintptr_t replacement;
    } extra_lights[] = {
        {0xA970C6, 0xABB420, (uintptr_t)submit_anchored_shadows_for_view},
        {0xA63897, 0xA63FF0, (uintptr_t)submit_workbench_spotlight},
    };
    for (unsigned i = 0; options && i < sizeof(extra_lights) / sizeof(*extra_lights); ++i) {
        const uint64_t at = extra_lights[i].call;
        const int32_t delta = (int32_t)((int64_t)extra_lights[i].original - (int64_t)(at + 5));
        unsigned char expected[5] = {0xe8};
        memcpy(expected + 1, &delta, sizeof(delta));
        if (!mapped(segments, count, at, sizeof(expected)) ||
            !mapped(segments, count, extra_lights[i].original, 1) ||
            memcmp(image + at, expected, sizeof(expected))) continue;
        unsigned char *stub = stubs + FIRE_SHADOW_STUB + 16 * (i + 1);
        stub[0] = 0x48;
        stub[1] = 0xb8;
        memcpy(stub + 2, &extra_lights[i].replacement, sizeof(uintptr_t));
        stub[10] = 0xff;
        stub[11] = 0xe0;
        if (install_jump(at, sizeof(expected), stub)) {
            image[at] = 0xe8;
            if (i == 0) anchor_views_installed = 1;
        }
    }
    if (options && install_barrel_view_test(stubs + FIRE_SHADOW_STUB + 64, segments, count)) {
        puts("TLOU_BARREL_SHADOW_VIEW_BOUNDS");
    }
    protect(stubs, page_size, 5);
    runtime_menu_exit_set_arena(arena, page_size * ARENA_PAGES);
    const uint64_t confirm_at = gow3_signature_find(image, size, &gow3_menu_confirm_open);
    const int confirm = confirm_at != UINT64_MAX && mapped(segments, count, confirm_at, 36);
    if (confirm) runtime_menu_exit_set_confirm(image + confirm_at);
    if (!install_jump(click_at, GOW3_MENU_CLICK_LENGTH, stubs + CLICK_STUB) ||
        !install_jump(load_at, GOW3_MENU_LOAD_LENGTH, stubs)) {
        puts("Menu: QUIT button unavailable (hook outside relative-jump range)");
        return;
    }
    const int label = text && install_jump(text_at, GOW3_MENU_TEXT_LENGTH, stubs + TEXT_STUB);
    if (label) runtime_menu_exit_set_text(stubs + TEXT_STUB + text_original);
    printf("Menu: QUIT button hooks ready (data loader %#llx, dispatcher %#llx, label %s, pause menu asks %s)\n",
           (unsigned long long)load_at, (unsigned long long)click_at,
           label ? "QUIT TO DESKTOP" : "QUIT (text lookup not found)",
           confirm ? "on the game's Yes/No page" : "in an overlay (game's Yes/No page not found)");
    /* The PC settings page's labels are ours: without the text hook it is left out. Its values
     * need both factory calls redirected (5-byte calls: only the target changes). */
    int redirected = 0;
    for (unsigned i = 0; options && label && i < GOW3_MENU_BINDING_CALLS; ++i) {
        const int64_t delta = (int64_t)(uintptr_t)(stubs + OPTION_STUB) - (int64_t)(uintptr_t)(image + factory_calls[i] + 5);
        if (delta < INT32_MIN || delta > INT32_MAX) continue;
        const int32_t relative = (int32_t)delta;
        memcpy(image + factory_calls[i] + 1, &relative, 4);
        ++redirected;
    }
    if (!label) puts("Menu: Options > PC SETTINGS unavailable (text lookup not found)");
    else if (!pc_enabled) puts("Menu: Options > PC SETTINGS off (GOW3_PC_MENU=0)");
    else if (redirected == GOW3_MENU_BINDING_CALLS)
        printf("Menu: Options > PC SETTINGS ready (option factory %#llx)\n", (unsigned long long)factory_at);
    else
        printf("Menu: Options > PC SETTINGS without values (option factory %s, %u calls, %d redirected)\n",
               factory_at == UINT64_MAX ? "not found" : "found", factory_call_count, redirected);
    /* The save list page opens from the menu's own update (src/runtime_menu_saves.c). */
    const int update = redirected == GOW3_MENU_BINDING_CALLS && update_at != UINT64_MAX &&
                       install_jump(update_at, GOW3_MENU_UPDATE_LENGTH, stubs + UPDATE_STUB);
    if (update) runtime_menu_saves_set_update_hooked();
    printf("Menu: save list page %s\n", update ? "ready (opened from the menu update)" : "unavailable (menu update not found)");
    const int commentary = update && commentary_at != UINT64_MAX &&
                           install_jump(commentary_at, GOW3_MENU_COMMENTARY_LENGTH, stubs + COMMENTARY_STUB);
    if (update && !commentary) puts("Menu: save list footer keeps the Triangle prompt (footer check not found)");
}
/* G3PATCH2 (patches.py): the patches' image base, then byte writes at image offsets, applied
 * after relocation. A write may replace a whole base-relative pointer slot (60/90 FPS++ swap
 * function pointers): the patch holds the address at the patches' base, rebased here. */
static void apply_patches(const char *path, Segment *segments, uint64_t ns, const Reloc *relocs, uint64_t nr) {
    FILE *f=fopen(path,"rb");
    char magic[8];
    if (!f || fread(magic,1,8,f)!=8 || memcmp(magic,"G3PATCH2",8)) fail("invalid patch file");
    uint64_t base=read64(f), count=read64(f), bytes=0, rebased=0;
    unsigned char data[4096];
    for (uint64_t i=0;i<count;++i) {
        uint64_t offset=read64(f), length=read64(f);
        if (!length || length>sizeof(data) || !mapped(segments,ns,offset,length) || fread(data,1,length,f)!=length)
            fail("bad patch entry");
        uint64_t slots[sizeof(data)/8]; size_t nslots=0;
        for (uint64_t r=0;r<nr;++r) {
            if (!(relocs[r].target<offset+length && offset<relocs[r].target+8)) continue;
            uint64_t target=relocs[r].target, value;
            if (relocs[r].kind || target<offset || target+8>offset+length) fail("patch overlaps a relocation");
            memcpy(&value,data+(target-offset),8);
            if (value<base || !mapped(segments,ns,value-base,1)) fail("patch writes a pointer outside the image");
            slots[nslots++]=target;
        }
        memcpy(image+offset,data,length);
        for (size_t s=0;s<nslots;++s) {
            uint64_t value;
            memcpy(&value,image+slots[s],8);
            value=(uint64_t)(uintptr_t)image+(value-base);
            memcpy(image+slots[s],&value,8);
        }
        rebased+=nslots;
        bytes+=length;
    }
    if (fgetc(f)!=EOF) fail("trailing data in patch file");
    fclose(f);
    printf("Patches: %" PRIu64 " writes, %" PRIu64 " bytes applied, %" PRIu64 " pointers rebased\n",count,bytes,rebased);
}
/* A TSC divisor of the game (TLOU Remastered eboot 1.11, 0x153D958): about 150 inline
 * "rdtsc; (now - start) / [0x153D958]" sequences turn raw TSC deltas into milliseconds with
 * this qword, burned in as 3,000,000 ticks per ms (the PS4's 3.0 GHz TSC). They are elapsed-time
 * checks (the renderer's frame watchdog at 0xB3D7E3 is one, FIXES["frame-timeout"] in
 * tools/probe_tlou.py); on a host whose TSC is not 3.0 GHz their milliseconds are off by
 * host/3.0 GHz, on every platform.
 * They are not the game's speed: the Windows build runs at normal speed with this divisor
 * untouched. The game's clock follows sceKernelGetTscFrequency, which must match the inline
 * RDTSC rate: tools/probe_tlou.py sets GOW3_NATIVE_TSC=1 for that, and before this port the
 * POSIX runtime ignored it (1 GHz reported against a 4 GHz rdtsc: the Linux "fast game").
 * GOW3_TSC_PATCH=1 (opt-in, POSIX) sets the divisor to the measured host rate anyway, to make
 * those checks' milliseconds real; the default leaves the image as on Windows. */
#ifndef _WIN32
static void calibrate_guest_tsc(unsigned char *image, Segment *segments, uint64_t ns) {
    const char *on = getenv("GOW3_TSC_PATCH");
    if (!on || strcmp(on, "1")) return;
    const uint64_t at = 0x153D958;
    if (!mapped(segments, ns, at, 8)) { printf("TSC: 0x%llx is not in this image; left as is\n", (unsigned long long)at); return; }
    uint64_t before;
    memcpy(&before, image + at, 8);
    if (before != 3000000) { printf("TSC: divisor already %llu (the PS4's 3,000,000 expected); left as is\n", (unsigned long long)before); return; }
    struct timespec t0, t1, sleep = {0, 125000000};
    clock_gettime(CLOCK_MONOTONIC, &t0);
    const uint64_t begin = __rdtsc();
    nanosleep(&sleep, NULL);
    const uint64_t stop = __rdtsc();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const long double seconds = (long double)(t1.tv_sec - t0.tv_sec) + (long double)(t1.tv_nsec - t0.tv_nsec) / 1e9L;
    if (seconds <= 0.05L) { printf("TSC: the measurement was too short; left as is\n"); return; }
    const uint64_t hz = (uint64_t)((long double)(stop - begin) / seconds);
    if (hz < 1000000000ULL || hz > 20000000000ULL) { printf("TSC: %.2f GHz is implausible; left as is\n", (double)hz / 1e9); return; }
    const uint64_t per_ms = (hz + 500) / 1000;
    memcpy(image + at, &per_ms, 8);
    printf("Runtime: guest TSC divisor %llu -> %llu (host TSC %.3f GHz; GOW3_TSC_PATCH, the game's elapsed-time checks ran %.3fx fast)\n",
           (unsigned long long)before, (unsigned long long)per_ms, (double)hz / 1e9, (double)per_ms / (double)before);
}
#endif
/* Restarts the game through run.sh (the settings menu: a new render resolution is a patch
 * applied at start). Descriptors are closed first so the old GPU device and its memory are
 * released before the new process opens its own. */
void runtime_restart(void) {
    fflush(NULL);
#ifdef _WIN32
    /* run.py sets GOW3_RESTART_COMMAND: the launch command line, started again (new patches). */
    const char *command=getenv("GOW3_RESTART_COMMAND");
    puts("Runtime: restarting through run.py");
    if (!command || !*command) { fputs("runtime_restart: GOW3_RESTART_COMMAND is not set\n",stderr); _exit(1); }
    char *line=_strdup(command);
    /* Same output handles: a launcher reading the game's log keeps reading the new launch. */
    STARTUPINFOA startup={.cb=sizeof(startup),.dwFlags=STARTF_USESTDHANDLES,
                          .hStdInput=GetStdHandle(STD_INPUT_HANDLE),.hStdOutput=GetStdHandle(STD_OUTPUT_HANDLE),
                          .hStdError=GetStdHandle(STD_ERROR_HANDLE)};
    PROCESS_INFORMATION process;
    if (!line || !CreateProcessA(NULL,line,NULL,NULL,TRUE,0,NULL,NULL,&startup,&process)) {
        fprintf(stderr,"runtime_restart: CreateProcess failed (%lu)\n",GetLastError()); _exit(1);
    }
    CloseHandle(process.hThread); CloseHandle(process.hProcess);
    /* The new launch waits for this process's GPU device to go away (run.py --after). */
    TerminateProcess(GetCurrentProcess(),0);
#else
    const char *command = getenv("GOW3_RESTART_COMMAND");
    syscall(SYS_close_range, 3u, ~0u, 0u);
    if (command && *command) {
        puts("Runtime: restarting through GOW3_RESTART_COMMAND");
        execl("/bin/sh", "sh", "-c", command, (char *)NULL);
    } else {
        puts("Runtime: restarting through run.sh");
        execlp("bash", "bash", "run.sh", (char *)NULL);
    }
    perror("runtime_restart: exec");
    _exit(1);
#endif
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
#ifndef _WIN32
    /* Keep host heap objects handed to the guest (thread handles, TLS) in the
       non-PIE brk heap, i.e. below 1 TiB: the guest packs pointers into 40 bits. */
    mallopt(M_ARENA_MAX,1);
    mallopt(M_MMAP_THRESHOLD,32*1024*1024);
#endif
    if (argc == 2 && !strcmp(argv[1], "--vulkan-only")) return vulkan_smoke();
#ifdef _WIN32
    /* The PS4 user range must stay free of host allocations (GPU driver, heaps), and the guest
     * TCB needs one of the 64 TEB TLS slots before DLLs (Vulkan drivers) take them. */
    runtime_memory_reserve();
    runtime_thread_tcb_offset();
#endif
    int cpu_only = 0, strict_imports = 0, check_imports = 0, game_hooks = 1, low_image = 0;
    /* Guest argv: --guest-path is argv[0], then --guest-arg values (a LoadExec handoff passes the
     * new program's complete argv). */
    const char *guest_argv[17]={"/app0/eboot.bin"};
    int guest_argc=1;
    unsigned timeout_seconds = 10;
    const char *content_profile=NULL, *app0=NULL, *user_dir=NULL, *patch_file=NULL, *red_zone_file=NULL;
    const char *app0_overlays[4]; int app0_overlay_count=0;
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--cpu-only")) cpu_only = 1;
        else if (!strcmp(argv[i], "--strict-imports")) strict_imports = 1;
        else if (!strcmp(argv[i], "--check-imports")) check_imports = 1;
        else if (!strcmp(argv[i], "--no-game-hooks")) game_hooks = 0;
        else if (!strcmp(argv[i], "--low-image")) low_image = 1;
        else if (!strcmp(argv[i], "--content-profile") && i+1<argc) content_profile=argv[++i];
        else if (!strcmp(argv[i], "--app0") && i+1<argc) app0=argv[++i];
        else if (!strcmp(argv[i], "--app0-overlay") && i+1<argc) {
            if (app0_overlay_count==4) { fprintf(stderr, "Too many --app0-overlay options\n"); return 1; }
            app0_overlays[app0_overlay_count++]=argv[++i];
        }
        else if (!strcmp(argv[i], "--user") && i+1<argc) user_dir=argv[++i];
        else if (!strcmp(argv[i], "--patches") && i+1<argc) patch_file=argv[++i];
        else if (!strcmp(argv[i], "--red-zone") && i+1<argc) red_zone_file=argv[++i];
        else if (!strcmp(argv[i], "--timeout") && i+1<argc) timeout_seconds=(unsigned)strtoul(argv[++i],NULL,10);
        else if (!strcmp(argv[i], "--guest-path") && i+1<argc) guest_argv[0]=argv[++i];
        else if (!strcmp(argv[i], "--guest-arg") && i+1<argc) {
            if (guest_argc+1>=(int)(sizeof(guest_argv)/sizeof(*guest_argv))) { fprintf(stderr, "Too many --guest-arg options\n"); return 1; }
            guest_argv[guest_argc++]=argv[++i];
        }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); return 1; }
    }
    if (argc < 2) {
        fprintf(stderr, "Usage: %s boot.bin [--cpu-only] [--strict-imports] [--check-imports] [--no-game-hooks] [--content-profile file] [--app0 dir] [--app0-overlay dir]... [--user dir] [--patches file] [--timeout seconds] [--guest-path /app0/file] [--guest-arg value]... | --vulkan-only\n", argv[0]);
        return 1;
    }
    /* The guest code needs the PS4's instruction sets (#33, #66); GOW3_CPU_CHECK=0 skips this. */
    const char *cpu_check = getenv("GOW3_CPU_CHECK");
    if (!cpu_check || strcmp(cpu_check, "0")) {
        char missing[128];
        if (cpu_missing_features(cpu_feature_words(), missing, sizeof missing)) {
            printf("STOP: this CPU lacks instruction sets the game's code uses: %s "
                   "(Intel Haswell 2013 or newer, or AMD Ryzen or newer, is needed)\n", missing);
            return 20;
        }
    }
    if (content_profile) {
        FILE *profile=fopen(content_profile,"rb");
        unsigned char data[28];
        if (!profile) fail("cannot open content profile");
        if (fread(data,1,sizeof(data),profile)!=sizeof(data) || fgetc(profile)!=EOF || memcmp(data,"G3CONT01",8)) fail("invalid content profile");
        fclose(profile);
        uint32_t values[5];
        for (unsigned i=0;i<5;++i) {
            unsigned char *p=data+8+i*4;
            values[i]=(uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;
        }
        runtime_content_configure(values);
    }
    if (app0) {
        runtime_file_configure(app0, user_dir ? user_dir : "user");
        for (int i=0;i<app0_overlay_count;++i) runtime_file_add_overlay(app0_overlays[i]);
        char sfo[4096], id[16]="";
        if (runtime_file_translate("/app0/sce_sys/param.sfo",sfo,sizeof(sfo))) snprintf(sfo,sizeof(sfo),"%s/sce_sys/param.sfo",app0);
        if (sfo_value(sfo,"INSTALL_DIR_SAVEDATA",id,sizeof(id),NULL) || sfo_value(sfo,"TITLE_ID",id,sizeof(id),NULL))
            runtime_savedata_configure(id);
    }
    gow3gpu_register_kernel();
#ifdef _WIN32
    SYSTEM_INFO system_info; GetSystemInfo(&system_info); page_size = system_info.dwPageSize;
    AddVectoredExceptionHandler(1, vectored_fault);
    SetUnhandledExceptionFilter(unhandled_fault);
    runtime_host_scheduling_init();
    if (getenv("GOW3_STRICT_HANDLES")) {
        PROCESS_MITIGATION_STRICT_HANDLE_CHECK_POLICY strict={0};
        strict.RaiseExceptionOnInvalidHandleReference=1;
        strict.HandleExceptionsPermanentlyEnabled=1;
        if (!SetProcessMitigationPolicy(ProcessStrictHandleCheckPolicy,&strict,sizeof(strict)))
            fprintf(stderr,"GOW3_STRICT_HANDLES: policy not set (%lu)\n",GetLastError());
    }
    (void)timeout_seconds; /* no watchdog on Windows */
#else
    page_size = (size_t)sysconf(_SC_PAGESIZE);
    struct sigaction sa = {0}; sa.sa_sigaction = fault; sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, NULL); sigaction(SIGILL, &sa, NULL); sigaction(SIGBUS, &sa, NULL);
    Dl_info self_info;
    if (dladdr((void *)main,&self_info)) exe_base=(uintptr_t)self_info.dli_fbase;
    struct sigaction dump = {0}; dump.sa_sigaction = thread_dump; dump.sa_flags = SA_SIGINFO|SA_RESTART;
    sigemptyset(&dump.sa_mask); sigaction(SIGUSR2, &dump, NULL);
    struct sigaction alarm_action = {0}; alarm_action.sa_sigaction = watchdog; alarm_action.sa_flags = SA_SIGINFO;
    sigemptyset(&alarm_action.sa_mask); sigaction(SIGALRM, &alarm_action, NULL);
    alarm(timeout_seconds); /* 0 disables the watchdog */
#endif
    FILE *f = fopen(argv[1], "rb");
    if (!f) fail("cannot open boot file; run prepare.py first");
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || (memcmp(magic, "G3PROBE1", 8) && memcmp(magic, "G3PROBE2", 8) && memcmp(magic,"G3PROBE3",8) && memcmp(magic,"G3PROBE4",8) && memcmp(magic,"G3PROBE5",8))) fail("bad boot file signature");
    uint64_t size = read64(f), entry = read64(f), ns = read64(f), nr = read64(f);
    import_count = read64(f);
    uint64_t capabilities = memcmp(magic, "G3PROBE1", 8) ? read64(f) : 0;
    if (capabilities & ~UINT64_C(1)) fail("unknown runtime capabilities");
    runtime_start(strict_imports ? 0 : capabilities);
    if (!size || size > 512*1024*1024 || entry >= size || !ns || ns > 64 || nr > 1000000 || import_count > 100000)
        fail("boot file limits exceeded");
    int multi=!memcmp(magic,"G3PROBE5",8);
    int linked=!memcmp(magic,"G3PROBE3",8) || !memcmp(magic,"G3PROBE4",8) || multi;
    int native_libc=linked && !strict_imports && (capabilities&1);
    uint64_t main_tls[4]={0};
    uint64_t nb=0,procparam=0;
    uint64_t *bindings=calloc(import_count ? import_count : 1,sizeof(*bindings));
    uint64_t *binding_kinds=calloc(import_count ? import_count : 1,sizeof(*binding_kinds));
    if (!bindings || !binding_kinds) fail("allocation failed");
    if (multi) {
        /* G3PROBE5: procparam, eboot TLS, module table, bindings (link_modules.py). */
        procparam=read64(f);
        for (int i=0;i<4;++i) main_tls[i]=read64(f);
        module_count=read64(f);
        if (!module_count || module_count>sizeof(modules)/sizeof(*modules)) fail("invalid module count");
        for (uint64_t m=0;m<module_count;++m) {
            LinkedModule *x=&modules[m];
            x->base=read64(f); x->size=read64(f); x->init=read64(f); x->tls_address=read64(f);
            x->tls_memsz=read64(f); x->tls_filesz=read64(f); x->tls_module=read64(f);
        }
        nb=read64(f);
    } else if (linked) {
        LinkedModule *x=&modules[0];
        module_count=1;
        x->base=read64(f); x->size=read64(f); x->init=read64(f);
        x->tls_address=read64(f); x->tls_memsz=read64(f); x->tls_filesz=read64(f); x->tls_module=2; nb=read64(f);
        procparam=read64(f);
        if (!memcmp(magic,"G3PROBE4",8)) for (int i=0;i<4;++i) main_tls[i]=read64(f);
    }
    if (linked) {
        if (main_tls[1]>main_tls[2] || main_tls[2]>1024*1024 || main_tls[0]>size ||
            main_tls[1]>size-main_tls[0] || (main_tls[3] & (main_tls[3]-1)) || main_tls[3]>4096)
            fail("invalid eboot TLS metadata");
        for (uint64_t m=0;m<module_count;++m) {
            LinkedModule *x=&modules[m];
            if (x->base>=size || !x->size || x->size>size-x->base || x->init<x->base || x->init-x->base>=x->size ||
                (x->tls_module && (x->tls_address>=size || x->tls_memsz>size-x->tls_address || x->tls_memsz>1024*1024 ||
                                   x->tls_filesz>x->tls_memsz || x->tls_module<2 || x->tls_module>7)))
                fail("invalid linked module metadata");
        }
        if (nb>import_count) fail("invalid binding count");
        for (uint64_t i=0;i<nb;++i) {
            uint64_t index=read64(f),address=read64(f),kind=read64(f);
            int inside=0;
            for (uint64_t m=0;m<module_count;++m)
                if (address>=modules[m].base && address-modules[m].base<modules[m].size) inside=1;
            if (index>=import_count || !inside || (kind!=1 && kind!=2) || bindings[index]) fail("invalid native binding");
            bindings[index]=address; binding_kinds[index]=kind;
        }
    }
    Segment *segments = calloc(ns, sizeof(*segments));
    Reloc *relocs = calloc(nr ? nr : 1, sizeof(*relocs));
    names = calloc(import_count ? import_count : 1, sizeof(*names));
    if (!segments || !relocs || !names) fail("allocation failed");
    for (uint64_t i = 0; i < ns; ++i) {
        segments[i].address = read64(f);
        segments[i].size = read64(f);
        segments[i].flags = read64(f);
        if (segments[i].address > size || segments[i].size > size - segments[i].address ||
            segments[i].address % page_size || segments[i].flags > 7) fail("bad segment");
    }
    if (fread(names, 128, import_count, f) != import_count) fail("truncated import names");
    for (uint64_t i = 0; i < import_count; ++i)
        if (!memchr(names[i], 0, 128)) fail("unterminated import name");
    for (uint64_t i = 0; i < nr; ++i) {
        relocs[i].target = read64(f);
        relocs[i].kind = read64(f);
        relocs[i].value = read64(f);
        relocs[i].addend = read64(f);
        if (!mapped(segments, ns, relocs[i].target, 8) || relocs[i].kind > 2 ||
            (relocs[i].kind && relocs[i].value >= import_count) ||
            (relocs[i].kind != 2 && relocs[i].addend) || relocs[i].addend >= page_size) fail("bad relocation");
    }
    for (uint64_t m=0;m<module_count;++m)
        if (!mapped(segments,ns,modules[m].init,1) || (modules[m].tls_module && !mapped(segments,ns,modules[m].tls_address,modules[m].tls_memsz)))
            fail("unmapped module metadata");
    if (module_count && !mapped(segments,ns,procparam,64)) fail("unmapped procparam");
    for (uint64_t i=0;i<import_count;++i) {
        if (!bindings[i]) continue;
        if (!mapped(segments,ns,bindings[i],1)) fail("unmapped native export");
        if (binding_kinds[i]==1) {
            int executable=0;
            for (uint64_t s=0;s<ns;++s)
                if ((segments[s].flags&1) && bindings[i]>=segments[s].address &&
                    bindings[i]-segments[s].address<segments[s].size) executable=1;
            if (!executable) fail("native function is not executable");
        }
    }
    /* The red-zone stubs follow the image (rel32 jmps reach them). */
    if (red_zone_file) red_zone=red_zone_load(red_zone_file);
    if (red_zone && red_zone->rz_code_offset!=round_page(size)) {
        printf("Red zone: protection unavailable (stub offset does not match this page size)\n");
        red_zone=NULL;
    }
    const size_t red_zone_area = red_zone ? round_page(red_zone_code_size(red_zone)) : 0;
    const size_t image_area = round_page(size) + red_zone_area;
    image = low_image ? allocate_low_image(image_area) : allocate(image_area);
    printf("Guest image base: %p (%s)\n",image,low_image ? "below 2 GiB" : "default range");
    if (fread(image, 1, size, f) != size || fgetc(f) != EOF) fail("incorrect memory image size");
    fclose(f);
    if (red_zone && !red_zone_matches(red_zone, image, size)) {
        printf("Red zone: protection unavailable (the table is for another image)\n");
        red_zone = NULL;
    }
    if (!cpu_only) {
        char title[128]="The Last of Us Remastered", serial[16]="UNKNOWN", sfo[4096];
        uint32_t attributes=0;
        if (!app0 || runtime_file_translate("/app0/sce_sys/param.sfo",sfo,sizeof(sfo))) snprintf(sfo,sizeof(sfo),"%s/sce_sys/param.sfo",app0 ? app0 : ".");
        sfo_value(sfo,"TITLE",title,sizeof(title),NULL);
        sfo_value(sfo,"TITLE_ID",serial,sizeof(serial),NULL);
        sfo_value(sfo,"ATTRIBUTE",NULL,0,&attributes);
        uint64_t sdk=0;
        if (procparam) memcpy(&sdk,image+procparam+16,8); /* procparam: size, magic, count, sdk_version */
        Gow3GpuConfig gpu={title,serial,user_dir ? user_dir : "user",(uint32_t)sdk,attributes,1920,1080};
        gpu_enabled=1; /* page tracking starts while the rasterizer registers guest memory */
        if (gow3gpu_init(&gpu)) fail("GPU initialization failed");
        printf("GPU: window and Vulkan presenter ready; SDK 0x%08x, %u HLE symbols\n",(unsigned)sdk,gow3gpu_symbol_count());
    }
    unsigned char *traps = allocate(round_page((import_count + 1) * 32));
    unsigned char *data_traps = allocate((import_count + 1) * page_size);
    protect(data_traps, (import_count + 1) * page_size, 0);
    for (uint64_t i = 0; i < import_count; ++i) {
        unsigned char *t = traps + i * 32;
        /* SysV: mov edi, index; movabs rax, handler; jmp rax. No fake return values. */
        uint32_t index = (uint32_t)i;
        uintptr_t handler = (uintptr_t)unresolved;
        t[0] = 0x48; t[1] = 0x89; t[2] = 0xfe; /* mov rsi,rdi: preserve arg0 */
        t[3] = 0xbf; memcpy(t + 4, &index, 4);
        t[8] = 0x48; t[9] = 0xb8; memcpy(t + 10, &handler, 8);
        t[18] = 0xff; t[19] = 0xe0;
    }
    for (uint64_t i = 0; i < nr; ++i) {
        uintptr_t value = relocs[i].kind == 2 ? (uintptr_t)(data_traps + page_size * relocs[i].value + relocs[i].addend)
                        : relocs[i].kind == 1 ? (uintptr_t)(traps + 32 * relocs[i].value)
                        : (uintptr_t)image + relocs[i].value;
        if (relocs[i].kind) {
            uintptr_t resolved = runtime_resolve(names[relocs[i].value], relocs[i].kind == 2);
            if (resolved) value = resolved + relocs[i].addend;
            else if (native_libc && bindings[relocs[i].value]) {
                uint64_t address=bindings[relocs[i].value];
                if (binding_kinds[relocs[i].value]!=relocs[i].kind || !mapped(segments,ns,address,relocs[i].addend+1)) fail("native export kind/range mismatch");
                value=(uintptr_t)image+address+relocs[i].addend;
            }
        }
        memcpy(image + relocs[i].target, &value, 8);
    }
    if (check_imports) {
        /* Lists every import the game would stop at, without running it (needs the GPU
         * library: without --cpu-only its symbols are registered above). */
        unsigned char *checked=calloc(import_count ? import_count : 1,1);
        if (!checked) fail("allocation failed");
        uint64_t missing=0;
        for (uint64_t i=0;i<nr;++i) {
            uint64_t index=relocs[i].value;
            if (!relocs[i].kind || checked[index]) continue;
            checked[index]=1;
            if (runtime_resolve(names[index],relocs[i].kind==2) || (native_libc && bindings[index])) continue;
            printf("MISSING: %s (%s)\n",names[index],runtime_import_name(names[index]));
            ++missing;
        }
        printf("Imports: %" PRIu64 " missing\n",missing);
        fflush(NULL);
        _exit(missing ? 1 : 0); /* no destructors: GPU threads are running */
    }
    if (patch_file) apply_patches(patch_file, segments, ns, relocs, nr);
#ifndef _WIN32
    calibrate_guest_tsc(image, segments, ns);
#endif
    if (gpu_enabled) install_mouse_look(segments, ns, size);
    if (gpu_enabled) install_stick_boost(segments, ns, size);
    if (gpu_enabled) install_menu_exit(segments, ns, size);
    runtime_gamesettings_set_image(image, size);
    if (gpu_enabled && game_hooks) {
        install_orb_multiplier(segments, ns, size);
        install_cheats(segments, ns, size);
        install_actor_watch(segments, ns, size);
        install_stat_multipliers(segments, ns, size);
    }
#ifdef _WIN32
    printf("Guest TCB loads redirected to the TEB TLS slot: %" PRIu64 "\n", patch_tcb_loads(segments, ns));
#endif
    /* Last: a group whose bytes a patch, a hook or the TCB rewrite changed is skipped whole. */
    if (red_zone) {
        const RedZoneReport report = red_zone_apply(red_zone, image, image + round_page(size));
        printf("Red zone: %u instructions protected in %u patches (%u through relays), %u patches skipped\n",
               report.protected_instructions, report.groups, report.short_relays, report.skipped);
        protect(image + round_page(size), red_zone_area, 5);
    }
    protect(traps, round_page((import_count + 1) * 32), 5);
    protect(image, round_page(size), 0);
    int executable_entry = 0;
    for (uint64_t i = 0; i < ns; ++i) {
        protect(image + segments[i].address, round_page(segments[i].size), (unsigned)segments[i].flags);
        if ((segments[i].flags & 1) && entry >= segments[i].address && entry - segments[i].address < segments[i].size)
            executable_entry = 1;
    }
    if (!executable_entry) fail("entry is not executable");
    printf("Mapped %" PRIu64 " bytes, %" PRIu64 " segments; applied %" PRIu64 " relocations\n", size, ns, nr);
    if (native_libc) {
        for (uint64_t m=0;m<module_count;++m) {
            int init_executable=0;
            for (uint64_t i=0;i<ns;++i)
                if ((segments[i].flags&1) && modules[m].init>=segments[i].address && modules[m].init-segments[i].address<segments[i].size) init_executable=1;
            if (!init_executable) fail("module init is not executable");
            if (modules[m].tls_module)
                runtime_set_module_tls(modules[m].tls_module,image+modules[m].tls_address,modules[m].tls_filesz,modules[m].tls_memsz);
        }
        runtime_set_main_tls(image+main_tls[0],main_tls[1],main_tls[2],main_tls[3]);
        runtime_thread_attach_main();
        runtime_set_procparam(image+procparam);
        _Static_assert(sizeof(Segment)==3*sizeof(uint64_t),"Segment is address, size, flags");
        runtime_set_main_module(image,(const uint64_t (*)[3])segments,ns);
        /* Dependencies start in link order (libc first), as the PS4 dynamic linker does. */
        for (uint64_t m=0;m<module_count;++m) {
            printf("Starting linked module %" PRIu64 " at image offset 0x%" PRIx64 "; native bindings=%" PRIu64 "\n",m,modules[m].init,nb);
            typedef int (ABI *ModuleInit)(uint64_t,void *,void *);
            int result=((ModuleInit)(image+modules[m].init))(0,NULL,NULL);
            printf("Module %" PRIu64 " initializer returned %d\n",m,result);
            if (result) fail("module initializer failed");
        }
    }
    printf("Entering original x86-64 code at guest offset 0x%" PRIx64 "\n", entry);
#ifdef _WIN32
    start_exec_watch();
    start_pokes();
#endif
    entered_game=1;
    struct { uint64_t argc; const char *argv[17]; } params = {(uint64_t)guest_argc, {NULL}};
    for (int i=0;i<guest_argc;++i) params.argv[i]=guest_argv[i];
    if (guest_argc>1) {
        printf("Guest arguments:");
        for (int i=0;i<guest_argc;++i) printf(" \"%s\"",guest_argv[i]);
        putchar('\n');
    }
#ifdef _WIN32
    typedef void (ABI *Entry)(void *, void (ABI *)(void));
    ((Entry)(image + entry))(&params, guest_exit);
#else
    /* The guest main thread runs on a stack below 1 TiB like PS4 stacks. */
    enum { MAIN_STACK=8*1024*1024 };
    unsigned char *stack=runtime_low_map(MAIN_STACK,PROT_READ|PROT_WRITE);
    if (!stack) fail("cannot allocate guest main stack");
    enter_on_stack(image+entry,&params,(void *)guest_exit,stack+MAIN_STACK-64);
#endif
    fail("entry unexpectedly returned");
}
