/* MIT: Linux launcher for a plugin-owned child; no independent service. */
#define _GNU_SOURCE
#include <sys/prctl.h>
#include <sys/resource.h>
#include <unistd.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc < 3) { fputs("usage: pins-ai-exec parent-pid executable [args]\n", stderr); return 64; }
    char *end;
    long expected = strtol(argv[1], &end, 10);
    if (*end || expected <= 1) return 64;
    /* Check both sides of prctl: parent can die before the signal is armed. */
    if (getppid() != expected || prctl(PR_SET_PDEATHSIG, SIGKILL) || getppid() != expected) return 70;
    if (setpriority(PRIO_PROCESS, 0, 10)) { perror("setpriority"); return 70; }
    struct rlimit core = { 0, 0 };
    if (setrlimit(RLIMIT_CORE, &core)) return 70;
    execv(argv[2], &argv[2]);
    perror("execv");
    return 71;
}
