/* TLOU Launcher.exe: opens the game launcher (tlou_launcher.pyw). No console window. In the
 * repository it sits beside the launcher and Python comes from the PATH; in a release package
 * (tools/package_windows.sh) it sits at the top, over launcher\ and python\. Extra arguments are
 * passed on. */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdlib.h>
#include <wchar.h>

#include "tlou-messages.h"

#define TITLE L"The Last of Us Remastered"
#define FAILED_WITHIN_MS 4000

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR arguments, int show)
{
    (void)instance, (void)previous, (void)show;
    static wchar_t dir[32768], script[32768], line[32768], shell[MAX_PATH], packaged[32768];
    DWORD length = GetModuleFileNameW(NULL, dir, 32768);
    if (length == 0 || length >= 32768)
        return 1;
    wchar_t *slash = wcsrchr(dir, L'\\');
    if (slash)
        *slash = 0;
    swprintf(script, 32768, L"%ls\\tlou_launcher.pyw", dir);
    if (GetFileAttributesW(script) == INVALID_FILE_ATTRIBUTES)
        swprintf(script, 32768, L"%ls\\launcher\\tlou_launcher.pyw", dir);
    if (GetFileAttributesW(script) == INVALID_FILE_ATTRIBUTES) {
        MessageBoxW(NULL, message(dir, MSG_NO_LAUNCHER_SCRIPT), TITLE,
                    MB_OK | MB_ICONERROR);
        return 1;
    }
    /* The package's pythonw.exe is started directly: a hidden cmd.exe running an interpreter is
     * what antivirus heuristics look for in malware loaders (issues #88, #112, #148). Without it
     * (the repository), through cmd.exe: a version manager's pythonw on the PATH may be a batch
     * file. */
    const wchar_t *program;
    swprintf(packaged, 32768, L"%ls\\python\\pythonw.exe", dir);
    if (GetFileAttributesW(packaged) != INVALID_FILE_ATTRIBUTES) {
        program = packaged;
        swprintf(line, 32768, L"\"%ls\" \"%ls\"%ls%ls", packaged, script,
                 arguments && *arguments ? L" " : L"", arguments ? arguments : L"");
    } else {
        if (!GetEnvironmentVariableW(L"ComSpec", shell, MAX_PATH))
            wcscpy(shell, L"C:\\Windows\\System32\\cmd.exe");
        program = shell;
        swprintf(line, 32768, L"\"%ls\" /c pythonw \"%ls\"%ls%ls", shell, script,
                 arguments && *arguments ? L" " : L"", arguments ? arguments : L"");
    }

    STARTUPINFOW startup = {.cb = sizeof(startup)};
    PROCESS_INFORMATION process;
    DWORD code = 1;
    BOOL started = CreateProcessW(program, line, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, dir, &startup, &process);
    if (started) {
        CloseHandle(process.hThread);
        /* Still running: the launcher is open. Gone with an error: it never opened. */
        if (WaitForSingleObject(process.hProcess, FAILED_WITHIN_MS) == WAIT_TIMEOUT)
            code = 0;
        else
            GetExitCodeProcess(process.hProcess, &code);
        CloseHandle(process.hProcess);
    }
    if (!started || code != 0) {
        MessageBoxW(NULL, message(dir, MSG_LAUNCHER_FAILED), TITLE,
                    MB_OK | MB_ICONERROR);
        return 1;
    }
    return 0;
}
