/* TLOU Game.exe: starts the game with the settings the launcher saved, without opening the
 * launcher (tools/run_game.py reads launcher.json through the launcher's own code). No console
 * window: the run's output goes to analysis\tlou\launcher-run.log. In the repository it sits at the
 * top and Python comes from the PATH; in a release package it sits beside python\. Extra arguments
 * are passed on (--print shows the settings the run would use). */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdlib.h>
#include <wchar.h>

#include "tlou-messages.h"

#define TITLE L"The Last of Us Remastered"
/* Enough for Python to start and fail loudly: the game itself runs far longer. */
#define FAILED_WITHIN_MS 6000

/* pyenv-win exposes a batch-file shim as `pythonw` and it can have no global version selected.
 * Its version folders contain normal python.exe files, which are usable without a console because
 * this starter creates the child with CREATE_NO_WINDOW. */
static BOOL find_pyenv_python(wchar_t *python, size_t size)
{
    static wchar_t root[32768], pattern[32768];
    WIN32_FIND_DATAW entry;
    DWORD length = GetEnvironmentVariableW(L"PYENV_ROOT", root, 32768);
    if (length == 0 || length >= 32768)
        return FALSE;
    swprintf(pattern, 32768, L"%ls\\versions\\*", root);
    HANDLE search = FindFirstFileW(pattern, &entry);
    if (search == INVALID_HANDLE_VALUE)
        return FALSE;
    do {
        if (!(entry.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || entry.cFileName[0] == L'.')
            continue;
        swprintf(python, size, L"%ls\\versions\\%ls\\python.exe", root, entry.cFileName);
        if (GetFileAttributesW(python) != INVALID_FILE_ATTRIBUTES) {
            FindClose(search);
            return TRUE;
        }
    } while (FindNextFileW(search, &entry));
    FindClose(search);
    return FALSE;
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR arguments, int show)
{
    (void)instance, (void)previous, (void)show;
    static wchar_t dir[32768], script[32768], line[32768], shell[MAX_PATH], packaged[32768], pyenv[32768];
    DWORD length = GetModuleFileNameW(NULL, dir, 32768);
    if (length == 0 || length >= 32768)
        return 1;
    wchar_t *slash = wcsrchr(dir, L'\\');
    if (slash)
        *slash = 0;
    swprintf(script, 32768, L"%ls\\tools\\run_game.py", dir);
    if (GetFileAttributesW(script) == INVALID_FILE_ATTRIBUTES)
        swprintf(script, 32768, L"%ls\\..\\tools\\run_game.py", dir);
    if (GetFileAttributesW(script) == INVALID_FILE_ATTRIBUTES) {
        MessageBoxW(NULL, message(dir, MSG_NO_GAME_SCRIPT), TITLE,
                    MB_OK | MB_ICONERROR);
        return 1;
    }
    /* A version manager's pythonw on the PATH may be a batch file. Bundled and pyenv Python
     * runtimes are normal executables, so launch those directly and bypass cmd.exe entirely. */
    if (!GetEnvironmentVariableW(L"ComSpec", shell, MAX_PATH))
        wcscpy(shell, L"C:\\Windows\\System32\\cmd.exe");
    swprintf(packaged, 32768, L"%ls\\python\\python.exe", dir);
    const wchar_t *python = NULL;
    if (GetFileAttributesW(packaged) != INVALID_FILE_ATTRIBUTES)
        python = packaged;
    else {
        /* A repository checkout can use the locally built release runtime too. */
        swprintf(packaged, 32768, L"%ls\\dist\\TLOU-Remastered-windows\\python\\python.exe", dir);
        if (GetFileAttributesW(packaged) != INVALID_FILE_ATTRIBUTES)
            python = packaged;
    }
    if (!python && find_pyenv_python(pyenv, 32768))
        python = pyenv;
    if (python)
        swprintf(line, 32768, L"\"%ls\" \"%ls\"%ls%ls", python, script,
                 arguments && *arguments ? L" " : L"", arguments ? arguments : L"");
    else
        swprintf(line, 32768, L"\"%ls\" /d /c start \"\" /b /wait pythonw \"%ls\"%ls%ls", shell, script,
                 arguments && *arguments ? L" " : L"", arguments ? arguments : L"");
    /* The child's output is kept beside this file: a windowless process has nowhere else to say
     * why it failed, and this is the only trace of a launch that never reached the game. */
    STARTUPINFOW startup = {.cb = sizeof(startup)};
    PROCESS_INFORMATION process;
    DWORD code = 1;
    static wchar_t redirect[32768];
    HANDLE child_log = INVALID_HANDLE_VALUE;
    wchar_t *command = line;
    if (python) {
        swprintf(redirect, 32768, L"%ls\\tlou-game-child.log", dir);
        /* Inheritable, or the child gets no output handles and the log stays empty. */
        SECURITY_ATTRIBUTES inherit = {.nLength = sizeof(inherit), .bInheritHandle = TRUE};
        child_log = CreateFileW(redirect, GENERIC_WRITE, FILE_SHARE_READ, &inherit, CREATE_ALWAYS,
                                FILE_ATTRIBUTE_NORMAL, NULL);
        if (child_log != INVALID_HANDLE_VALUE) {
            startup.dwFlags |= STARTF_USESTDHANDLES;
            startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
            startup.hStdOutput = child_log;
            startup.hStdError = child_log;
        }
    } else {
        swprintf(redirect, 32768, L"%ls > \"%ls\\tlou-game-child.log\" 2>&1", line, dir);
        command = redirect;
    }
    BOOL started = CreateProcessW(python, command, NULL, NULL, child_log != INVALID_HANDLE_VALUE,
                                  CREATE_NO_WINDOW, NULL, dir,
                                  &startup, &process);
    if (child_log != INVALID_HANDLE_VALUE)
        CloseHandle(child_log);
    if (started) {
        CloseHandle(process.hThread);
        /* Still running: the game was started. Gone at once: Python or the script failed. */
        if (WaitForSingleObject(process.hProcess, FAILED_WITHIN_MS) == WAIT_TIMEOUT)
            code = 0;
        else
            GetExitCodeProcess(process.hProcess, &code);
        CloseHandle(process.hProcess);
    }
    if (!started || code != 0) {
        MessageBoxW(NULL, message(dir, MSG_GAME_FAILED), TITLE,
                    MB_OK | MB_ICONERROR);
        return 1;
    }
    return 0;
}
