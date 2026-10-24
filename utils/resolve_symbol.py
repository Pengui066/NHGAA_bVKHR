"""Resolve an address in vkhr.exe using its PDB via dbghelp."""
import ctypes
from ctypes import wintypes

dbghelp = ctypes.WinDLL("dbghelp.dll")

class SYMBOL_INFO(ctypes.Structure):
    _fields_ = [
        ("SizeOfStruct", wintypes.ULONG),
        ("TypeIndex", wintypes.ULONG),
        ("Reserved", ctypes.c_uint64 * 2),
        ("Index", wintypes.ULONG),
        ("Size", wintypes.ULONG),
        ("ModBase", ctypes.c_uint64),
        ("Flags", wintypes.ULONG),
        ("Value", ctypes.c_uint64),
        ("Address", ctypes.c_uint64),
        ("Register", wintypes.ULONG),
        ("Scope", wintypes.ULONG),
        ("Tag", wintypes.ULONG),
        ("NameLen", wintypes.ULONG),
        ("MaxNameLen", wintypes.ULONG),
        ("Name", ctypes.c_char * 2048),
    ]

dbghelp.SymInitialize.argtypes = [wintypes.HANDLE, wintypes.LPCSTR, wintypes.BOOL]
dbghelp.SymInitialize.restype = wintypes.BOOL
dbghelp.SymSetOptions.argtypes = [wintypes.DWORD]
dbghelp.SymLoadModule64.argtypes = [wintypes.HANDLE, wintypes.HANDLE, wintypes.LPCSTR,
                                    wintypes.LPCSTR, ctypes.c_uint64, wintypes.DWORD]
dbghelp.SymLoadModule64.restype = ctypes.c_uint64
dbghelp.SymFromAddr.argtypes = [wintypes.HANDLE, ctypes.c_uint64,
                                ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(SYMBOL_INFO)]
dbghelp.SymFromAddr.restype = wintypes.BOOL
dbghelp.SymGetLineFromAddr64.argtypes = [wintypes.HANDLE, ctypes.c_uint64,
                                         ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
dbghelp.SymGetLineFromAddr64.restype = wintypes.BOOL

EXE = r"D:\pengui000\Denoise\vkhr\bin\vkhr.exe"
PDB = r"D:\pengui000\Denoise\vkhr\bin\vkhr.pdb"
BASE = 0x00007ff7648f0000  # module base from the dump

h = ctypes.windll.kernel32.GetCurrentProcess()
dbghelp.SymSetOptions(0x10)  # SYMOPT_LOAD_LINES
assert dbghelp.SymInitialize(h, None, False)
assert dbghelp.SymLoadModule64(h, 0, EXE.encode(), PDB.encode(), BASE, 0x13c000)

def resolve(rva, label):
    addr = BASE + rva
    disp = ctypes.c_uint64()
    info = SYMBOL_INFO()
    info.SizeOfStruct = ctypes.sizeof(SYMBOL_INFO)
    info.MaxNameLen = 2048
    ok = dbghelp.SymFromAddr(h, addr, ctypes.byref(disp), ctypes.byref(info))
    if ok:
        print("%s: %s + 0x%x" % (label, info.Name.decode(), disp.value))
    else:
        print("%s: no symbol (err %d)" % (label, ctypes.windll.kernel32.GetLastError()))

# caller of wcsstr + a couple of neighbors for context
resolve(0x0F06F4, "vkhr.exe+0xF06F4 (wcsstr caller)")
resolve(0x0F06F4 - 0x10, "      -0x10")
resolve(0x0F06F4 - 0x20, "      -0x20")
resolve(0x0F06F4 - 0x40, "      -0x40")
resolve(0x0F06F4 - 0x60, "      -0x60")
