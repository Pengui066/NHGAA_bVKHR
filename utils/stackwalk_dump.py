"""Proper x64 stack walk of the faulting thread using dbghelp StackWalk64,
serving memory from the minidump's memory64 segments."""
import ctypes
from ctypes import wintypes
import sys
from minidump.minidumpfile import MinidumpFile

DUMP = sys.argv[1] if len(sys.argv) > 1 else "dumps/procdump/vkhr.exe_261024_170030.dmp"
EXE = r"D:\pengui000\Denoise\vkhr\bin\vkhr.exe"
PDB = r"D:\pengui000\Denoise\vkhr\bin\vkhr.pdb"

mf = MinidumpFile.parse(DUMP)
segs = mf.memory_segments_64.memory_segments

# ---- memory serving ----
seg_by_va = sorted(((s.start_virtual_address, s.start_virtual_address + s.size, s) for s in segs))
f = open(DUMP, "rb")

def read_mem(addr, n):
    out = bytearray()
    cur, remain = addr, n
    for start, end, s in seg_by_va:
        if cur >= end or start >= cur + remain:
            continue
        off = max(cur, start) - start
        ln = min(cur + remain, end) - max(cur, start)
        f.seek(s.start_file_address + off)
        out += f.read(ln)
        cur += ln
        remain -= ln
        if remain <= 0:
            break
    return bytes(out)

READMEM = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HANDLE, ctypes.c_uint64,
                             ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(ctypes.c_uint64))
@READMEM
def _readmem(hp, addr, buf, n, nread):
    try:
        data = read_mem(addr, n)
    except Exception:
        return False
    ctypes.memmove(buf, data, len(data))
    if nread:
        nread[0] = len(data)
    return len(data) == n

# ---- dbghelp ----
dbghelp = ctypes.WinDLL("dbghelp.dll")
k32 = ctypes.windll.kernel32

class KDHELP64(ctypes.Structure):
    _fields_ = [("Thread", ctypes.c_uint64), ("ThCallbackStack", wintypes.DWORD),
                ("NextCallback", wintypes.DWORD), ("FramePointer", wintypes.DWORD),
                ("KiCallUserMode", ctypes.c_uint64), ("KeUserCallbackDispatcher", ctypes.c_uint64),
                ("SystemRangeStart", ctypes.c_uint64), ("Reserved", ctypes.c_uint64 * 8)]

class STACKFRAME64(ctypes.Structure):
    _fields_ = [("AddrPC", ctypes.c_uint64), ("AddrReturn", ctypes.c_uint64),
                ("AddrFrame", ctypes.c_uint64), ("AddrStack", ctypes.c_uint64),
                ("AddrBStore", ctypes.c_uint64), ("FuncTableEntry", ctypes.c_void_p),
                ("Params", ctypes.c_uint64 * 4), ("Far", wintypes.BOOL),
                ("Virtual", wintypes.BOOL), ("Reserved", ctypes.c_uint64 * 3),
                ("KdHelp", KDHELP64)]

class SYMBOL_INFO(ctypes.Structure):
    _fields_ = [("SizeOfStruct", wintypes.ULONG), ("TypeIndex", wintypes.ULONG),
                ("Reserved", ctypes.c_uint64 * 2), ("Index", wintypes.ULONG), ("Size", wintypes.ULONG),
                ("ModBase", ctypes.c_uint64), ("Flags", wintypes.ULONG), ("Value", ctypes.c_uint64),
                ("Address", ctypes.c_uint64), ("Register", wintypes.ULONG), ("Scope", wintypes.ULONG),
                ("Tag", wintypes.ULONG), ("NameLen", wintypes.ULONG), ("MaxNameLen", wintypes.ULONG),
                ("Name", ctypes.c_char * 2048)]

dbghelp.SymInitialize.argtypes = [wintypes.HANDLE, wintypes.LPCSTR, wintypes.BOOL]
dbghelp.SymSetOptions.argtypes = [wintypes.DWORD]
dbghelp.SymLoadModuleExW.argtypes = [wintypes.HANDLE, wintypes.HANDLE, wintypes.LPCWSTR,
                                     wintypes.LPCWSTR, ctypes.c_uint64, wintypes.DWORD,
                                     ctypes.c_void_p, wintypes.DWORD]
dbghelp.SymLoadModuleExW.restype = ctypes.c_uint64
dbghelp.SymFromAddr.argtypes = [wintypes.HANDLE, ctypes.c_uint64,
                                ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(SYMBOL_INFO)]
dbghelp.SymFromAddr.restype = wintypes.BOOL
dbghelp.StackWalk64.argtypes = [wintypes.DWORD, wintypes.HANDLE, wintypes.HANDLE,
                                ctypes.POINTER(STACKFRAME64), ctypes.c_void_p,
                                READMEM, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
dbghelp.StackWalk64.restype = wintypes.BOOL
dbghelp.SymFunctionTableAccess64.argtypes = [wintypes.HANDLE, ctypes.c_uint64]
dbghelp.SymFunctionTableAccess64.restype = ctypes.c_void_p
dbghelp.SymGetModuleBase64.argtypes = [wintypes.HANDLE, ctypes.c_uint64]
dbghelp.SymGetModuleBase64.restype = ctypes.c_uint64

h = k32.GetCurrentProcess()
dbghelp.SymSetOptions(0x2 | 0x10)  # UNDNAME | LOAD_LINES
assert dbghelp.SymInitialize(h, None, False)
mods = {}
for m in mf.modules.modules:
    mods[m.name.split("\\")[-1]] = m.baseaddress
loaded = 0
for m in mf.modules.modules:
    p = m.name
    if not p or not p.lower().endswith((".exe", ".dll")):
        continue
    name = p.split("\\")[-1]
    if name == "vkhr.exe":
        p, _pdbfile = EXE, PDB
    else:
        _pdbfile = None
    dbghelp.SymLoadModuleExW(h, 0, p, _pdbfile, m.baseaddress, m.size, None, 0)
    loaded += 1
print("loaded %d modules at dump bases" % loaded)

# faulting thread
line = str(mf.exception).splitlines()
data_line = [l for l in line if "EXCEPTION_ACCESS_VIOLATION" in l][0]
parts = data_line.split("|")
tid = int(parts[0].strip(), 16)
target = None
for t in mf.threads.threads:
    if t.ThreadId == tid:
        target = t
        break
ctx_obj = target.ContextObject

class M128A(ctypes.Structure):
    _fields_ = [("Low", ctypes.c_uint64), ("High", ctypes.c_int64)]

class CONTEXT(ctypes.Structure):
    _pack_ = 16
    _fields_ = [
        ("P1Home", ctypes.c_uint64), ("P2Home", ctypes.c_uint64),
        ("P3Home", ctypes.c_uint64), ("P4Home", ctypes.c_uint64),
        ("P5Home", ctypes.c_uint64), ("P6Home", ctypes.c_uint64),
        ("ContextFlags", wintypes.DWORD), ("MxCsr", wintypes.DWORD),
        ("SegCs", wintypes.WORD), ("SegDs", wintypes.WORD),
        ("SegEs", wintypes.WORD), ("SegFs", wintypes.WORD),
        ("SegGs", wintypes.WORD), ("SegSs", wintypes.WORD),
        ("EFlags", wintypes.DWORD),
        ("Dr0", ctypes.c_uint64), ("Dr1", ctypes.c_uint64),
        ("Dr2", ctypes.c_uint64), ("Dr3", ctypes.c_uint64),
        ("Dr6", ctypes.c_uint64), ("Dr7", ctypes.c_uint64),
        ("Rax", ctypes.c_uint64), ("Rcx", ctypes.c_uint64),
        ("Rdx", ctypes.c_uint64), ("Rbx", ctypes.c_uint64),
        ("Rsp", ctypes.c_uint64), ("Rbp", ctypes.c_uint64),
        ("Rsi", ctypes.c_uint64), ("Rdi", ctypes.c_uint64),
        ("R8", ctypes.c_uint64), ("R9", ctypes.c_uint64),
        ("R10", ctypes.c_uint64), ("R11", ctypes.c_uint64),
        ("R12", ctypes.c_uint64), ("R13", ctypes.c_uint64),
        ("R14", ctypes.c_uint64), ("R15", ctypes.c_uint64),
        ("Rip", ctypes.c_uint64),
        ("Xmm", M128A * 16),
        ("VectorRegister", M128A * 26), ("VectorControl", ctypes.c_uint64),
        ("DebugControl", ctypes.c_uint64), ("LastBranchToRip", ctypes.c_uint64),
        ("LastBranchFromRip", ctypes.c_uint64), ("LastExceptionToRip", ctypes.c_uint64),
        ("LastExceptionFromRip", ctypes.c_uint64),
    ]

ctx = CONTEXT()
ctx.ContextFlags = ctx_obj.ContextFlags
ctx.MxCsr = ctx_obj.MxCsr
ctx.SegCs = ctx_obj.SegCs; ctx.SegDs = ctx_obj.SegDs; ctx.SegEs = ctx_obj.SegEs
ctx.SegFs = ctx_obj.SegFs; ctx.SegGs = ctx_obj.SegGs; ctx.SegSs = ctx_obj.SegSs
ctx.EFlags = ctx_obj.EFlags
for name in ("Dr0", "Dr1", "Dr2", "Dr3", "Dr6", "Dr7",
             "Rax", "Rcx", "Rdx", "Rbx", "Rsp", "Rbp", "Rsi", "Rdi",
             "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15", "Rip",
             "P1Home", "P2Home", "P3Home", "P4Home", "P5Home", "P6Home"):
    setattr(ctx, name, getattr(ctx_obj, name))
ctx_buf = ctypes.addressof(ctx)
print("faulting tid %#x rip %016x rsp %016x" % (tid, ctx_obj.Rip, ctx_obj.Rsp))

frame = STACKFRAME64()
frame.AddrPC = ctx_obj.Rip
frame.AddrFrame = ctx_obj.Rbp
frame.AddrStack = ctx_obj.Rsp
machine = 0x8664  # IMAGE_FILE_MACHINE_AMD64

mods = {}
for m in mf.modules.modules:
    mods[m.name.split("\\")[-1]] = m.baseaddress

def name_of(addr):
    mname = None
    for mn, base in mods.items():
        if base <= addr < base + 0x1000000:
            mname = mn
            break
    if mname is None:
        return "???"
    if mname == "vkhr.exe":
        disp = ctypes.c_uint64()
        info = SYMBOL_INFO()
        info.SizeOfStruct = ctypes.sizeof(SYMBOL_INFO)
        info.MaxNameLen = 2048
        if dbghelp.SymFromAddr(h, addr, ctypes.byref(disp), ctypes.byref(info)):
            return "vkhr!%s+0x%x" % (info.Name.decode(), disp.value)
    return "%s+%#x" % (mname, addr - mods[mname])

for i in range(64):
    ok = dbghelp.StackWalk64(machine, h, ctypes.c_void_p(tid), ctypes.byref(frame),
                             ctx_buf, _readmem,
                             ctypes.WINFUNCTYPE(ctypes.c_void_p, wintypes.HANDLE, ctypes.c_uint64)(
                                 dbghelp.SymFunctionTableAccess64),
                             ctypes.WINFUNCTYPE(ctypes.c_uint64, wintypes.HANDLE, ctypes.c_uint64)(
                                 dbghelp.SymGetModuleBase64),
                             None)
    err = ctypes.windll.kernel32.GetLastError()
    print("  iter %d ok=%d err=%d pc=%016x ret=%016x frame=%016x stack=%016x"
          % (i, ok, err, frame.AddrPC, frame.AddrReturn, frame.AddrFrame, frame.AddrStack))
    if not ok or frame.AddrPC == 0:
        break
    print("     %s" % name_of(frame.AddrPC))
