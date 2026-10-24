"""Stack-walk the faulting thread of the procdump minidump."""
import struct
import sys
from minidump.minidumpfile import MinidumpFile

PATH = sys.argv[1] if len(sys.argv) > 1 else "dumps/procdump/vkhr.exe_261024_170030.dmp"
mf = MinidumpFile.parse(PATH)

# exception
exc = mf.exception
line = str(exc).splitlines()
data_line = [l for l in line if "EXCEPTION_ACCESS_VIOLATION" in l][0]
parts = data_line.split("|")
tid = int(parts[0].strip(), 16)
fault_addr = int(parts[4].strip(), 16)
print("faulting thread:", hex(tid), "fault addr:", hex(fault_addr) if fault_addr else "?")

mods = {}
for m in mf.modules.modules:
    name = m.name.split("\\")[-1]
    mods[name] = (m.baseaddress, m.size)

def module_of(addr):
    for name, (base, size) in mods.items():
        if base <= addr < base + size:
            return name, addr - base
    return None, None

# faulting thread stack
target = None
for t in mf.threads.threads:
    if t.ThreadId == tid:
        target = t
        break
if target is None:
    print("thread not found; using first")
    target = mf.threads.threads[0]

ctx = target.ContextObject
rip, rsp, rbp = ctx.Rip, ctx.Rsp, ctx.Rbp
print("rip=%016x rsp=%016x rbp=%016x" % (rip, rsp, rbp))
m, off = module_of(rip)
print("rip in %s + %#x" % (m, off) if m else "rip not in known module")

st = target.Stack
# The stack bytes live in the dump file at the descriptor's RVA.
with open(PATH, "rb") as f:
    f.seek(st.Rva)
    data = f.read(st.DataSize)
base = st.StartOfMemoryRange
print("stack region: %016x len=%#x" % (base, len(data)))

# scan the whole stack region for values inside module ranges, ordered by address
hits = []
for off in range(0, len(data) - 7, 8):
    val = struct.unpack_from("<Q", data, off)[0]
    mname, rva = module_of(val)
    if mname:
        hits.append((base + off, val, mname, rva))

# heuristic: dedupe consecutive repeats, keep first occurrence
seen = {}
for addr, val, mname, rva in hits:
    seen.setdefault((val, mname), addr)
ordered = sorted(seen.items(), key=lambda kv: kv[1])
print("return-address candidates (stack addr ascending):")
for (val, mname), saddr in ordered:
    print("  stack %016x -> %s + %#x" % (saddr, mname, val - mods[mname][0]))
