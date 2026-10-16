from pathlib import Path

import fiona

gdb = Path(
    r"C:\Users\CND367\Documents\Python_Scripts\PRMS\PRMS_Gallatin"
    r"\Domain_Builder\Domain_Files\gSSURGO_MT\gSSURGO_MT.gdb"
)

files = sorted(gdb.iterdir(), key=lambda p: p.stat().st_size, reverse=True)
print("total GB:", round(sum(p.stat().st_size for p in files) / 1e9, 2))
for path in files[:8]:
    print(f"{path.stat().st_size / 1e6:10.1f} MB  {path.name}")

try:
    names = fiona.listlayers(str(gdb), LIST_ALL_TABLES="YES")
    print([n for n in names if n.lower().startswith(("fras", "mu", "gdb_item"))])
except Exception as error:
    print("hidden table listing failed:", repr(error))