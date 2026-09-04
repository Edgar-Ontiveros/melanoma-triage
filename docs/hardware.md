# Diagnóstico de hardware para entrenamiento de CNNs

- **Equipo:** Lenovo 21T1001BLM (laptop), host Windows 11 + WSL2 Ubuntu 24.04
- **Fecha del diagnóstico:** 2026-09-04
- **Modo:** solo lectura. No se instaló nada ni se modificó configuración. El único archivo temporal creado (1 GiB para la prueba de disco) fue borrado.

Leyenda de cada valor:

- **[M]** medido o leído directamente del sistema con el comando indicado.
- **[E]** estimado; el razonamiento se explica junto al número.
- **[F]** ficha técnica del fabricante, no verificado en esta máquina.

Columna **Fuente**:

- **host Windows**: leído de la máquina física a través de `powershell.exe`, `wsl.exe` o `cmd.exe` invocados desde WSL. Describe el hardware real.
- **WSL/Ubuntu**: leído desde dentro de la VM de WSL2. Describe lo que Linux ve, que en RAM y disco es un recorte virtual del hardware real.

---

## 1. Sistema

| Campo | Valor | Tipo | Fuente |
|---|---|---|---|
| SO (dentro de WSL) | Ubuntu 24.04.4 LTS (Noble Numbat) | [M] | WSL/Ubuntu |
| Kernel | 6.18.33.2-microsoft-standard-WSL2 | [M] | WSL/Ubuntu |
| Arquitectura | x86_64 | [M] | WSL/Ubuntu |
| Host | Microsoft Windows 11, build 10.0.26200.9168 | [M] | host Windows |
| ¿WSL2? | Sí. WSL versión 2.7.12.0, distro `Ubuntu-24.04` (versión 2). Hay una segunda distro `docker-desktop`, detenida. | [M] | host Windows |
| Hostname | ECCM-0177 | [M] | WSL/Ubuntu |

## 2. CPU

| Campo | Valor | Tipo | Fuente |
|---|---|---|---|
| Modelo | AMD Ryzen 7 250 w/ Radeon 780M Graphics (familia 25, modelo 117, Zen 4 "Hawk Point") | [M] | WSL/Ubuntu y host Windows (coinciden) |
| Núcleos físicos | 8 | [M] | host Windows (WSL reporta lo mismo) |
| Hilos lógicos | 16 (todos visibles en WSL, `nproc` = 16) | [M] | host Windows y WSL/Ubuntu |
| Frecuencia base | 3.30 GHz (`MaxClockSpeed` = 3301 MHz en WMI) | [M] | host Windows |
| Frecuencia actual al momento de la lectura | 2.65 GHz (`CurrentClockSpeed` = 2646 MHz) | [M] | host Windows |
| Frecuencia turbo | no disponible desde el SO. WMI no expone el boost y `/sys/devices/system/cpu/cpu0/cpufreq` no existe en WSL2. Ficha AMD: hasta 5.1 GHz. | [F] | ninguna (ficha técnica) |
| AVX2 | Sí | [M] | WSL/Ubuntu |
| AVX-512 | Sí: `avx512f`, `avx512bw`, `avx512vl`, `avx512dq`, `avx512_vnni`, `avx512_bf16`, `avx512vbmi` | [M] | WSL/Ubuntu |
| FMA | Sí | [M] | WSL/Ubuntu |
| Caché | L1d 8×32 KiB, L2 8×1 MiB, L3 16 MiB | [M] | WSL/Ubuntu |
| Virtualización | Hypervisor Microsoft (WSL2 corre como VM) | [M] | WSL/Ubuntu |

Los flags de CPU que ve WSL son los del procesador físico: Hyper-V expone el conjunto de instrucciones completo a la VM, así que los datos de CPU de ambas fuentes describen el mismo hardware.

## 3. RAM

> **Nota VM vs físico.** Los valores marcados WSL/Ubuntu son los de la máquina virtual, no los de la laptop. La RAM física real es **32 GB DDR5-5600**. WSL ve **16 GB** porque `.wslconfig` en el host lo limita a `memory=16GB`. Verificado el 2026-09-04 en dos lecturas: la primera durante el diagnóstico inicial y la segunda tras un reinicio de la VM (uptime 11 min); `.wslconfig` no cambió entre ambas y `MemTotal` fue idéntico. Para que Linux vea más RAM habría que editar ese archivo y reiniciar WSL, cosa que no se hizo.

| Campo | Valor | Tipo | Fuente |
|---|---|---|---|
| Total físico en el host | 32 GB (`TotalPhysicalMemory` = 31,854,321,664 bytes) | [M] | host Windows |
| Módulos | 1 × 32 GB SK Hynix, SO-DIMM (FormFactor 12) | [M] | host Windows |
| Tipo | DDR5 (`SMBIOSMemoryType` = 34) | [M] | host Windows |
| Velocidad | 5600 MT/s (Speed y ConfiguredClockSpeed = 5600) | [M] | host Windows |
| Canales | Un solo módulo, es decir, un solo canal. Reduce el ancho de banda de memoria, relevante para la iGPU y para entrenar en CPU. | [M] | host Windows |
| Límite configurado para la VM | `memory=16GB` en `C:\Users\<usuario>\.wslconfig` (leído dos veces, sin cambios) | [M] | host Windows (`.wslconfig`) |
| Total visible en WSL (VM) | 15.6 GiB (`MemTotal` = 16,375,448 kB), idéntico en ambas lecturas | [M] | WSL/Ubuntu |
| Disponible en WSL (VM) | Lectura 1: 14.5 GiB (`MemAvailable` = 15,185,356 kB). Lectura 2, tras reinicio de la VM: 14.5 GiB (`MemAvailable` = 15,176,812 kB) | [M] | WSL/Ubuntu |
| Swap en WSL (VM) | 4.0 GiB | [M] | WSL/Ubuntu |

Nota: `dmidecode` dentro de WSL requiere sudo con contraseña y no se ejecutó; el tipo y la velocidad se leyeron por WMI desde el host.

## 4. Almacenamiento

> **Nota VM vs físico.** La partición `/` de WSL (donde vive `/home/edgar`) es un disco virtual: un archivo `ext4.vhdx` de 1 TB nominal que hoy ocupa 9.2 GB y crece dinámicamente dentro de C:. Los 948 GB "libres" que reporta Linux son el tope nominal del vhdx, no espacio real. El disco físico es un **NVMe Samsung de 1 TB** con **768 GB libres en C:**; ese es el límite verdadero para dataset y checkpoints. Las cifras `ROTA=1` de `lsblk` son un artefacto de Hyper-V.

| Campo | Valor | Tipo | Fuente |
|---|---|---|---|
| Disco físico | SAMSUNG MZAL81T0HFLB-00BLL, 1.02 TB, `MediaType` = SSD, `BusType` = **NVMe** | [M] | host Windows |
| Unidad C: del host | NTFS, 951.6 GB, **767.9 GB libres**. Límite real, porque el vhdx crece sobre C:. | [M] | host Windows |
| Archivo del disco virtual de WSL | `C:\Users\EONTIVEROS\AppData\Local\wsl\{abbdc2bc-…}\ext4.vhdx`, tamaño actual 9.2 GB | [M] | host Windows |
| Partición raíz de WSL (`/`, donde vive `/home/edgar`) (VM) | `/dev/sdd`, ext4, 1007 GB nominales, 7.8 GB usados, 948 GB "libres" (tope del vhdx, no espacio físico) | [M] | WSL/Ubuntu |
| Naturaleza de `/dev/sdd` (VM) | Disco virtual de Hyper-V (`MODEL` = Virtual Disk) | [M] | WSL/Ubuntu |
| `lsblk ROTA` (VM) | Reporta 1 (rotacional) para todos los discos virtuales; no refleja el disco físico. | [M] | WSL/Ubuntu |
| Escritura secuencial 1 GiB (`oflag=direct`, `fsync`) | 3.6 GB/s, medido sobre el vhdx desde la VM | [M] | WSL/Ubuntu |
| Lectura secuencial 1 GiB (`iflag=direct`), corrida 1 | 5.5 GB/s, medido sobre el vhdx desde la VM | [M] | WSL/Ubuntu |
| Lectura secuencial 1 GiB (`iflag=direct`), corrida 2 | 7.8 GB/s, medido sobre el vhdx desde la VM | [M] | WSL/Ubuntu |

Nota sobre la medición: `iflag=direct` evita la caché de páginas de Linux, pero no la caché del host Windows sobre el vhdx. Los valores superan el máximo teórico de un NVMe PCIe 4.0 x4 (~7 GB/s), así que la segunda corrida seguramente se sirvió parcialmente desde RAM del host. El valor conservador es 3.6 a 5.5 GB/s; en cualquier caso el disco no será cuello de botella para cargar imágenes. No se midió velocidad directamente sobre NTFS.

## 5. GPU

| Campo | Valor | Tipo | Fuente |
|---|---|---|---|
| ¿GPU dedicada? | **No. No hay GPU dedicada.** Solo la iGPU integrada en el procesador. | [M] | host Windows |
| Fabricante / modelo | AMD Radeon 780M Graphics (RDNA 3 integrada, 12 CU) | [M] (modelo), [F] (CUs) | host Windows |
| VRAM | 2 GB reservados (`AdapterRAM` = 2,147,483,648 bytes). Es memoria del sistema compartida (UMA), no VRAM propia. | [M] | host Windows |
| Driver AMD | 32.0.22024.13003 | [M] | host Windows |
| NVIDIA | No hay. `nvidia-smi` no existe en WSL (`command not found`), no hay `nvcc` ni `/usr/local/cuda`. `lspci` no está instalado en WSL. WMI del host solo lista la Radeon 780M. | [M] | WSL/Ubuntu y host Windows |
| Versión de CUDA del driver | no disponible: no hay driver NVIDIA | [M] | WSL/Ubuntu y host Windows |
| ROCm en WSL | No instalado (`rocminfo` no existe, `/opt/rocm` no existe). | [M] | WSL/Ubuntu |
| `/dev/dxg` | Presente (paravirtualización de GPU de WSL2). Solo serviría para DirectML, que no es una ruta soportada para entrenar con PyTorch de forma seria. | [M] | WSL/Ubuntu |
| Apple Silicon | No aplica | [M] | host Windows |

## 6. Entorno Python

Solo se inventarió el Python de Linux. No se revisó si existe un Python nativo de Windows.

| Campo | Valor | Tipo | Fuente |
|---|---|---|---|
| Intérpretes | Python 3.12.3 en `/usr/bin/python3` (único; `uv python list` solo muestra el del sistema) | [M] | WSL/Ubuntu |
| Gestores | pip 24.0, uv 0.12.5. conda, mamba, poetry y pipx: no instalados | [M] | WSL/Ubuntu |
| Entornos virtuales encontrados | `projects/extractor-sap/.venv` y `projects/test-python/.venv`, ambos Python 3.12.3 | [M] | WSL/Ubuntu |
| `torch` instalado | **No**, ni en el Python del sistema ni en los dos venvs (`ModuleNotFoundError: No module named 'torch'`) | [M] | WSL/Ubuntu |
| `torch.__version__` | no disponible: torch no está instalado | [M] | WSL/Ubuntu |
| `torch.cuda.is_available()` | no disponible: torch no está instalado. Si se instalara, devolvería `False` porque no hay GPU NVIDIA ni ROCm. | [M] / [E] | WSL/Ubuntu |
| `torch.backends.mps.is_available()` | no disponible: torch no está instalado. Devolvería `False`: MPS es exclusivo de macOS. | [M] / [E] | WSL/Ubuntu |
| `onnxruntime` | No instalado | [M] | WSL/Ubuntu |
| `numpy` | No instalado en el Python del sistema | [M] | WSL/Ubuntu |
| Python nativo de Windows | no disponible: no se revisó | — | host Windows (no consultado) |

## 7. Contenedores

| Campo | Valor | Tipo | Fuente |
|---|---|---|---|
| Docker en WSL | El binario `docker` que aparece en el PATH es el shim de Docker Desktop; al ejecutarlo responde: "The command 'docker' could not be found in this WSL 2 distro. We recommend to activate the WSL integration". La integración WSL **no está activada** para `Ubuntu-24.04`. | [M] | WSL/Ubuntu |
| Docker Desktop | Docker version 29.7.2, build a7dcaa6 | [M] | host Windows |
| Docker Compose | v5.4.0 (plugin `docker compose`; `docker-compose` v1 no existe) | [M] | host Windows |
| Daemon | No estaba corriendo al momento del diagnóstico (`docker.exe info` falla con "cannot find the file specified" en el pipe `dockerDesktopLinuxEngine`; distro `docker-desktop` en estado Stopped). Los runtimes no pudieron leerse. | [M] | host Windows |
| NVIDIA Container Toolkit | **No presente**: `nvidia-ctk` y `nvidia-container-cli` no existen, ningún paquete `nvidia-container*` en dpkg, no hay `/etc/docker/daemon.json`. Tampoco tendría sentido: no hay GPU NVIDIA. | [M] | WSL/Ubuntu |

## 8. Red

| Campo | Valor | Tipo | Fuente |
|---|---|---|---|
| Endpoint | `https://speed.cloudflare.com/__down?bytes=50000000` (50 MB) | [M] | WSL/Ubuntu |
| Corrida 1 | 50,000,000 bytes en 4.55 s → 11.0 MB/s ≈ **88 Mbit/s** | [M] | WSL/Ubuntu |
| Corrida 2 | 50,000,000 bytes en 4.50 s → 11.1 MB/s ≈ **89 Mbit/s** | [M] | WSL/Ubuntu |
| Modo de red WSL | `networkingMode=mirrored` en `.wslconfig` | [M] | host Windows (`.wslconfig`) |

En modo `mirrored` la VM usa directamente la interfaz de red del host, así que la velocidad medida desde WSL representa la conexión física de la laptop. Con ~11 MB/s, descargar un dataset de 20 GB toma unos 30 minutos; subirlo a una GPU rentada dependería de la velocidad de subida, que no se midió.

---

## Comandos ejecutados

Todos se corrieron desde bash dentro de WSL (Ubuntu-24.04). Los que terminan en `.exe` invocan al host Windows a través de la interoperabilidad de WSL y son los que alimentan la fuente "host Windows".

### 1. Sistema
```bash
cat /etc/os-release
uname -a
uname -m
cat /proc/version
cat /proc/sys/fs/binfmt_misc/WSLInterop
echo "$WSL_DISTRO_NAME"
wsl.exe --version | tr -d '\0'
wsl.exe -l -v | tr -d '\0\r'
cmd.exe /c ver | tr -d '\r'
```

### 2. CPU
```bash
lscpu
grep -o -w 'avx2\|avx512f\|avx512bw\|avx512vl\|avx512_vnni\|avx_vnni\|fma\|amx_tile' /proc/cpuinfo | sort | uniq -c
nproc
cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq   # no existe en WSL2
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,CurrentClockSpeed | Format-List"
```

### 3. RAM
```bash
free -h
grep -E 'MemTotal|MemAvailable|SwapTotal' /proc/meminfo
sudo -n dmidecode -t memory        # falla: requiere contraseña, no se usó
cat /mnt/c/Users/*/.wslconfig     # ejecutado dos veces, junto con free -h y uptime, para verificar el límite
uptime
powershell.exe -NoProfile -Command "Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory,Manufacturer,Model | Format-List; Get-CimInstance Win32_PhysicalMemory | Select-Object Manufacturer,Capacity,Speed,ConfiguredClockSpeed,SMBIOSMemoryType,FormFactor | Format-List"
```

### 4. Almacenamiento
```bash
df -h -x tmpfs -x devtmpfs -x overlay -x 9p -x drvfs
df -h /home /home/edgar/projects
findmnt -T /home/edgar -o TARGET,SOURCE,FSTYPE,OPTIONS
lsblk -o NAME,SIZE,TYPE,ROTA,MOUNTPOINTS,MODEL,TRAN
for d in /sys/block/*/queue/rotational; do echo "$d: $(cat $d)"; done
powershell.exe -NoProfile -Command "Get-PhysicalDisk | Select-Object FriendlyName,MediaType,BusType,Size | Format-List; Get-Volume | Where-Object DriveLetter | Select-Object DriveLetter,FileSystemType,@{n='SizeGB';e={[math]::Round(\$_.Size/1GB,1)}},@{n='FreeGB';e={[math]::Round(\$_.SizeRemaining/1GB,1)}} | Format-Table -AutoSize"
powershell.exe -NoProfile -Command "Get-ChildItem -Recurse -Filter ext4.vhdx \$env:LOCALAPPDATA\\wsl, \$env:LOCALAPPDATA\\Packages -ErrorAction SilentlyContinue | Select-Object FullName,@{n='GB';e={[math]::Round(\$_.Length/1GB,1)}} | Format-List"

# Prueba de lectura secuencial (1 GiB, archivo temporal borrado al final)
F=/home/edgar/.hw_seqread_test_1g.bin
dd if=/dev/zero of=$F bs=1M count=1024 oflag=direct conv=fsync
dd if=$F of=/dev/null bs=1M iflag=direct
dd if=$F of=/dev/null bs=1M iflag=direct
rm -f $F
```

### 5. GPU
```bash
command -v nvidia-smi && nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,compute_cap --format=csv
command -v lspci && lspci | grep -i -E 'vga|3d|display'
ls -l /dev/dxg
command -v nvcc && nvcc --version
ls /usr/local/cuda*
command -v rocminfo; ls /opt/rocm
powershell.exe -NoProfile -Command "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | Format-List"
```

### 6. Entorno Python
```bash
for p in python3 python python3.10 python3.11 python3.12 python3.13; do command -v $p >/dev/null && echo "$p -> $(command -v $p) : $($p --version 2>&1)"; done
for m in pip pip3 conda mamba uv poetry pipx; do command -v $m >/dev/null && echo "$m -> $(command -v $m) : $($m --version 2>&1 | head -1)"; done
ls -d /home/edgar/projects/*/.venv /home/edgar/projects/.venv /home/edgar/.venv*
uv python list --only-installed
python3 -c "import torch;print('torch',torch.__version__);print('cuda',torch.cuda.is_available());print('mps',torch.backends.mps.is_available())"
/home/edgar/projects/extractor-sap/.venv/bin/python -c "import torch;print(torch.__version__)"
/home/edgar/projects/test-python/.venv/bin/python -c "import torch;print(torch.__version__)"
pip3 show torch
python3 -c "import onnxruntime as o;print(o.__version__, o.get_available_providers())"
python3 -c "import numpy"
```

### 7. Contenedores
```bash
command -v docker && docker --version
docker compose version
docker-compose --version
docker info --format '{{.ServerVersion}} runtimes={{json .Runtimes}}'
docker.exe --version | tr -d '\r'
docker.exe compose version | tr -d '\r'
docker.exe info --format 'Server={{.ServerVersion}} OS={{.OperatingSystem}} Runtimes={{json .Runtimes}} Default={{.DefaultRuntime}}'
command -v nvidia-ctk && nvidia-ctk --version
command -v nvidia-container-cli && nvidia-container-cli --version
dpkg -l | grep -i nvidia-container
cat /etc/docker/daemon.json
```

### 8. Red
```bash
curl -s -o /dev/null -w 'bytes=%{size_download} time=%{time_total}s speed=%{speed_download} B/s http=%{http_code}\n' --max-time 120 'https://speed.cloudflare.com/__down?bytes=50000000'
```
(se ejecutó dos veces)

---

## Veredicto

**Resumen: el entrenamiento local NO es viable. Hay que rentar GPU.** No hay GPU dedicada, no hay CUDA ni ROCm, y la RAM que ve WSL está limitada a 16 GB por `.wslconfig` (de 32 GB físicos). El equipo sí sirve para preparar datos, desarrollar el pipeline, correr inferencia en CPU y orquestar el entrenamiento remoto.

### a) ¿Se puede entrenar EfficientNetV2-S a 384×384 con batch ≥ 16 localmente?

**No de forma práctica.**

- **VRAM disponible: 0 GB.** [M] No hay GPU dedicada. La Radeon 780M es integrada, sin soporte CUDA, y ROCm no está instalado ni soporta esta iGPU en WSL. `torch.cuda.is_available()` daría `False`.
- **Batch máximo por VRAM: 0.** No hay dispositivo de entrenamiento acelerado.
- **Batch máximo en RAM (solo CPU)** [E]: EfficientNetV2-S tiene ~21.5 M parámetros; pesos + gradientes + estados de Adam en fp32 ocupan ~0.35 GB. Las activaciones a 384×384 en fp32 rondan 300 a 400 MB por imagen (estimación a partir del tamaño de los mapas de activación de los bloques MBConv/Fused-MBConv; no se pudo medir porque torch no está instalado). Con el límite medido de 16 GB (~14.5 GB disponibles): (14.5 − 0.35 − ~1 de overhead) / 0.35 ≈ **batch 32 a 36 en fp32**, y hasta ~64 con bf16 (la CPU tiene `avx512_bf16`). Es decir, batch 16 **cabe en RAM**, pero eso solo responde si arranca, no si termina en un tiempo útil (ver b). Si se subiera el límite de `.wslconfig` a 24 GB el batch estimado en fp32 rondaría 56 a 60; es un escenario hipotético, no el estado actual.

### b) ¿Cuántas horas tomaría una época sobre 33,000 imágenes de 384×384?

**Estimación: entre 5 y 10 horas por época en CPU.** [E] Todo lo siguiente es estimado; no hay torch para medirlo. Subir el límite de RAM de WSL no alteraría esta estimación: el cuello de botella es cómputo y ancho de banda, no capacidad.

Razonamiento:

1. **Costo por imagen.** EfficientNetV2-S a 384×384 cuesta ~8.8 GMACs en el forward (paper de Tan & Le, 2021), es decir ~17.6 GFLOPs. Entrenar (forward + backward) cuesta ~3× → **~53 GFLOPs por imagen**. Una época: 33,000 × 53 GFLOPs ≈ **1.75 PFLOP**.
2. **Cómputo disponible.** Zen 4 ejecuta 2 FMA de 256 bits por ciclo por núcleo = 32 FLOPs fp32/ciclo. Con 8 núcleos a ~3.5 a 4.0 GHz sostenidos en laptop (el reloj leído estaba en 2.65 GHz en reposo) el pico teórico es ~0.9 a 1.0 TFLOPS.
3. **Eficiencia real.** EfficientNet está lleno de convoluciones depthwise, SiLU y squeeze-excitation, operaciones limitadas por ancho de banda de memoria, no por cómputo. En CPU con oneDNN se suele alcanzar 5 a 15 % del pico, y aquí hay un solo canal de DDR5, lo que empeora ese punto. Eso da **50 a 150 GFLOPS efectivos**.
4. **Resultado.** 1.75 PFLOP / 150 GFLOPS ≈ 3.2 h en el mejor caso; / 50 GFLOPS ≈ 9.7 h. Rango razonable: **5 a 10 h por época, es decir 1 a 2 imágenes por segundo**. Un entrenamiento típico de 30 épocas serían de 6 a 12 días continuos, con la laptop a plena carga y con throttling térmico probable.

Comparación con GPU rentada [E], AMP bf16/fp16, batch 32 a 64. Fuente: referencia pública, no medido en esta máquina.

| GPU | Imágenes/s entrenando (aprox.) | Tiempo por época de 33,000 | Tipo | Fuente |
|---|---|---|---|---|
| NVIDIA T4 (16 GB) | 60 a 90 | ~7 a 9 min | [E] | referencia externa |
| NVIDIA L4 / A10 (24 GB) | 150 a 250 | ~2.5 a 4 min | [E] | referencia externa |
| NVIDIA A100 / RTX 4090 | 350 a 600 | ~1 a 1.5 min | [E] | referencia externa |

Con 33,000 imágenes y 30 épocas, una L4 termina en unas 2 horas. Todos estos números son de referencia y hay que confirmarlos con un benchmark en la GPU elegida.

### c) ¿El almacenamiento libre alcanza para ~20 GB de dataset más checkpoints de ~500 MB por corrida?

**Sí, con mucho margen.** [M]

- El límite que cuenta es el físico: **767.9 GB libres en C:** (host Windows). Los 948 GB de WSL son el tope nominal del vhdx y no deben usarse como referencia.
- 20 GB de dataset + 500 MB × 100 corridas = 70 GB → **menos del 10 % del espacio libre real**.
- El disco es NVMe con lectura secuencial medida ≥ 3.6 GB/s a través del vhdx, suficiente para alimentar cualquier data loader.
- Advertencia: el vhdx de WSL crece pero no se encoge solo. Si el dataset se borra, el espacio en C: no se recupera hasta compactar el disco virtual.

### d) ¿Puede servir inferencia ONNX en CPU a 384×384 en menos de 1.5 s por imagen?

**Sí, muy probablemente, con un margen de 10× o más.** [E] No se pudo medir porque `onnxruntime` no está instalado.

Razonamiento:

- Inferencia de EfficientNetV2-S a 384: ~17.6 GFLOPs por imagen (solo forward, sin backward).
- El presupuesto de 1.5 s exige solo **~12 GFLOPS efectivos**. Un solo núcleo Zen 4 con AVX-512 supera eso sin esfuerzo, incluso al 10 % de eficiencia.
- Con onnxruntime fp32 en 8 hilos, lo esperable en esta CPU es **~40 a 120 ms por imagen** en batch 1; con cuantización INT8 (la CPU tiene `avx512_vnni`) bajaría a ~20 a 50 ms. Aun con un solo hilo debería quedar por debajo de 0.5 s.
- Requisitos prácticos: los 16 GB de la VM sobran para el modelo (~85 MB en ONNX fp32). Hay que fijar `intra_op_num_threads` ≤ 8 (núcleos físicos) para evitar que el SMT degrade el rendimiento.
- Recomendación: en cuanto exista un venv con onnxruntime, medir con 100 imágenes reales y registrar p50 y p95, no solo el promedio.

### Recomendación final

1. **Entrenar en GPU rentada** (una L4 o A10 con 24 GB es suficiente para EfficientNetV2-S a 384 con batch 32 a 64 en AMP). Preparar el dataset localmente y subirlo una sola vez; medir la velocidad de subida antes, porque no se midió aquí.
2. **Usar esta laptop para** preprocesamiento, aumentación offline, validación del pipeline con un subconjunto pequeño (por ejemplo 200 imágenes a 224×224 en CPU), y para servir inferencia ONNX en CPU, donde sí cumple el objetivo de < 1.5 s.
3. Si se decide probar algo en CPU, instalar torch en un venv con uv y subir `memory=` en `.wslconfig` a 24 o 28 GB seguido de `wsl --shutdown`; ambas cosas están fuera del alcance de este diagnóstico y no se hicieron. Confirmar con `free -h` después del reinicio.
