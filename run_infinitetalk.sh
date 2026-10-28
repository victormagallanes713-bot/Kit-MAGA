#!/usr/bin/env bash
# Uso: run_infinitetalk.sh <foto> <audio> <salida.mp4>
# Genera un video donde la persona habla con movimiento de cabeza, cuerpo y expresiones (InfiniteTalk).
# Por defecto usa la version fp8 con descarga a CPU para caber en 24 GB (RTX 4090).
# Variables opcionales: IT_STEPS (40), IT_SIZE (infinitetalk-480), IT_PROMPT, IT_QUANT (fp8|no)
#   IT_OFFLOAD = auto (defecto) | gpu | cpu
#     auto: GPU primero. Con 40 GB o mas de VRAM el modelo vive entero en la GPU (rapido);
#           con menos, usa la RAM como apoyo (lento). Si en modo gpu falta memoria, reintenta solo con cpu.
set -euo pipefail
export PATH="${INFINITETALK_VENV_BIN:-/workspace/venv_infinitetalk/bin}:$PATH"
FOTO="${1:?falta la foto}"; AUDIO="${2:?falta el audio}"; SALIDA="${3:?falta la salida}"
FOTO="$(realpath "$FOTO")"; AUDIO="$(realpath "$AUDIO")"; SALIDA="$(realpath -m "$SALIDA")"
IT_DIR="${INFINITETALK_DIR:-/workspace/InfiniteTalk}"
PY="${INFINITETALK_PY:-/workspace/venv_infinitetalk/bin/python}"
W="${INFINITETALK_WEIGHTS:-/workspace/models_repository/infinitetalk}"
STEPS="${IT_STEPS:-40}"; SIZE="${IT_SIZE:-infinitetalk-480}"; QUANT="${IT_QUANT:-fp8}"
PROMPT="${IT_PROMPT:-A person talking naturally to the camera, with natural head movements, blinking, subtle body motion and expressive hand gestures while speaking.}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

[ -f "$IT_DIR/generate_infinitetalk.py" ] || { echo "InfiniteTalk no esta instalado en $IT_DIR (bash instalar_infinitetalk.sh instalar)" >&2; exit 2; }
[ -x "$PY" ] || { echo "falta el entorno $PY" >&2; exit 2; }
for f in "$W/Wan2.1-I2V-14B-480P" "$W/chinese-wav2vec2-base/model.safetensors" "$W/InfiniteTalk/single/infinitetalk.safetensors"; do
  [ -e "$f" ] || { echo "falta $f (bash instalar_infinitetalk.sh pesos)" >&2; exit 2; }
done
[ "$QUANT" = "fp8" ] && [ ! -s "$W/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors" ] && { echo "falta el modelo fp8" >&2; exit 2; }

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
EJ="$IT_DIR/examples/single_example_image.json"
[ -f "$EJ" ] || { echo "no encuentro $EJ para copiar su formato" >&2; exit 4; }
# El JSON de entrada se arma a partir del ejemplo del repositorio; si su formato cambio, se avisa en vez de adivinar.
"$PY" - "$EJ" "$FOTO" "$AUDIO" "$PROMPT" "$TMP/entrada.json" <<'EOF'
import json, sys
ej, foto, audio, prompt, out = sys.argv[1:6]
d = json.load(open(ej))
if not isinstance(d, dict) or "cond_video" not in d or not isinstance(d.get("cond_audio"), dict) or not d["cond_audio"]:
    print("El formato del ejemplo no es el esperado. Contenido:\n" + json.dumps(d, indent=2)[:800], file=sys.stderr); sys.exit(4)
d["cond_video"] = foto
k = next(iter(d["cond_audio"]))
d["cond_audio"] = {k: audio}
if "prompt" in d: d["prompt"] = prompt
json.dump(d, open(out, "w"), indent=2)
EOF

BASE_ARGS=(--ckpt_dir "$W/Wan2.1-I2V-14B-480P" --wav2vec_dir "$W/chinese-wav2vec2-base"
      --infinitetalk_dir "$W/InfiniteTalk/single/infinitetalk.safetensors"
      --input_json "$TMP/entrada.json" --size "$SIZE" --sample_steps "$STEPS"
      --mode streaming --motion_frame 9 --save_file "$TMP/resultado")
[ "$QUANT" = "fp8" ] && BASE_ARGS+=(--quant fp8 --quant_dir "$W/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors")

OFFLOAD="${IT_OFFLOAD:-auto}"
if [ "$OFFLOAD" = "auto" ]; then
  VRAM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -dc '0-9')"
  if [ -n "$VRAM" ] && [ "$VRAM" -ge 40000 ]; then OFFLOAD=gpu; else OFFLOAD=cpu; fi
  echo "GPU con ${VRAM:-?} MiB de VRAM -> modo $OFFLOAD" >&2
fi

correr() {  # $1 = gpu|cpu ; muestra la salida en vivo y la guarda para detectar falta de memoria
  local modo="$1" extra=()
  [ "$modo" = "cpu" ] && extra=(--num_persistent_param_in_dit 0)
  rm -f "$TMP"/resultado*.mp4 "$IT_DIR"/resultado*.mp4 2>/dev/null || true
  local t0=$SECONDS rc
  set +e
  ( cd "$IT_DIR" && "$PY" generate_infinitetalk.py "${BASE_ARGS[@]}" "${extra[@]}" ) 2>&1 | tee "$TMP/salida_$modo.log"
  rc=${PIPESTATUS[0]}
  set -e
  echo "TIEMPO modo=$modo segundos=$((SECONDS - t0)) codigo=$rc" >&2
  return $rc
}

if ! correr "$OFFLOAD"; then
  if [ "$OFFLOAD" = "gpu" ] && grep -qiE "out of memory|OutOfMemory" "$TMP/salida_gpu.log"; then
    echo "AVISO: falta memoria en modo gpu; reintentando con descarga a RAM (mas lento)" >&2
    correr cpu || { echo "InfiniteTalk fallo tambien en modo cpu" >&2; exit 3; }
  else
    echo "InfiniteTalk fallo (ver arriba)" >&2; exit 3
  fi
fi
OUT="$(ls "$TMP"/resultado*.mp4 "$IT_DIR"/resultado*.mp4 2>/dev/null | head -n 1 || true)"
[ -n "$OUT" ] && [ -s "$OUT" ] || { echo "InfiniteTalk no produjo ningun video" >&2; exit 3; }
mkdir -p "$(dirname "$SALIDA")"; mv "$OUT" "$SALIDA"
[ -s "$SALIDA" ] || { echo "no se pudo escribir $SALIDA" >&2; exit 3; }
echo "OK $SALIDA"
