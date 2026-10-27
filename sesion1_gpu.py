#!/usr/bin/env python3
"""
Sesion 1 con GPU (solo espanol e ingles). Fases con reanudacion y registro OK/FALLO:

  caras  : 10 rostros INVENTADOS de adultos (FLUX), uno por area, en /workspace/avatares/sesion1/
  voces  : voz sintetica base de Chatterbox, una frase corta en es y en por avatar (20 audios)
  lipsync: prueba corta con LatentSync en 3 areas (IA, finanzas, negocios) x es/en = 6 videos
  resumen: informe final; nada se da por bueno sin comprobar que el archivo existe y dura algo

Uso:  python3 sesion1_gpu.py [--fases caras,voces,lipsync] [--forzar]
No inventa resultados: si una fase falla, queda FALLO con la causa y el resto continua.
"""
import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

BASE = Path(os.getenv("MOTORES_BASE", "/workspace/projects/sistema_contenido_viral"))
sys.path.insert(0, str(BASE))
SALIDA = Path(os.getenv("SESION1_DIR", "/workspace/avatares/sesion1"))
FLUX_DIR = Path(os.getenv("FLUX_DIR", "/workspace/models_repository/flux"))
LOG = SALIDA / "registro.json"

AVATARES = [
    # id, area, region (solo aspecto), descripcion, seed
    ("av01", "ia_tecnologia", "Mexico", "35-year-old man, short dark hair, friendly smile, casual smart shirt", 1101),
    ("av02", "finanzas", "Estados Unidos", "45-year-old woman, shoulder-length brown hair, blazer, confident calm look", 1102),
    ("av03", "negocios", "Republica Dominicana", "38-year-old man, short curly hair, light beard, open collar shirt", 1103),
    ("av04", "bienes_raices", "Colombia", "42-year-old woman, long dark hair, professional blouse, warm expression", 1104),
    ("av05", "desarrollo_personal", "Chile", "33-year-old woman, tied-back hair, cozy sweater, relaxed expression", 1105),
    ("av06", "bienestar_mayores", "Argentina", "60-year-old man, grey hair, kind face, light cardigan", 1106),
    ("av07", "ciencia", "Espana", "40-year-old woman, short dark bob, glasses, curious expression", 1107),
    ("av08", "programacion", "Peru", "29-year-old man, short hair, hoodie, focused friendly look", 1108),
    ("av09", "viajes_estilo_vida", "Canada", "37-year-old woman, light brown wavy hair, linen shirt, bright smile", 1109),
    ("av10", "cocina_nutricion", "Venezuela", "44-year-old man, short hair, apron over shirt, cheerful look", 1110),
]
FRASES = {
    "es": "Hola, soy un personaje creado con inteligencia artificial. Hoy te explico una idea en menos de un minuto.",
    "en": "Hello, I am a character created with artificial intelligence. Today I will explain one idea in under a minute.",
}
LIPSYNC_AREAS = ["ia_tecnologia", "finanzas", "negocios"]

# Avatares que hablan con la voz del titular (clonada con su consentimiento firmado).
# Solo los de aspecto coherente con esa voz. El resto usa la voz base sintetica.
VOZ_TITULAR = {"av03"}
PROYECTO_TITULAR = Path(os.getenv("PROYECTO_TITULAR", str(BASE / "03_inputs_usuarios/DO-VOMA-001/DO-VOMA-001-2")))
VOZ_TITULAR_WAV = Path(os.getenv("VOZ_TITULAR_WAV", str(PROYECTO_TITULAR / "B" / "voz_titular.wav")))
CONSENT_TXT = Path(os.getenv("CONSENT_TXT", str(PROYECTO_TITULAR / "consentimiento_voz.txt")))


def referencia_consentimiento():
    """Lee 'Referencia: CONSENT-...' del consentimiento firmado. Sin ella no se clona."""
    import re
    if not CONSENT_TXT.exists():
        return None
    m = re.search(r"Referencia:\s*(CONSENT-[A-Z0-9-]+)", CONSENT_TXT.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def ahora():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def cargar_log():
    return json.loads(LOG.read_text()) if LOG.exists() else {"pasos": {}}


def anotar(reg, clave, estado, detalle="", archivo=None):
    reg["pasos"][clave] = {"estado": estado, "detalle": detalle, "archivo": str(archivo) if archivo else None, "ts": ahora()}
    LOG.write_text(json.dumps(reg, indent=2, ensure_ascii=False))
    print(f"[{estado:6}] {clave} {detalle}", flush=True)


def ya_ok(reg, clave, archivo, forzar):
    p = reg["pasos"].get(clave)
    return (not forzar) and p and p["estado"] == "OK" and Path(archivo).exists() and Path(archivo).stat().st_size > 0


def duracion(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def fase_caras(reg, forzar):
    pend = [a for a in AVATARES if not ya_ok(reg, f"cara_{a[0]}", SALIDA / f"{a[0]}.png", forzar)]
    if not pend:
        print("caras: todas listas"); return
    try:
        import torch
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(str(FLUX_DIR), torch_dtype=torch.bfloat16)
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30
        pipe.to("cuda") if vram >= 40 else pipe.enable_model_cpu_offload()
    except Exception as e:  # noqa: BLE001
        for a in pend:
            anotar(reg, f"cara_{a[0]}", "FALLO", f"no se pudo cargar FLUX: {type(e).__name__}: {str(e)[:160]}")
        return
    for id_, area, region, desc, seed in pend:
        archivo = SALIDA / f"{id_}.png"
        prompt = (f"Studio portrait photograph of a {desc}, {region} appearance, looking at the camera, "
                  f"neutral plain soft background, head and shoulders, natural light, photorealistic, sharp focus")
        try:
            g = torch.Generator("cuda").manual_seed(seed)
            img = pipe(prompt, num_inference_steps=4, guidance_scale=0.0, height=768, width=768, generator=g).images[0]
            img.save(archivo)
            ficha = {"id": id_, "area": area, "region_aspecto": region, "seed": seed, "prompt": prompt,
                     "motor": "FLUX.1-schnell", "persona_real": False, "etiqueta": "personaje creado con IA", "adulto": True}
            (SALIDA / f"{id_}.json").write_text(json.dumps(ficha, indent=2, ensure_ascii=False))
            anotar(reg, f"cara_{id_}", "OK", area, archivo)
        except Exception as e:  # noqa: BLE001
            anotar(reg, f"cara_{id_}", "FALLO", f"{type(e).__name__}: {str(e)[:160]}")
    del pipe
    torch.cuda.empty_cache()


def fase_voces(reg, forzar):
    import pp_modulo_17_motores as M17
    for id_, area, *_ in AVATARES:
        for lang, frase in FRASES.items():
            archivo = SALIDA / f"{id_}_{lang}.wav"
            clave = f"voz_{id_}_{lang}"
            if ya_ok(reg, clave, archivo, forzar):
                continue
            if id_ in VOZ_TITULAR:
                ref_id = referencia_consentimiento()
                if not ref_id or not VOZ_TITULAR_WAV.exists():
                    anotar(reg, clave, "FALLO", "falta la muestra de voz del titular o su consentimiento; no se clona")
                    continue
                r = M17.clonar_voz(frase, str(VOZ_TITULAR_WAV), str(archivo), lang, consentimiento_ref=ref_id)
            else:
                r = M17.sintetizar_voz_base(frase, str(archivo), lang)
            if r["estado"] == "OK" and duracion(archivo) > 1.0:
                anotar(reg, clave, "OK", f"{duracion(archivo):.1f}s", archivo)
            else:
                anotar(reg, clave, "FALLO", r.get("detalle", r["estado"]))


def fase_lipsync(reg, forzar):
    import pp_modulo_18_video as M18
    for id_, area, *_ in AVATARES:
        if area not in LIPSYNC_AREAS:
            continue
        cara = SALIDA / f"{id_}.png"
        for lang in FRASES:
            audio = SALIDA / f"{id_}_{lang}.wav"
            archivo = SALIDA / f"{id_}_{lang}.mp4"
            clave = f"lipsync_{id_}_{lang}"
            if ya_ok(reg, clave, archivo, forzar):
                continue
            if not cara.exists() or not audio.exists():
                anotar(reg, clave, "OMITIDO", "falta la cara o la voz de este avatar")
                continue
            r = M18.lipsync_latentsync(str(cara), str(audio), str(archivo), es_persona_real=False)
            if r["estado"] == "OK" and duracion(archivo) > 1.0:
                anotar(reg, clave, "OK", f"{duracion(archivo):.1f}s", archivo)
            else:
                anotar(reg, clave, "FALLO", (r.get("detalle") or r.get("stderr_tail") or r["estado"])[-300:])


def resumen(reg):
    c = {"OK": 0, "FALLO": 0, "OMITIDO": 0}
    for p in reg["pasos"].values():
        c[p["estado"]] = c.get(p["estado"], 0) + 1
    print(f"\nRESUMEN  OK {c['OK']} | FALLO {c['FALLO']} | OMITIDO {c['OMITIDO']}   (detalle: {LOG})")
    for k, p in reg["pasos"].items():
        if p["estado"] != "OK":
            print(f"  {p['estado']}: {k} -> {p['detalle']}")
    return c["FALLO"] == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fases", default="caras,voces,lipsync")
    ap.add_argument("--forzar", action="store_true")
    a = ap.parse_args()
    SALIDA.mkdir(parents=True, exist_ok=True)
    reg = cargar_log()
    fases = [f.strip() for f in a.fases.split(",")]
    if "caras" in fases:
        fase_caras(reg, a.forzar)
    if "voces" in fases:
        fase_voces(reg, a.forzar)
    if "lipsync" in fases:
        fase_lipsync(reg, a.forzar)
    sys.exit(0 if resumen(reg) else 1)


if __name__ == "__main__":
    main()
