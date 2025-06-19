import os
import sys
import json
import time
import logging
import numpy as np
from multiprocessing import Pool

# --- CONFIGURACIÓN DE RUTAS ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
sys.path.append(SRC_DIR)

try:
    from no_verbal.analysis_logic import run_full_analysis as run_non_verbal_analysis
    from habla.speech_logic import run_speech_analysis
except ImportError as e:
    print(f"Error: No se pudo importar un módulo de análisis: {e}")
    sys.exit(1)

TASKS_DIR = os.path.join(DATA_DIR, "tasks")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(TASKS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - WORKER - %(levelname)s - %(message)s')

def convert_to_native_types(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float16, np.float32)):
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_native_types(obj.tolist())
    return obj

def run_analysis_task(task_args):
    analysis_function, url = task_args
    return analysis_function(url)

def process_task(task_filepath):
    task_filename = os.path.basename(task_filepath)
    job_id = None
    start_time = time.time()
    try:
        logging.info(f"📥 Nueva tarea encontrada: {task_filename}")
        with open(task_filepath, 'r') as f:
            task_data = json.load(f)

        url = task_data['url']
        job_id = task_data['job_id']

        logging.info(f"Lanzando análisis en paralelo para el job_id: {job_id}")

        tasks_to_run = [
            (run_non_verbal_analysis, url),
            (run_speech_analysis, url)
        ]

        with Pool(processes=2) as pool:
            results_list = pool.map(run_analysis_task, tasks_to_run)

        final_results = {}
        for res in results_list:
            final_results.update(res)

        end_time = time.time()
        duration_seconds = end_time - start_time

        final_results["processing_metadata"] = {
            "duration_seconds": duration_seconds
        }

        cleaned_results = convert_to_native_types(final_results)

        logging.info(f"Todos los análisis para {job_id} completados con éxito.")
        result_filepath = os.path.join(RESULTS_DIR, f"{job_id}.json")

        with open(result_filepath, 'w') as f:
            json.dump(cleaned_results, f, indent=4)

        logging.info(f"✅ Resultado fusionado para {job_id} guardado.")

    except Exception as e:
        logging.error(f"❌ Error procesando la tarea {task_filename}: {e}", exc_info=True)
        if job_id:
            error_filepath = os.path.join(RESULTS_DIR, f"{job_id}.error")
            with open(error_filepath, 'w') as f:
                f.write(str(e))
            logging.warning(f"Archivo de error para {job_id} guardado.")

    finally:
        logging.info(f"Limpiando tarea: {task_filename}")
        if os.path.exists(task_filepath):
            os.remove(task_filepath)

def main():
    logging.info("🔥 Worker iniciado. Esperando tareas...")
    while True:
        try:
            tasks = sorted(
                [os.path.join(TASKS_DIR, f) for f in os.listdir(TASKS_DIR) if f.endswith(".json")],
                key=os.path.getmtime
            )
            if not tasks:
                time.sleep(5)
                continue
            task_filepath = tasks[0]
            process_task(task_filepath)
        except KeyboardInterrupt:
            logging.info("🚫 Worker detenido por el usuario.")
            break
        except Exception as e:
            logging.critical(f"Ha ocurrido un error inesperado en el bucle principal del worker: {e}", exc_info=True)
            time.sleep(10)

if __name__ == "__main__":
    main()
