import os
import sys
import json
import time
import logging
import numpy as np
from multiprocessing import Pool # Importar la herramienta para paralelizar

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

# --- Función de conversión de tipos (sin cambios) ---
def convert_to_native_types(obj):
    """
    Función recursiva para convertir todos los tipos de NumPy en un objeto
    (diccionario, lista) a tipos nativos de Python que JSON pueda entender.
    """
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float16, np.float32, np.float64)): # He vuelto a añadir np.float_ por si acaso
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_native_types(obj.tolist())
    return obj

# --- INICIO DE LA MODIFICACIÓN: Función helper para el pool ---
# Esta función es necesaria para que el Pool de multiprocessing pueda llamar
# a nuestras funciones de análisis con sus argumentos.
def run_analysis_task(task_args):
    """Desempaqueta los argumentos y ejecuta la función de análisis correspondiente."""
    analysis_function, url = task_args
    return analysis_function(url)
# --- FIN DE LA MODIFICACIÓN ---

def process_task(task_filepath):
    task_filename = os.path.basename(task_filepath)
    job_id = None
    try:
        logging.info(f"📥 Nueva tarea encontrada: {task_filename}")
        with open(task_filepath, 'r') as f:
            task_data = json.load(f)
        
        url = task_data['url']
        job_id = task_data['job_id']

        # --- INICIO DE LA MODIFICACIÓN: Ejecución en paralelo ---
        logging.info(f"Lanzando análisis en paralelo para el job_id: {job_id}")

        # 1. Definir la lista de tareas que se ejecutarán en paralelo.
        # Cada tupla contiene la función a ejecutar y su argumento (la URL).
        tasks_to_run = [
            (run_non_verbal_analysis, url),
            (run_speech_analysis, url)
        ]

        # 2. Crear un "pool" de 2 procesos trabajadores.
        # El bloque 'with' asegura que los procesos se cierren correctamente.
        with Pool(processes=2) as pool:
            # 3. Mapear las tareas al pool. La función 'map' distribuye las tareas
            # entre los procesos disponibles y espera a que todas terminen.
            # 'results_list' contendrá una lista con los diccionarios de resultados,
            # uno por cada análisis.
            results_list = pool.map(run_analysis_task, tasks_to_run)
        
        # 4. Fusionar los resultados de la lista en un solo diccionario final.
        final_results = {}
        for res in results_list:
            final_results.update(res)
        # --- FIN DE LA MODIFICACIÓN ---
        
        # El resto del proceso se mantiene idéntico.
        # 2. "Limpiar" el diccionario de tipos de NumPy antes de guardar
        cleaned_results = convert_to_native_types(final_results)
        
        logging.info(f"Todos los análisis para {job_id} completados con éxito.")
        result_filepath = os.path.join(RESULTS_DIR, f"{job_id}.json")
        
        # 3. Guardar el diccionario ya limpio.
        with open(result_filepath, 'w') as f:
            json.dump(cleaned_results, f, indent=4)
        
        logging.info(f"✅ Resultado fusionado para {job_id} guardado.")

    except Exception as e:
        logging.error(f"❌ Error procesando la tarea {task_filename}: {e}", exc_info=True)
        if job_id:
            error_filepath = os.path.join(RESULTS_DIR, f"{job_id}.error")
            with open(error_filepath, 'w') as f: f.write(str(e))
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
            logging.info("🛑 Worker detenido por el usuario.")
            break
        except Exception as e:
            logging.critical(f"Ha ocurrido un error inesperado en el bucle principal del worker: {e}", exc_info=True)
            time.sleep(10)

if __name__ == "__main__":
    # Esta comprobación es crucial para que multiprocessing funcione correctamente
    # en algunos sistemas operativos como Windows.
    main()