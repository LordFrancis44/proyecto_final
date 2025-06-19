import os
import json

# 1. Crear __init__.py en src/no_verbal y src/habla
packages = [
    "src/no_verbal",
    "src/habla"
]

for package in packages:
    init_path = os.path.join(package, "__init__.py")
    if not os.path.exists(init_path):
        os.makedirs(package, exist_ok=True)
        with open(init_path, "w") as f:
            f.write("# Este archivo hace que esta carpeta sea un paquete Python\n")
        print(f"✅ Se creó: {init_path}")
    else:
        print(f"✔️ Ya existe: {init_path}")

# 2. Crear carpeta .vscode si no existe
vscode_dir = ".vscode"
os.makedirs(vscode_dir, exist_ok=True)

# 3. Crear o actualizar settings.json con python.analysis.extraPaths
settings_path = os.path.join(vscode_dir, "settings.json")
settings_data = {}

if os.path.exists(settings_path):
    with open(settings_path, "r") as f:
        try:
            settings_data = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ El archivo settings.json no es válido. Se sobrescribirá.")

settings_data["python.analysis.extraPaths"] = ["./src"]

with open(settings_path, "w") as f:
    json.dump(settings_data, f, indent=4)

print(f"✅ Se actualizó {settings_path} con extraPaths")

print("\n🚀 Ahora reinicia VS Code (Reload Window) para que los imports funcionen correctamente.")
