## Instalación de paquetes
Los paquetes necessarios se encuentran listados en el archivo *requirements.txt*.
- Crear el entorno virtual: `python3.10 -m venv venv`
- Activar el entorno virtual: `source venv/bin/activate`
- Upgrade pip: `python3.10 -m pip install --upgrade pip`
- Instalar paquetes desde *requirements.txt*: `pip install -r requirements.txt`
- Desinstalar todos los paquetes: `pip freeze | xargs pip uninstall -y`
- Desactivar el entorno virtual: `deactivate`
- Eliminar le entorno virtual (opcional): `rm -rf venv`