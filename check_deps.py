import tomllib
import importlib
import sys
from pathlib import Path

def get_import_name(package_name):
    """
    Mapea el nombre del paquete PyPI al nombre de importación de Python.
    """
    mapping = {
        "python-dotenv"          : "dotenv",
        "scikit-learn"           : "sklearn",
        "matplotlib"             : "matplotlib",
        "seaborn"                : "seaborn",
        "statsmodels"            : "statsmodels",
        "huggingface-hub"        : "huggingface_hub",
        "pillow"                 : "PIL",
        "opencv-python-headless" : "cv2"  # << Añadido
    }
    return mapping.get(package_name.lower(), package_name.lower().replace('-', '_'))

def check_dependencies():
    """
    Lee pyproject.toml usando la biblioteca estándar (tomllib) y verifica las dependencias.
    Propone comandos de corrección si encuentra errores.
    """
    pyproject_path = Path("pyproject.toml")
    
    if not pyproject_path.exists():
        print("❌ Error: No se encontró el archivo 'pyproject.toml'.")
        sys.exit(1)

    try:
        with open(pyproject_path, 'rb') as f:
            data = tomllib.load(f)
        
        # 1. Extraer y clasificar todas las dependencias
        deps_prod = data.get('tool', {}).get('poetry', {}).get('dependencies', {})
        deps_dev = data.get('tool', {}).get('poetry', {}).get('group', {}).get('dev', {}).get('dependencies', {})
        
        all_dependencies = {}
        for pkg, ver in deps_prod.items():
            if pkg != 'python':
                all_dependencies[pkg] = {'version': ver, 'group': 'prod'}
        for pkg, ver in deps_dev.items():
            all_dependencies[pkg] = {'version': ver, 'group': 'dev'}

        print(f"🔬 Se encontraron {len(all_dependencies)} paquetes para verificar.")
        print("="*60)
        
        all_passed = True
        failed_packages = []
        
        for package_name, details in all_dependencies.items():
            import_name = get_import_name(package_name)
            group = details['group']
            
            print(f"▶️ Verificando: {package_name} (Grupo: {group}, Importar como: '{import_name}')")
            
            try:
                # Intentar importar la librería
                module = importlib.import_module(import_name)
                
                version_info = getattr(module, '__version__', 'N/A')
                print(f"   ✅ Éxito. Versión detectada: {version_info}")
                
            except ImportError:
                all_passed = False
                failed_packages.append((package_name, group))
                
                # Proponer el comando de corrección
                if group == 'prod':
                    fix_command = f"poetry add {package_name}"
                else:
                    fix_command = f"poetry add {package_name} --group {group}"
                
                print(f"   ❌ ERROR: La librería '{import_name}' NO está instalada en el entorno.")
                print(f"   👉 Comando sugerido: {fix_command}")
                
            except Exception as e:
                print(f"   ⚠️ ADVERTENCIA: Importación exitosa, pero error al intentar acceder a la versión: {e}")
                
            print("-" * 60)

        if all_passed:
            print("\n🎉 ¡Todas las dependencias se importaron correctamente!")
        else:
            print("\n❌ Advertencia: ¡Una o más dependencias fallaron la verificación!")
            print("=============================================================")
            print("🚀 Recomendación Principal:")
            print("Dado que fallaron varios paquetes, el entorno no está sincronizado.")
            print("Ejecute el siguiente comando para instalar todos los faltantes:")
            print("\n    poetry install\n")
            print("=============================================================")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error fatal al procesar pyproject.toml: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_dependencies()