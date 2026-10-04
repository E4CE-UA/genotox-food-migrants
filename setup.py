"""Copy canonical data/report assets into the installed private package."""
from pathlib import Path
from shutil import copy2
from setuptools import setup
from setuptools.command.build_py import build_py

class BuildWithAssets(build_py):
    def run(self):
        super().run()
        root = Path(__file__).parent
        for source, target in [('data', '_data'), ('docs', '_validation')]:
            for path in (root / source).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts:
                    destination = Path(self.build_lib) / 'genotox_food_migrants' / target / path.relative_to(root / source)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    copy2(path, destination)

setup(cmdclass={'build_py': BuildWithAssets})
