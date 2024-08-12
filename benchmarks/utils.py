import os

CI_MODE = bool(int(os.getenv("CI_MODE", default="0")))
