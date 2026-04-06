# -*- coding:utf-8 -*-

import importlib.util
import os


_base_dir = os.path.dirname(os.path.abspath(__file__))
_prompt_file = os.path.join(_base_dir, "issue_detection", "prompt_temp_revise.py")
_spec = importlib.util.spec_from_file_location("_prompt_temp_revise", _prompt_file)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

for _name in dir(_module):
    if _name.startswith("PROMPT_"):
        globals()[_name] = getattr(_module, _name)
