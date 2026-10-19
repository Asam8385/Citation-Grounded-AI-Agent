from datetime import date
from pathlib import PurePosixPath
from typing import Self


from pydantic import BaseModel , ConfigDict , Field , field_validator , model_validator

def i