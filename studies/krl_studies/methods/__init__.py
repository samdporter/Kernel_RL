from krl_studies.methods.base import Iterate, Method
from krl_studies.methods.baselines import PostSmoothingMethod
from krl_studies.methods.dtv import DTVMethod
from krl_studies.methods.iterative_yang import IterativeYangMethod
from krl_studies.methods.petpvc import GTMMethod
from krl_studies.methods.richardson_lucy import HKRLMethod, KRLMethod, RLMethod

__all__ = [
    "METHOD_REGISTRY",
    "DTVMethod",
    "GTMMethod",
    "HKRLMethod",
    "Iterate",
    "IterativeYangMethod",
    "KRLMethod",
    "Method",
    "PostSmoothingMethod",
    "RLMethod",
]

METHOD_REGISTRY = {
    cls.name: cls for cls in (
        RLMethod, KRLMethod, HKRLMethod, DTVMethod,
        IterativeYangMethod, PostSmoothingMethod, GTMMethod,
    )
}
