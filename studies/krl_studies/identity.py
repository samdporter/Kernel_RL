"""Canonical campaign identities: forward, input, and run.

The three identities form a refinement hierarchy::

    forward_id  subset  input_id  subset  run_id

Each id is the sha256 digest of the sorted-key compact JSON of a canonical
dict, prefixed with a short readable tag. The full canonical dicts are
available from :func:`forward_identity`, :func:`input_identity`, and
:func:`run_identity` for recording in per-run manifests. Plan rows store the
short ids plus the resolved configuration, from which the dicts are recomputed
on read.

The split follows the campaign contract:

* ``forward_id`` covers the physical forward model (PET/uMap checksums, truth
  PSF, scanner/projection/image geometry, physical transforms and the
  simulation implementation/environment). It must NOT change when guidance,
  method, assumed PSF, counts or realisation change.
* ``input_id`` adds counts, the deterministic noise seed, the reconstruction
  settings (recon-PSF condition, RDP beta, OSEM settings/initialisation), and
  the output-grid contract. It must NOT change with guidance, method or the
  method's assumed PSF.
* ``run_id`` adds the effective method parameters, guidance
  source/preprocessing, stopping policy and protocol/implementation identity.
  It excludes output location and timestamps.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

ID_SCHEMA_VERSION = "campaign_identity_v1"

GUIDANCE_MODALITIES = ("t1", "t2", "umap")
GUIDANCE_CONDITIONS = ("exact", "shift_p2", "shift_m2", "shift_p5", "shift_m5")
GUIDANCE_LESION_STATES = ("absent", "present")
DEFAULT_GUIDANCE_LESION_STATE = "absent"

# Methods that actually consume anatomy guidance. Only these carry the T1
# lesion-state identity; RL/post-smoothing (and non-guided inputs) never see it.
GUIDED_METHODS = frozenset({"krl", "hkrl", "dtv"})

# Legacy ``guidance_condition: t2`` means "use the T2 anatomy with exact
# alignment"; the new contract spells that as (modality=t2, condition=exact).
LEGACY_T2_CONDITION = "t2"

STAGES = ("development", "evaluation")

# Documented fallbacks used only when a scenario predates the campaign block.
# They preserve loading of the existing spheres*/brainweb*/patient* YAMLs.
LEGACY_SCANNER = "Siemens mMR"
LEGACY_TRUTH_FWHM_MM = 5.0
LEGACY_OSEM_SUBSETS = 21  # _SUBSET_CANDIDATES[0] for the reduced 42-view geometry
LEGACY_OSEM_FULL_ITERATIONS = 1
LEGACY_OSEM_SUBITERATIONS = 1
LEGACY_PROTOCOL_VERSION = "legacy"
DEFAULT_STAGE = "development"
DEFAULT_SELECTION_POLICY = "oracle_min_nrmse"

NOISE_REALISATION_STRIDE = 7919

# Legacy -> new guidance mapping, applied on config load:
#   exact                     -> (guidance_modality=t1,   condition=exact)
#   t2                        -> (guidance_modality=t2,   condition=exact)
#   shift_p2/m2/p5/m5         -> (guidance_modality=t1,   condition=<shift>)
# An explicit ``guidance_modality`` overrides the legacy t1 default; the only
# conflict is ``guidance_condition: t2`` combined with a non-t2 modality.
LEGACY_GUIDANCE_MAPPING = {
    "exact": ("t1", "exact"),
    "t2": ("t2", "exact"),
    "shift_p2": ("t1", "shift_p2"),
    "shift_m2": ("t1", "shift_m2"),
    "shift_p5": ("t1", "shift_p5"),
    "shift_m5": ("t1", "shift_m5"),
}

# Static, stdlib-only per-method runtime defaults. These mirror the ``.get``
# fallbacks in ``krl_studies.methods.*``, ``runner/execute.py`` and the core
# plugin ``krl.operators.kernel_operator.DEFAULT_PARAMETERS``. They are the
# single source of truth for effective-parameter identity; T5 must make the
# runtime consume this registry so identities cannot drift. Importing the
# method/plugin modules is deliberately avoided so dry-run expansion stays free
# of CIL/SIRF.
_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))

# Hard-coded copy of ``krl.operators.kernel_operator.DEFAULT_PARAMETERS``
# (src/krl/operators/kernel_operator.py). FLAGGED FOR T5: keep in sync.
KERNEL_DEFAULTS: dict[str, Any] = {
    "num_neighbours": 5,
    "sigma_anat": 0.1,
    "sigma_dist": 10000,
    "sigma_emission": 0.1,
    "normalize_features": True,
    "normalize_kernel": True,
    "use_mask": True,
    "mask_k": 20,
    "recalc_mask": False,
    "distance_weighting": False,
    "hybrid": False,
}

METHOD_DEFAULTS: dict[str, dict[str, Any]] = {
    "rl": {"backend": "numba", "epsilon": 1e-10},
    "krl": {**KERNEL_DEFAULTS, "backend": "numba", "epsilon": 1e-10, "freeze_iteration": 0},
    "hkrl": {
        **KERNEL_DEFAULTS,
        "backend": "numba",
        "epsilon": 1e-10,
        "freeze_iteration": 1,
    },
    "dtv": {
        "backend": "numba",
        "lbfgs_max_linesearch": 20,
        "lbfgs_ftol": 1e-6,
        "lbfgs_gtol": 1e-6,
    },
    # iY's psf_sigma_vox is derived in runner/execute.py from fwhm_mm (5.0
    # fallback); resolve_method_params expands it after merging.
    "iy": {"damping": 1.0, "fwhm_mm": 5.0},
    "post_smoothing": {"voxel_mm": (1.0, 1.0, 1.0)},
    "gtm": {"petpvc_bin": "petpvc", "pvc_fwhm": (5.0, 5.0, 5.0)},
}

_CHECKSUM_CACHE: dict[tuple[str, int, int], str] = {}


@dataclass(frozen=True)
class ForwardConfig:
    """Physical forward-model settings (truth side of the acquisition)."""

    truth_fwhm_mm: float
    scanner: str
    attenuation_path: str | None = None
    projection_geometry: dict[str, Any] = field(default_factory=dict)
    image_geometry: dict[str, Any] = field(default_factory=dict)
    physical_transforms: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class OsemConfig:
    """Reconstruction settings that define the observed input."""

    subsets: int
    full_iterations: int
    subiterations: int
    initialisation: str = "ones"


@dataclass(frozen=True)
class IdentityConfig:
    """Fully resolved configuration from which the three identities are built."""

    study: str
    subject: Any
    input_kind: str
    forward: ForwardConfig
    counts: float | None
    realisation: int
    base_seed: int
    osem: OsemConfig
    condition: str | None
    beta: float | None
    output_grid: dict[str, Any]
    method_name: str
    method_params: dict[str, Any]
    iterations: int
    guidance_modality: str
    guidance_condition: str
    guidance_lesion_state: str
    guidance_lesion: dict[str, Any]
    guidance_preprocessing: dict[str, Any]
    protocol_version: str
    stage: str
    selection_policy: str
    pet_checksum: str
    attenuation_checksum: str | None
    object_perturbations: dict[str, Any] = field(default_factory=dict)

    @property
    def noise_seed(self) -> int:
        return int(self.base_seed) + int(self.realisation) * NOISE_REALISATION_STRIDE


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _slug(value: Any) -> str:
    text = re.sub(r"[^0-9A-Za-z_-]+", "-", str(value)).strip("-")
    return text or "na"


def canonical_json(obj: Any) -> str:
    """Sorted-key compact JSON, the canonical form every id hashes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def identity_digest(identity: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()


def short_id(prefix: str, identity: dict[str, Any]) -> str:
    return f"{prefix}_{identity_digest(identity)[:16]}"


def _require_scalar_fwhm(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{where}: truth_fwhm_mm must be an explicit scalar number, got {value!r}"
        )
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(
            f"{where}: truth_fwhm_mm must be a positive finite number, got {value!r}"
        )
    return number


def validate_stage(stage: str) -> str:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    return stage


def validate_guidance_modality(modality: str) -> str:
    if modality not in GUIDANCE_MODALITIES:
        raise ValueError(
            f"guidance_modality must be one of {GUIDANCE_MODALITIES}, got {modality!r}"
        )
    return modality


def validate_guidance_condition(condition: str) -> str:
    if condition not in GUIDANCE_CONDITIONS:
        raise ValueError(
            f"guidance_condition must be one of {GUIDANCE_CONDITIONS}, got {condition!r}"
        )
    return condition


def resolve_guidance_lesion_state(params: dict[str, Any], where: str = "input") -> str:
    """Return the validated T1 lesion state (``absent``/``present``)."""
    lesion_state = str(params.get("guidance_lesion_state", DEFAULT_GUIDANCE_LESION_STATE))
    if lesion_state not in GUIDANCE_LESION_STATES:
        raise ValueError(
            f"{where}: guidance_lesion_state must be one of {GUIDANCE_LESION_STATES}, "
            f"got {lesion_state!r}"
        )
    return lesion_state


def resolve_guidance(params: dict[str, Any], where: str = "input") -> tuple[str, str]:
    """Lift legacy guidance fields into ``(modality, condition)``.

    Raises ``ValueError`` on unknown values or conflicting modality/condition.
    """
    modality = params.get("guidance_modality")
    condition = params.get("guidance_condition")

    if condition == LEGACY_T2_CONDITION:
        if modality is not None and modality != "t2":
            raise ValueError(
                f"{where}: conflicting guidance (legacy condition 't2' but "
                f"guidance_modality={modality!r})"
            )
        return ("t2", "exact")

    if condition is None:
        condition = "exact"
    if modality is None:
        modality = "t1"
    validate_guidance_modality(modality)
    validate_guidance_condition(condition)
    return (modality, condition)


def _canonical_number(value: Any) -> Any:
    """Normalise numeric scalars so ``5.0e7``, ``5e7`` and ``50000000.0`` hash alike.

    PyYAML resolves some scientific-notation literals to strings; canonicalising
    here keeps the identity stable across equivalent physical values.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    return _canonical_number(value)


def resolve_method_params(
    method_name: str, supplied: dict[str, Any] | None
) -> tuple[dict[str, Any], int]:
    """Return ``(effective_params, iterations)`` for a method.

    Effective params are the static per-method defaults (wrapper, kernel and
    derived iY PSF) overridden by the supplied params. ``iterations`` is split
    out as the stopping policy so an omitted value and an explicit default
    value hash identically.
    """
    name = str(method_name)
    given = dict(supplied or {})
    effective = {**METHOD_DEFAULTS.get(name, {}), **given}

    if name == "iy" and "psf_sigma_vox" not in effective:
        # runner/execute.py derives iY's PSF from fwhm_mm (5.0 fallback).
        fwhm = float(effective.get("fwhm_mm", 5.0))
        effective["psf_sigma_vox"] = tuple(fwhm * _FWHM_TO_SIGMA for _ in range(3))
    if name == "hkrl":
        # HKRL decides hybrid mode from the SUPPLIED sigma_emission (its own
        # 0.0 fallback), overriding the kernel default of 0.1.
        effective["hybrid"] = float(given.get("sigma_emission", 0.0)) > 0

    iterations = int(effective.pop("iterations", 1))
    return _canonicalize(effective), iterations


def _validate_counts(value: Any, where: str) -> int | float:
    if value is None:
        raise ValueError(f"{where}: counts must be a positive finite number, got None")
    canonical = _canonical_number(value)
    if isinstance(canonical, bool) or not isinstance(canonical, (int, float)):
        raise ValueError(f"{where}: counts must be a positive finite number, got {value!r}")
    number = float(canonical)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{where}: counts must be a positive finite number, got {value!r}")
    return canonical


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} must be an integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"{where} must be positive, got {value!r}")
    return int(value)


def resolve_osem(
    params: dict[str, Any],
    sim: dict[str, Any],
    scenario_osem: dict[str, Any] | None = None,
    where: str = "input",
) -> OsemConfig:
    """Resolve OSEM settings from an explicit block or legacy ``n_subits``."""
    block = params.get("osem", scenario_osem)
    if block is not None:
        if not isinstance(block, dict):
            raise ValueError(f"{where}: osem must be a mapping, got {block!r}")
        if "subsets" not in block or "full_iterations" not in block:
            raise ValueError(
                f"{where}: osem requires explicit 'subsets' and 'full_iterations'"
            )
        subsets = _positive_int(block["subsets"], f"{where}.osem.subsets")
        full_iterations = _positive_int(block["full_iterations"], f"{where}.osem.full_iterations")
        expected_subiterations = subsets * full_iterations
        if "subiterations" in block:
            subiterations = _positive_int(block["subiterations"], f"{where}.osem.subiterations")
            if subiterations != expected_subiterations:
                raise ValueError(
                    f"{where}: osem.subiterations={subiterations} conflicts with "
                    f"subsets*full_iterations={expected_subiterations}"
                )
        else:
            subiterations = expected_subiterations
        initialisation = str(block.get("initialisation", "ones"))
        return OsemConfig(subsets, full_iterations, subiterations, initialisation)

    n_subits = params.get("n_subits", params.get("n_subiterations"))
    if n_subits is None:
        n_subits = sim.get("n_subits", sim.get("n_subiterations", LEGACY_OSEM_SUBITERATIONS))
    subiterations = _positive_int(n_subits, f"{where}.subiterations")
    subsets = _positive_int(params.get("subsets", LEGACY_OSEM_SUBSETS), f"{where}.subsets")
    full_iterations = _positive_int(
        params.get("full_iterations", LEGACY_OSEM_FULL_ITERATIONS), f"{where}.full_iterations"
    )
    return OsemConfig(subsets, full_iterations, subiterations, "ones")


def resolve_forward(
    forward_block: dict[str, Any] | None,
    params: dict[str, Any],
    sim: dict[str, Any],
    where: str = "scenario",
) -> ForwardConfig:
    """Resolve the forward-model block, or derive a legacy approximation."""
    if forward_block is not None:
        if not isinstance(forward_block, dict):
            raise ValueError(f"{where}: forward must be a mapping, got {forward_block!r}")
        if "truth_fwhm_mm" not in forward_block:
            raise ValueError(f"{where}: forward requires explicit 'truth_fwhm_mm'")
        truth = _require_scalar_fwhm(forward_block["truth_fwhm_mm"], where)
        if "scanner" not in forward_block:
            raise ValueError(f"{where}: forward requires explicit 'scanner'")
        for key in ("projection_geometry", "image_geometry"):
            if key not in forward_block:
                raise ValueError(f"{where}: forward requires explicit '{key}'")
        projection_geometry = forward_block["projection_geometry"]
        image_geometry = forward_block["image_geometry"]
        if not isinstance(projection_geometry, dict):
            raise ValueError(f"{where}: forward.projection_geometry must be a mapping")
        if not isinstance(image_geometry, dict):
            raise ValueError(f"{where}: forward.image_geometry must be a mapping")
        transitions = forward_block.get("physical_transforms", [])
        if not isinstance(transitions, list) or not all(isinstance(t, dict) for t in transitions):
            raise ValueError(f"{where}: forward.physical_transforms must be a list of mappings")
        return ForwardConfig(
            truth_fwhm_mm=truth,
            scanner=str(forward_block["scanner"]),
            attenuation_path=forward_block.get("attenuation_path", params.get("attenuation_path")),
            projection_geometry=dict(projection_geometry),
            image_geometry=dict(image_geometry),
            physical_transforms=tuple(dict(t) for t in transitions),
        )

    # Match the execution path: quick simulation reads input_params["fwhm_mm"]
    # first, so the identity must record that value rather than sim.fwhm_mm.
    truth = params.get("fwhm_mm", sim.get("fwhm_mm", LEGACY_TRUTH_FWHM_MM))
    return ForwardConfig(
        truth_fwhm_mm=_require_scalar_fwhm(truth, where),
        scanner=str(sim.get("scanner", LEGACY_SCANNER)),
        attenuation_path=params.get("attenuation_path"),
        projection_geometry=dict(sim.get("projection_geometry", {})),
        image_geometry=dict(sim.get("image_geometry", {})),
        physical_transforms=(),
    )


def _file_checksum(path: Path | str | None) -> str:
    if path is None:
        return ""
    p = Path(path)
    try:
        stat = p.stat()
    except OSError:
        return f"absent:{p}"
    # Key on size + mtime so a variant created/replaced within the same process
    # is rehashed rather than served from a stale path-only entry.
    key = (str(p), stat.st_size, stat.st_mtime_ns)
    cached = _CHECKSUM_CACHE.get(key)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _CHECKSUM_CACHE[key] = digest
    return digest


def _pet_source_path(run: Any) -> Path | None:
    root = Path(run.dataset.get("root", ""))
    if run.study == "spheres":
        name = "phant_pet.nii" if run.input_kind == "reference" else "phant_orig.nii"
        return root / name
    if run.study == "brainweb":
        subject = run.subject if run.subject is not None else run.dataset.get("subject_id")
        if subject is None:
            return None
        return root / f"subject_{int(subject):02d}" / "pet_gt.nii.gz"
    if run.study == "patient":
        subject = run.subject if run.subject is not None else run.dataset.get("subject_id")
        return root / str(subject) / "PET.nii.gz" if subject is not None else None
    return None


def _brainweb_subject_dir(run: Any) -> Path | None:
    if run.study != "brainweb":
        return None
    subject = run.subject if run.subject is not None else run.dataset.get("subject_id")
    if subject is None:
        return None
    return Path(run.dataset.get("root", "")) / f"subject_{int(subject):02d}"


def _guidance_lesion_identity(run: Any, lesion_state: str) -> dict[str, Any]:
    """Checksum/hash/rule for the selected T1 lesion-state variant.

    These describe guidance only and are therefore placed in the run identity,
    never in the forward/input identities or the acquisition seed.
    """
    subject_dir = _brainweb_subject_dir(run)
    mask_union_hash = ""
    multiplier: float | None = None
    rule: str | None = None
    t1_checksum = ""
    if subject_dir is not None:
        preparation_path = subject_dir / "preparation.json"
        if preparation_path.exists():
            preparation = json.loads(preparation_path.read_text())
            guidance = preparation.get("guidance", {})
            mask_union_hash = str(guidance.get("mask_union_hash", ""))
            raw_multiplier = guidance.get("injection_multiplier")
            multiplier = float(raw_multiplier) if raw_multiplier is not None else None
            rule = guidance.get("injection_rule")
        variant = "mr_t1_present.nii.gz" if lesion_state == "present" else "mr_t1_absent.nii.gz"
        t1_checksum = _file_checksum(subject_dir / variant)
    return {
        "lesion_state": lesion_state,
        "t1_checksum": t1_checksum,
        "mask_union_hash": mask_union_hash,
        "injection": {"multiplier": multiplier, "rule": rule},
    }


def identity_config(run: Any, where: str = "run") -> IdentityConfig:
    """Build the resolved identity config for a run-like object.

    Scenario expansion populates the new fields explicitly; direct ``RunSpec``
    construction falls back to the documented legacy derivation so the
    identity functions stay usable in tests and migration tooling.
    """
    params = dict(run.input_params)
    sim = dict(run.sim)

    forward = getattr(run, "forward", None)
    if forward is None:
        forward = resolve_forward(None, params, sim, where=where)

    osem = getattr(run, "osem", None)
    if osem is None:
        osem = resolve_osem(params, sim, where=where)

    modality = getattr(run, "guidance_modality", None)
    condition = getattr(run, "guidance_condition", None)
    guidance_params = {k: params[k] for k in ("guidance_modality", "guidance_condition") if k in params}
    if guidance_params:
        modality, condition = resolve_guidance(guidance_params, where=where)
    elif modality is None or condition is None:
        modality, condition = resolve_guidance(
            {"guidance_modality": modality, "guidance_condition": condition}, where=where
        )

    lesion_state = params.get("guidance_lesion_state", getattr(run, "guidance_lesion_state", None))
    lesion_state = resolve_guidance_lesion_state(
        {"guidance_lesion_state": lesion_state} if lesion_state is not None else {}, where=where
    )
    # The selected T1 variant only matters to guided methods using T1 guidance.
    is_guided_t1 = str(run.method_name) in GUIDED_METHODS and modality == "t1"
    guidance_lesion = _guidance_lesion_identity(run, lesion_state) if is_guided_t1 else {}

    attenuation_path = forward.attenuation_path or params.get("attenuation_path")
    effective_method_params, iterations = resolve_method_params(
        str(run.method_name), dict(run.method_params)
    )
    if "counts" in params:
        counts = _validate_counts(params["counts"], where)
    else:
        counts = None  # omitted is allowed for reference/native inputs
    return IdentityConfig(
        study=str(run.study),
        subject=getattr(run, "subject", None),
        input_kind=str(run.input_kind),
        forward=forward,
        counts=counts,
        realisation=int(params.get("realisation", 0)),
        base_seed=int(params.get("seed", sim.get("seed", 0))),
        osem=osem,
        condition=params.get("condition"),
        beta=_canonical_number(params.get("beta")),
        output_grid=dict(getattr(run, "output_grid", {}) or {}),
        method_name=str(run.method_name),
        method_params=effective_method_params,
        iterations=iterations,
        guidance_modality=modality,
        guidance_condition=condition,
        guidance_lesion_state=lesion_state,
        guidance_lesion=guidance_lesion,
        guidance_preprocessing=dict(params.get("guidance_preprocessing", {})),
        protocol_version=str(getattr(run, "protocol_version", LEGACY_PROTOCOL_VERSION)),
        stage=str(getattr(run, "stage", DEFAULT_STAGE)),
        selection_policy=str(getattr(run, "selection_policy", DEFAULT_SELECTION_POLICY)),
        pet_checksum=_file_checksum(_pet_source_path(run)),
        attenuation_checksum=_file_checksum(attenuation_path),
        object_perturbations={
            "add_tumours": bool(sim.get("add_tumours", False)),
            "tumour_contrast": sim.get("tumour_contrast"),
            "tumour_diameters_mm": sim.get("tumour_diameters_mm"),
        },
    )


def forward_identity(cfg: IdentityConfig) -> dict[str, Any]:
    return {
        "identity_schema": ID_SCHEMA_VERSION,
        "study": cfg.study,
        "subject": cfg.subject,
        "input_kind": cfg.input_kind,
        "pet_checksum": cfg.pet_checksum,
        "attenuation_checksum": cfg.attenuation_checksum,
        "truth_fwhm_mm": cfg.forward.truth_fwhm_mm,
        "scanner": cfg.forward.scanner,
        "projection_geometry": cfg.forward.projection_geometry,
        "image_geometry": cfg.forward.image_geometry,
        "physical_transforms": list(cfg.forward.physical_transforms),
        "object_perturbations": cfg.object_perturbations,
        "environment": {
            "cil-krl": _pkg_version("cil-krl"),
            "krl-studies": _pkg_version("krl-studies"),
        },
    }


def input_identity(cfg: IdentityConfig, forward_id_value: str | None = None) -> dict[str, Any]:
    return {
        "forward_id": forward_id_value if forward_id_value is not None else forward_id(cfg),
        "counts": cfg.counts,
        "noise_seed": cfg.noise_seed,
        "realisation": cfg.realisation,
        "reconstruction": {
            "condition": cfg.condition,
            "beta": cfg.beta,
            "subsets": cfg.osem.subsets,
            "full_iterations": cfg.osem.full_iterations,
            "subiterations": cfg.osem.subiterations,
            "initialisation": cfg.osem.initialisation,
        },
        "output_grid": cfg.output_grid,
    }


def run_identity(cfg: IdentityConfig, input_id_value: str | None = None) -> dict[str, Any]:
    guidance = {
        "modality": cfg.guidance_modality,
        "condition": cfg.guidance_condition,
        "preprocessing": cfg.guidance_preprocessing,
    }
    if cfg.guidance_lesion:
        guidance["lesion"] = cfg.guidance_lesion
    return {
        "input_id": input_id_value if input_id_value is not None else input_id(cfg),
        "method": cfg.method_name,
        "method_params": cfg.method_params,
        "stopping": {"iterations": cfg.iterations},
        "guidance": guidance,
        "implementation": {
            "protocol_version": cfg.protocol_version,
            "stage": cfg.stage,
            "selection_policy": cfg.selection_policy,
        },
    }


def forward_id(cfg: IdentityConfig) -> str:
    return short_id(f"fwd_{_slug(cfg.study)}_{_slug(cfg.subject)}", forward_identity(cfg))


def input_id(cfg: IdentityConfig) -> str:
    return short_id(f"inp_{_slug(cfg.study)}_{_slug(cfg.subject)}", input_identity(cfg))


def run_id(cfg: IdentityConfig) -> str:
    prefix = (
        f"run_{_slug(cfg.study)}_{_slug(cfg.subject)}_"
        f"{_slug(cfg.input_kind)}_{_slug(cfg.method_name)}"
    )
    return short_id(prefix, run_identity(cfg))


def all_identities(cfg: IdentityConfig) -> tuple[str, str, str, dict, dict, dict]:
    """Compute the three ids and their canonical dicts once, without recomputation."""
    fwd = forward_identity(cfg)
    fid = short_id(f"fwd_{_slug(cfg.study)}_{_slug(cfg.subject)}", fwd)
    inp = input_identity(cfg, forward_id_value=fid)
    iid = short_id(f"inp_{_slug(cfg.study)}_{_slug(cfg.subject)}", inp)
    run = run_identity(cfg, input_id_value=iid)
    prefix = (
        f"run_{_slug(cfg.study)}_{_slug(cfg.subject)}_"
        f"{_slug(cfg.input_kind)}_{_slug(cfg.method_name)}"
    )
    rid = short_id(prefix, run)
    return fid, iid, rid, fwd, inp, run
