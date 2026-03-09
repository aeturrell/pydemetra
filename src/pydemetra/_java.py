from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import jpype
import jpype.imports
from loguru import logger

_JARS_DIR = Path(__file__).parent / "_jars"

_NO_JAVA_MSG = (
    "Could not find a Java installation. "
    "Please install Java 17+ and ensure JAVA_HOME is set or java is on your PATH."
)


_JVM_LIB_CANDIDATES = [
    "lib/libjli.dylib",
    "lib/server/libjvm.dylib",
    "lib/server/libjvm.so",
]


def _find_jvm_in(java_home_dir: Path) -> str | None:
    """Return the JVM shared library path under a given JAVA_HOME, or None."""
    for rel in _JVM_LIB_CANDIDATES:
        candidate = java_home_dir / rel
        if candidate.exists():
            return str(candidate)
    return None


def _scan_jdk_dirs() -> str | None:
    """Search well-known JDK locations on macOS and Linux."""
    for brew_prefix in [Path("/opt/homebrew"), Path("/usr/local")]:
        openjdk_dir = brew_prefix / "opt" / "openjdk"
        if openjdk_dir.is_dir():
            libexec = openjdk_dir / "libexec"
            if libexec.is_dir():
                for jdk_bundle in sorted(libexec.glob("*.jdk"), reverse=True):
                    found = _find_jvm_in(jdk_bundle / "Contents" / "Home")
                    if found:
                        return found

    for jvm_dir in [Path("/Library/Java/JavaVirtualMachines")]:
        if jvm_dir.is_dir():
            for jdk_bundle in sorted(jvm_dir.iterdir(), reverse=True):
                found = _find_jvm_in(jdk_bundle / "Contents" / "Home")
                if found:
                    return found

    return None


def _find_jvm_path(java_home: str | None = None) -> str:
    """Locate the JVM shared library path."""
    if java_home is not None:
        found = _find_jvm_in(Path(java_home))
        if found:
            return found

    java_home_env = os.environ.get("JAVA_HOME")
    if java_home_env:
        found = _find_jvm_in(Path(java_home_env))
        if found:
            return found

    found = _scan_jdk_dirs()
    if found:
        return found

    try:
        return jpype.getDefaultJVMPath()
    except Exception:
        pass

    java_bin = shutil.which("java")
    if java_bin:
        java_real = Path(java_bin).resolve()
        java_home_dir = java_real.parent.parent
        found = _find_jvm_in(java_home_dir)
        if found:
            return found

        try:
            result = subprocess.run(
                [str(java_real), "-XshowSettings:properties", "-version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in (result.stdout + result.stderr).splitlines():
                if "java.home" in line:
                    jh = line.split("=", 1)[1].strip()
                    found = _find_jvm_in(Path(jh))
                    if found:
                        return found
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
            pass

    raise RuntimeError(_NO_JAVA_MSG)


def start_jvm(java_home: str | None = None, max_heap: str = "512m") -> None:
    """Start the JVM with all bundled JARs on the classpath.

    This is idempotent: calling it when the JVM is already running is a no-op.

    Args:
        java_home (str | None): Path to a JDK installation. If ``None``, JPype
            discovers it automatically (respecting ``JAVA_HOME``).
        max_heap (str): Maximum heap size, e.g. ``"512m"`` or ``"2g"``.

    Raises:
        FileNotFoundError: If no JAR files are found in the bundled JARs directory.
    """
    if jpype.isJVMStarted():
        return

    jars = sorted(_JARS_DIR.glob("*.jar"))
    if not jars:
        raise FileNotFoundError(f"No JAR files found in {_JARS_DIR}")

    classpath = os.pathsep.join(str(j) for j in jars)
    jvm_path = _find_jvm_path(java_home)

    logger.debug("Starting JVM with {} JARs, max_heap={}", len(jars), max_heap)
    jpype.startJVM(
        jvm_path,
        f"-Xmx{max_heap}",
        "--enable-native-access=ALL-UNNAMED",
        classpath=[classpath],
        convertStrings=True,
    )


def _ensure_jvm() -> None:
    """Ensure the JVM is running, starting it with defaults if needed."""
    if not jpype.isJVMStarted():
        start_jvm()


def get_java_version() -> str:
    """Return the Java runtime version string."""
    _ensure_jvm()
    return str(jpype.java.lang.System.getProperty("java.version"))
