FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu-24.04

# gcc + cmake + ninja + gdb already come from the base cpp devcontainer image.
# Add clang (CI builds with both gcc and clang) and Python (for demo/ and
# examples/, which each manage their own venv against requirements.txt).
RUN apt-get update && apt-get install -y --no-install-recommends \
    clang \
    lld \
    python3 \
    python3-venv \
    python3-pip \
    pipx \
  && rm -rf /var/lib/apt/lists/*

USER vscode

# Pin clang-format to the exact version CI checks formatting with (see
# .clang-format and .github/workflows/ci.yml) so local `clang-format` output
# matches the CI job bit-for-bit.
RUN pipx install clang-format==18.1.8 && pipx ensurepath
