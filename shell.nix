{ pkgs ? import <nixpkgs> {}}:
let
  fhs = pkgs.buildFHSUserEnv {
    name = "my-fhs-environment";

    targetPkgs = _: with pkgs; [
      micromamba libGL
    ];

    profile = ''
      set -e
      eval "$(micromamba shell hook --shell=posix)"
      export MAMBA_ROOT_PREFIX=${builtins.getEnv "PWD"}/.mamba
      if ! test -d $MAMBA_ROOT_PREFIX/envs/ai; then
          micromamba create --yes -q -n ai
      fi
      micromamba activate ai
      micromamba install --yes -f environment_hpc.yml -c conda-forge
      set +e
    '';
  };
in fhs.env
