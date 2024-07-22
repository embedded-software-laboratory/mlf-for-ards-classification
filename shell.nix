{ pkgs ? import <nixpkgs> {}}:
let
  fhs = pkgs.buildFHSUserEnv {
    name = "mlp-framework";

    targetPkgs = _: with pkgs; [
      micromamba python311 libGL
    ];

    profile = ''
      set -e
      eval "$(micromamba shell hook --shell=posix)"
      export MAMBA_ROOT_PREFIX=${builtins.getEnv "PWD"}/.mamba
			export TMPDIR=${builtins.getEnv "PWD"}/.tmp
			mkdir -p $TMPDIR
      micromamba create -q -n mlp_framework
      micromamba activate mlp_framework
      micromamba install --yes -f src/environment_hpc.yml -c conda-forge
      set +e
    '';
  };
in fhs.env
