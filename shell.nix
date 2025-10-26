{
  pkgs ? import <nixpkgs> {},
  run ? "bash",
}: let
  fhs = pkgs.buildFHSEnv {
    name = "fhs-shell";
    targetPkgs = pkgs: with pkgs; [uv zlib];
    runScript = "${run}";
  };
in
  fhs.env
