{
  description = "Default template flake.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    # old_nixpkgs.url = "github:nixos/nixpkgs/194c2aa446b2b059886bb68be15ef6736d5a8c31";
  };

  outputs = {
    nixpkgs,
    # old_nixpkgs,
    ...
  }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
    # oldpkgs = old_nixpkgs.legacyPackages.${system};
  in {
    devShells.${system}.default = pkgs.mkShell {
      nativeBuildInputs = with pkgs; [
        uv
        ffmpeg-full
        libxext
        mesa
        opencv
        pkg-config
        stdenv.cc.cc.lib
      ];

      LD_LIBRARY_PATH = "/run/opengl-driver/lib:/run/opengl-driver-32/lib";
    };

    doCheck = false;
  };
}
