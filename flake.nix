{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      allSystems = [
        "x86_64-linux" # 64-bit Intel/AMD Linux
        "aarch64-linux" # 64-bit ARM Linux
        "x86_64-darwin" # 64-bit Intel macOS
        "aarch64-darwin" # 64-bit ARM macOS
      ];

      forAllSystems = f: nixpkgs.lib.genAttrs allSystems (system: f {
        pkgs = import nixpkgs {
          inherit system;
        };
      });

      # Deepface requires mtcnn, which requires keras 2.14.0, but nixpkgs has 3.x releases in it.
      # So, manually package the required version.
      python-keras = pkgs: pkgs.python311Packages.buildPythonPackage rec {
        pname = "keras";
        version = "2.14.0";
        format = "wheel";

        src = pkgs.fetchPypi {
          inherit format pname version;
          hash = "sha256-10KdHSExzH6x8uouwzAifH2dONqz398ueN7+5OzEP80=";
          python = "py3";
          dist = "py3";
        };

        nativeCheckInputs = [
          pkgs.pytest
          pkgs.pytest-cov
          pkgs.pytest-xdist
        ];

        propagatedBuildInputs = [
          pkgs.python311Packages.six
          pkgs.python311Packages.pyyaml
          pkgs.python311Packages.numpy
          pkgs.python311Packages.scipy
          pkgs.python311Packages.h5py
          pkgs.python311Packages.keras-applications
          pkgs.python311Packages.keras-preprocessing
        ];

        # Skip tests.
        doCheck = false;
      };
      # Deepface requires mtcnn.
      # https://github.com/ipazc/mtcnn
      mtcnn-src = pkgs: pkgs.fetchFromGitHub {
        owner = "ipazc";
        repo = "mtcnn";
        rev = "master";
        sha256 = "sha256-GXUrLJ5XD6V2hT/gjyYSuh/CMMw2xIXKBsYFvQmbLYs=";
      };
      mtcnn = pkgs: pkgs.python311Packages.buildPythonPackage {
        pname = "mtcnn";
        version = "main";

        src = "${(mtcnn-src pkgs)}";

        # Do not run tests.
        doCheck = false;

        # Add dependencies.
        propagatedBuildInputs = [
          pkgs.python311Packages.tensorflow
          (python-keras pkgs)
          pkgs.python311Packages.opencv4
        ];

        # Patch https://github.com/ipazc/mtcnn/pull/129 which fixes a bug due to a breaking
        # change in setuptools.
        patches = [ ./mtcnn-0001-merge-pr-129.patch ];

        build-system = [
          pkgs.python311Packages.setuptools
          pkgs.python311Packages.wheel
        ];
      };
      # Deepface requires https://github.com/heewinkim/retinaface
      retinaface = pkgs: pkgs.python311Packages.buildPythonPackage {
        pname = "retinaface";
        version = "0.0.1";

        src = pkgs.fetchFromGitHub {
          owner = "heewinkim";
          repo = "retinaface";
          rev = "master";
          sha256 = "sha256-ELnTLgWiz8/b55JIPNZ0xkpCIyLyADomT07lp9LhAYU=";
        };

        # Do not run tests.
        doCheck = false;

        build-system = [
          pkgs.python311Packages.setuptools
          pkgs.python311Packages.wheel
        ];
      };
      deepface-deps = pkgs: [
        pkgs.python311Packages.tensorflow
        pkgs.python311Packages.pandas
        pkgs.python311Packages.gdown
        (mtcnn pkgs)
        (retinaface pkgs)
        pkgs.python311Packages.opencv4
        pkgs.python311Packages.gunicorn # Required to run app.
        pkgs.python311Packages.flask # Required to run app.
        pkgs.python311Packages.pytest # Required to run app.
      ];
      python-with-deepface-deps = pkgs: pkgs.python311.withPackages (python-pkgs:
        (deepface-deps pkgs)
      );
      deepface = pkgs: pkgs.python311Packages.buildPythonPackage {
        pname = "deepface";
        version = "v0.0.90";

        src = ./.;

        # Do not run tests.
        doCheck = false;

        # Add dependencies.
        propagatedBuildInputs = deepface-deps pkgs;

        build-system = [
          pkgs.python311Packages.setuptools
          pkgs.python311Packages.wheel
        ];
      };
      python-with-deepface = pkgs: pkgs.python311.withPackages (python-pkgs: [
        (deepface pkgs)
      ]);
      deepface-weights = pkgs: pkgs.fetchurl {
        url = "https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5";
        hash = "sha256-dZJmuWFND9XWW5e/cWgYt0bMd6tZRMe//JN8a6lFXYw=";
      };
      deepface-home = pkgs: pkgs.stdenv.mkDerivation {
        name = "deepface-home";

        dontUnpack = true;

        src = ./.;

        installPhase = ''
          mkdir -p $out/opt/deepface/home/.deepface/weights
          cp ${deepface-weights pkgs} $out/opt/deepface/home/.deepface/weights/vgg_face_weights.h5
        '';
      };
      deepface-api = pkgs: pkgs.stdenv.mkDerivation {
        name = "deepface-api";

        # Runtime dependencies.
        propagatedBuildInputs = [
          (python-with-deepface pkgs)
        ];

        # Unpacking runs the tests, which modify data on disk, so skip that,
        # because Nix requires read only inputs.
        dontUnpack = true;

        src = ./.;

        installPhase = ''
          mkdir -p $out/opt/deepface-api
          cp -r $src/deepface/api/src/* $out/opt/deepface-api
        '';
      };
    in
    {
      packages = forAllSystems ({ pkgs }: {
        default = pkgs.writeShellScriptBin "deepface-api" ''
          # Set the path for the opencv data files.
          # Data files are found at, for example, /nix/store/c1h6iyvpi2j7a9bypjnvz9zlw4lq1psk-opencv-4.9.0/share/opencv4/haarcascades/haarcascade_eye.xml
          export OPENCV_PATH="${(pkgs.python311Packages.opencv4)}/share/opencv4/haarcascades/"
          # Set the home directory where the weights are stored.
          export DEEPFACE_HOME="${(deepface-home pkgs)}/opt/deepface/home"
          ${(python-with-deepface pkgs)}/bin/python ${deepface-api pkgs}/opt/deepface-api/api.py
        '';
        python311Packages.deepface = (deepface pkgs);
      });
      devShells = forAllSystems ({ pkgs }: {
        default = pkgs.mkShell {
          packages = [
            (python-with-deepface-deps pkgs)
          ];
        };
      });
    };
}
