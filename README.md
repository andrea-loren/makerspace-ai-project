# makerspace-ai-project
csh makerspace project code notes

Loren:
- created newFlorenceUseThisVersionWORKING.py
    - runs the Florence-2 model in the terminal
    - change the filepath to direct the model to your image
    - make sure you have a virtual environment (venv) running with the required software to run it
- instructions
    - only do this once
        - create a venv named ".venv": python3 -m venv .venv
        - enter your venv: source .venv/bin/activate
        - install the software using these commands:
            - upgrade pip: python -m pip install --upgrade pip
            - install software (general): pip install torch torchvision torchaudio "transformers<4.46.0" accelerate
            - install software (vision & math libraries): pip install einops timm
            - install software (text tokens & data files): pip install protobuf sentencepiece
            - install software (image handling): pip install requests Pillow
            - install software (drawing bounding boxes & labels): pip install matplotlib
        - then go onto the part you can repeat below
    - this is the part you can repeat:
        - enter your venv if you're not already in it: source .venv/bin/activate
        - run file: python3 newFlorenceUseThisVersionWORKING.py
        - when it's done, re-run the program: python3 newFlorenceUseThisVersionWORKING.py
        - or exit the venv: deactivate
- for checking if the venv is healthy: python -c "import torch; import transformers; print(f'GPU Ready: {torch.backends.mps.is_available()}'); print(f'Transformers: {transformers.__version__}')"
