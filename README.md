# makerspace-ai-project
csh makerspace project code notes

Loren:
- created newFlorenceUseThisVersionWORKING.py
    - runs the Florence-2 model in the terminal
- created setupVenv.sh
    - Terminal shell that sets up a virtual environment (venv) named ".venv" for you in case you delete or don't know how to set it up
- instructions to run newFlorenceUseThisVersionWORKING.py
    - only do this once (if the setupVenv.sh file doesn't work for you)
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
- instructions to run setupVenv.sh
    - grant permission (only need to do this once): chmod +x setup_env.sh
    - set up .venv: ./setupVenv.sh
- notes
    - for checking if the venv is healthy: python -c "import torch; import transformers; print(f'GPU Ready: {torch.backends.mps.is_available()}'); print(f'Transformers: {transformers.__version__}')"
    - to change the filepath, go to line 45 and update the quote with the filepath to direct Florence to the image you want

Jaya:
- created index.html
    - what does it do?
- created fix_it.py
    - what does it do?
- created inventory.csv
    - what does it do?
- instructions to run (INSERT FILE)

Isabella:
