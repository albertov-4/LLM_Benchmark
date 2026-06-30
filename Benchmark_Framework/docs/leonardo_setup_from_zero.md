# Leonardo Setup From Zero

This is the working from-zero setup for `Benchmark_Framework` on Leonardo. It
uses placeholders for all personal data.

## 1. Windows PowerShell Access To Leonardo

Install Smallstep:

```powershell
winget install Smallstep.step
```

Bootstrap the CINECA CA:

```powershell
step ca bootstrap --force --ca-url=https://sshproxy.hpc.cineca.it --fingerprint 2ae1543202304d3f434bdc1a2c92eff2cd2b02110206ef06317e70c1c1735ecd
```

If `ssh-agent` is not running on Windows, enable and start it before login.

Log in with Smallstep:

```powershell
step ssh login "<CINECA_EMAIL>" --provisioner cineca-hpc
step ssh list
```

Connect to Leonardo:

```powershell
ssh <CINECA_USERNAME>@login.leonardo.cineca.it
```

If SSH reports a stale host key:

```powershell
ssh-keygen -R login.leonardo.cineca.it
```

## 2. First Checks Inside Leonardo

```bash
whoami
hostname
pwd
echo $HOME
echo $CINECA_SCRATCH
echo $PUBLIC
saldo -b
```

Use `$CINECA_SCRATCH` for this repository, Python environments, logs, and model
cache. Do not use `$HOME` for large data or models. `$PUBLIC` is visible to
other users by default, so do not store private tokens or private data there.

## 3. Clone Only `Benchmark_Framework`

```bash
cd $CINECA_SCRATCH

git clone --depth 1 --filter=blob:none --sparse <GITHUB_REPO_URL>
cd LLM_Benchmark
git sparse-checkout set Benchmark_Framework
```

Expected path:

```bash
$CINECA_SCRATCH/LLM_Benchmark/Benchmark_Framework
```

## 4. Configure GitHub SSH On Leonardo

```bash
ssh-keygen -t ed25519 -C "<LEONARDO_GITHUB_KEY_LABEL>"
cat ~/.ssh/id_ed25519.pub
```

Copy only the public key, `id_ed25519.pub`, to:

```text
GitHub -> Settings -> SSH and GPG keys -> New SSH key
```

Test GitHub SSH:

```bash
ssh -T git@github.com
```

Switch the repository remote from HTTPS to SSH:

```bash
cd $CINECA_SCRATCH/LLM_Benchmark
git remote set-url origin git@github.com:<GITHUB_OWNER>/<GITHUB_REPO>.git
git pull
```

## 5. Create `our_env`

`our_env` is the main environment for the standard benchmark workflow and
non-GPT-OSS local/API workflows. Nemotron local inference is not part of the
default active workflow; `mamba-ssm` is optional/legacy and is not installed by
default.

```bash
cd $CINECA_SCRATCH/LLM_Benchmark

module purge
module load python/3.11.7
module load gcc/12.2.0
module load cuda/12.1

python -m venv $CINECA_SCRATCH/our_env
source $CINECA_SCRATCH/our_env/bin/activate

python -m pip install --upgrade pip setuptools wheel packaging ninja
```

Install the CUDA 12.1 PyTorch stack first:

```bash
pip install \
  torch==2.5.1+cu121 \
  torchvision==0.20.1+cu121 \
  torchaudio==2.5.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

Check:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Expected:

```text
2.5.1+cu121 12.1
```

Install the rest without `causal-conv1d` in the same pass:

```bash
grep -v "causal-conv1d" Benchmark_Framework/requirements/leonardo-our-env.txt > /tmp/leonardo-our-env-no-causal.txt
pip install -r /tmp/leonardo-our-env-no-causal.txt
```

Install `causal-conv1d` separately:

```bash
pip install --no-build-isolation causal-conv1d==1.6.2.post1
```

Apply the Leonardo `libstdc++` fix required by binary imports such as
`causal-conv1d`:

```bash
export LD_LIBRARY_PATH=$(dirname $(g++ -print-file-name=libstdc++.so.6)):$LD_LIBRARY_PATH
export LD_PRELOAD=$(g++ -print-file-name=libstdc++.so.6)
```

Create the activation helper:

```bash
cat > $CINECA_SCRATCH/activate_our_env.sh <<'EOF'
module purge
module load python/3.11.7
module load gcc/12.2.0
module load cuda/12.1

source $CINECA_SCRATCH/our_env/bin/activate

export LD_LIBRARY_PATH=$(dirname $(g++ -print-file-name=libstdc++.so.6)):$LD_LIBRARY_PATH
export LD_PRELOAD=$(g++ -print-file-name=libstdc++.so.6)
EOF
```

Use:

```bash
source $CINECA_SCRATCH/activate_our_env.sh
```

Do not use only `source $CINECA_SCRATCH/our_env/bin/activate`; the helper also
loads Leonardo modules and fixes `libstdc++`.

Final checks:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import causal_conv1d; print('causal ok')"
python -c "import transformers; print(transformers.__version__)"
python -c "import openai; print('openai ok')"
pip check
```

Expected: Torch reports `2.5.1+cu121 12.1`, `causal_conv1d` imports, and
`pip check` reports no broken requirements.

## 6. Create `gptoss_env`

`gptoss_env` is dedicated to `hf_gpt_oss_120b`.

```bash
cd $CINECA_SCRATCH/LLM_Benchmark

module purge
module load python/3.11.7
module load gcc/12.2.0

python -m venv $CINECA_SCRATCH/gptoss_env
source $CINECA_SCRATCH/gptoss_env/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -r Benchmark_Framework/requirements/leonardo-gptoss-env.txt
```

Create the activation helper:

```bash
cat > $CINECA_SCRATCH/activate_gptoss_env.sh <<'EOF'
module purge
module load python/3.11.7
module load gcc/12.2.0

source $CINECA_SCRATCH/gptoss_env/bin/activate
EOF
```

Use:

```bash
source $CINECA_SCRATCH/activate_gptoss_env.sh
```

Checks:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import transformers; print(transformers.__version__)"
python -c "import triton; print(triton.__version__)"
pip check
```

## 7. Hugging Face Token

Do not write the Hugging Face token into repository files.

```bash
read -s HF_TOKEN
export HF_TOKEN
```

Verify:

```bash
hf auth whoami --token "$HF_TOKEN"
```

Pass the token explicitly in model download commands:

```bash
--token "$HF_TOKEN"
```

`HF_TOKEN` is a shell environment variable, not permanently tied to `our_env`
or `gptoss_env`. Set it again in each new session unless you use another secure
credential-management method.

## 8. Download Models Into The Framework Cache

Do not create random model folders outside the framework. The working cache is:

```bash
$CINECA_SCRATCH/LLM_Benchmark/Benchmark_Framework/models_cache
```

Downloads can run directly from the login node without `sbatch`; do not request
GPUs for simple Hugging Face downloads.

GPT-OSS:

```bash
cd $CINECA_SCRATCH/LLM_Benchmark/Benchmark_Framework

source $CINECA_SCRATCH/activate_gptoss_env.sh

unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

export HF_HOME=$PWD/models_cache/.hf_home
mkdir -p models_cache "$HF_HOME"

hf download openai/gpt-oss-120b \
  --token "$HF_TOKEN" \
  --local-dir models_cache/openai__gpt-oss-120b \
  2>&1 | tee -a prepare_gptoss_download.log
```

Other active Hugging Face local models:

```bash
cd $CINECA_SCRATCH/LLM_Benchmark/Benchmark_Framework

source $CINECA_SCRATCH/activate_our_env.sh

unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

export HF_HOME=$PWD/models_cache/.hf_home
mkdir -p models_cache "$HF_HOME"
```

Gemma:

```bash
hf download google/gemma-4-31B-it \
  --token "$HF_TOKEN" \
  --local-dir models_cache/google__gemma-4-31B-it \
  2>&1 | tee -a prepare_gemma_download.log
```

Phi-4:

```bash
hf download microsoft/phi-4 \
  --token "$HF_TOKEN" \
  --local-dir models_cache/microsoft__phi-4 \
  2>&1 | tee -a prepare_phi4_download.log
```

Qwen:

```bash
hf download Qwen/Qwen3.6-27B \
  --token "$HF_TOKEN" \
  --local-dir models_cache/Qwen__Qwen3.6-27B \
  2>&1 | tee -a prepare_qwen_download.log
```

Nemotron is disabled in the current Hugging Face registry and is not part of
the default active download list. If a download stops, rerun the same
`hf download` command. Do not delete `models_cache`.

## 9. Test Model Cache Offline

```bash
cd $CINECA_SCRATCH/LLM_Benchmark/Benchmark_Framework

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

bash Leonardo_script/test_models_cache.sh
```

If the test reports partial files:

```bash
find models_cache -type f \( -name "*.incomplete" -o -name "*.lock" -o -name "*.tmp" \)
```

After the matching `hf download` completes successfully, remove stale partial
files:

```bash
find models_cache -type f \( -name "*.incomplete" -o -name "*.lock" -o -name "*.tmp" \) -delete
```

Then rerun:

```bash
bash Leonardo_script/test_models_cache.sh
```

## 10. SLURM And Partition Notes

Model downloads and cache checks do not require GPU/Booster. They can write
directly to:

```bash
$CINECA_SCRATCH/LLM_Benchmark/Benchmark_Framework/models_cache
```

Use GPU/Booster for actual local model inference and benchmark jobs, not for
simple Hub downloads or offline cache checks. Do not request `--gres=gpu` for
downloads or cache checks.

For benchmark jobs, set the account explicitly:

```bash
export SLURM_ACCOUNT="<CINECA_PROJECT_ACCOUNT>"
```

## 11. Privacy Check

Before committing documentation changes, check for old personal values:

```bash
grep -R "<OLD_CINECA_USERNAME>\|<OLD_CINECA_PROJECT_ACCOUNT>\|<OLD_EMAIL>" Benchmark_Framework
```

Documentation examples must use placeholders such as `<CINECA_USERNAME>`,
`<CINECA_PROJECT_ACCOUNT>`, `<GITHUB_OWNER>`, and `<HF_TOKEN>`.
