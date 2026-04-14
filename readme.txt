Run this from the repo root:

deactivate 2>/dev/null

rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

Set the API keys you need in `.env` before running the notebook:

PINECONE_API_KEY="your-pinecone-key"
ANTHROPIC_API_KEY="your-claude-key"

To enter that environment later:

source .venv/bin/activate
To verify you’re in the right one:

python --version
