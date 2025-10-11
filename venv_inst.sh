set -e

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing package..."
pip install -e .

echo "Running inference..."
python ./rlt-inference.py

echo "Done!"