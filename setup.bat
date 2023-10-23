cd "%~dp0"

call .\venv\Scripts\activate

python -m pip install -r .\req.txt

cd .\ml

python dash_app.py
