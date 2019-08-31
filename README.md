
``` bash
# Activate venv
$ pipenv shell

# Install dependencies
$ pipenv install

# Create DB
$ python
>> from app import db
>> db.create_all()
>> exit()

# Run Server (http://localhst:5000)
python app.py
```

## Endpoints

* GET   http://localhost:5000/api/prediction/ecg2.txt
