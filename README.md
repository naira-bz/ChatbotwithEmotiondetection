# Django-registration-and-login-system
This web app has been developed using the popular Django and rasa framework for the backend and Bootstrap for the frontend.
### Basic Features of The App
    
* Register – Users can register and create a new profile
* Login - Registered users can login using username and password
* User Profile - Once logged in, users can create and update additional information such as avatar and bio in the profile page
* Update Profile – Users can update their information such as username, email, password, avatar and bio
* Forgot Password – Users can easily retrieve their password if they forget it 
* Admin Panel – admin can CRUD users

### Quick Start
To get this project up and running locally on your computer follow the following steps.
1. Set up a python virtual environment
2. Run the following commands for Rasa
    * rasa run actions
    * rasa run -m models --enable-api --cors "*" 
3. Run the following commands for Django
    * pip install -r requirements.txt
    * python manage.py makemigrations
    * python manage.py migrate
    * python manage.py createsuperuser
    * python manage.py runserver
4. Open a browser and go to http://localhost:8000/