import pickle
from pathlib import Path
import streamlit_authenticator as stauth

names = ["admin", "user"]
usernames = ["admin", "user"]
passwords  = ["1234", "1234"] #hashed

hashed_passwords = stauth.Hasher(passwords).generate()

file_path = Path(__file__).parent/"pkl/hashed_pw.pkl"
with file_path.open("wb") as file:
    pickle.dump(hashed_passwords, file)