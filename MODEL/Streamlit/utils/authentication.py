import pickle
from pathlib import Path
import streamlit_authenticator as stauth

names = ["admin", "doctor", "user"]
usernames = ["admin", "doctor", "user"]
passwords  = ["1234", "1234", '1234'] #hashed

hashed_passwords = stauth.Hasher(passwords).generate()

if __name__ == "__main__":
    file_path = Path(__file__).parent.parent/"pkl"/"hashed_pw.pkl"
    with file_path.open("wb") as file:
        pickle.dump(hashed_passwords, file)