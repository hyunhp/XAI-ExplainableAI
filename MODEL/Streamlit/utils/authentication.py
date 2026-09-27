"""Create hashed demo credentials for the Streamlit app.

Demo only: users are admin / doctor / user and share one password,
read from the DEMO_PASSWORD environment variable (default "1234").
Run once from MODEL/Streamlit:  python utils/authentication.py
"""
import os
import pickle
from pathlib import Path

import streamlit_authenticator as stauth

names = ["admin", "doctor", "user"]
usernames = ["admin", "doctor", "user"]
demo_password = os.environ.get("DEMO_PASSWORD", "1234")
passwords = [demo_password] * len(usernames)

hashed_passwords = stauth.Hasher(passwords).generate()

if __name__ == "__main__":
    file_path = Path(__file__).parent.parent / "pkl" / "hashed_pw.pkl"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as file:
        pickle.dump(hashed_passwords, file)
    print(f"Saved demo credentials to {file_path}")
