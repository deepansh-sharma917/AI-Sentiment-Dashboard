from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

# ---------------- LOAD ENV VARIABLES ----------------
load_dotenv()  # Loads DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME

DB_USER = os.getenv("DB_USER")
DB_PASS = quote_plus(os.getenv("DB_PASS"))  # encode special characters
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# ---------------- DATABASE URL ----------------
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ---------------- SQLALCHEMY ENGINE & SESSION ----------------
engine = create_engine(DATABASE_URL, echo=False)  # set echo=True for SQL logging
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ---------------- BASE CLASS ----------------
Base = declarative_base()

# ---------------- DEPENDENCY ----------------
def get_db():
    """
    FastAPI / Streamlit style database session.
    Usage in FastAPI: db: Session = Depends(get_db)
    Usage in Streamlit: db = next(get_db())
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
