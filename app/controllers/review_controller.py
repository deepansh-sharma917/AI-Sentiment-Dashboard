from models import Review
from database import get_db
from fastapi import APIRouter, Depends, HTTPException
import pandas as pd

router = APIRouter()

# ----------------- CORE LOGIC -----------------
def insert_reviews_from_df(df, db):
    """
    Insert new reviews from a DataFrame into the database.
    Avoids duplicates using review_hash.
    """
    # Normalize column names
    df = df.rename(columns=lambda x: x.strip())
    if 'reviewText' in df.columns:
        df['review_text'] = df['reviewText']

    # Fetch existing hashes to avoid duplicates
    existing_hashes = set(r[0] for r in db.query(Review.review_hash).all())

    new_reviews = []
    for _, row in df.iterrows():
        if row['review_hash'] not in existing_hashes:
            new_reviews.append(
                Review(
                    review_text=row['review_text'],
                    review_hash=row['review_hash'],
                    sentiment=row['sentiment'],
                    sentiment_score=row['sentiment_score']
                )
            )
            existing_hashes.add(row['review_hash'])  # Prevent duplicates in same batch

    if new_reviews:
        db.bulk_save_objects(new_reviews)
        db.commit()

    return len(new_reviews)


def fetch_reviews(db):
    """
    Fetch all reviews, ordered by creation date descending.
    """
    return db.query(Review).order_by(Review.created_at.desc()).all()


# ----------------- FASTAPI ROUTES -----------------
@router.post("/insert_reviews")
def insert_reviews(df: list, db = Depends(get_db)):
    """
    FastAPI endpoint to insert multiple reviews.
    Expects a list of dicts with keys: review_text/reviewText, review_hash, sentiment, sentiment_score
    """
    if not df:
        raise HTTPException(status_code=400, detail="No reviews provided")

    df = pd.DataFrame(df)
    inserted_count = insert_reviews_from_df(df, db)
    return {"message": f"{inserted_count} new reviews inserted successfully."}


@router.get("/reviews")
def get_reviews(db = Depends(get_db)):
    """
    FastAPI endpoint to fetch all reviews.
    """
    reviews = fetch_reviews(db)
    return [
        {
            "review_text": r.review_text,
            "sentiment": r.sentiment,
            "sentiment_score": r.sentiment_score,
            "created_at": r.created_at
        } for r in reviews
    ]
