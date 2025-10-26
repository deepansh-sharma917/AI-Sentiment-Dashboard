from sqlalchemy import Column, Integer, String, Float, Text, DateTime, func
from database import Base

class Review(Base):
    __tablename__ = "reviews"
    id = Column(Integer, primary_key=True, index=True)
    review_text = Column(Text, nullable=False)
    review_hash = Column(String(32), unique=True)
    sentiment = Column(String(20))
    sentiment_score = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
