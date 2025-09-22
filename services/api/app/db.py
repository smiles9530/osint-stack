import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from datetime import datetime
from .config import settings
from .uuid_validation_fix import safe_uuid_conversion, process_article_data
@contextmanager
def get_conn():
    conn = psycopg2.connect(settings.database_url)
    try:
        yield conn
    finally:
        conn.close()
def upsert_article(url, title, text, lang, published_at, source_name=None, article_id=None):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Check if we have the safe_upsert_article function available
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.routines 
                    WHERE routine_name = 'safe_upsert_article' 
                    AND routine_schema = 'public'
                )
            """)
            has_safe_function = cur.fetchone()[0]
            
            if has_safe_function:
                # Use the safe function that handles Reddit IDs and UUID validation
                cur.execute(
                    "SELECT safe_upsert_article(%s, %s, %s, %s, %s, %s, %s, %s)",
                    (url, title, text, lang, published_at, source_name, '{}', article_id)
                )
                article_id = cur.fetchone()[0]
                conn.commit()
                return article_id
            else:
                # Fallback to manual handling
                source_id = None
                if source_name:
                    cur.execute("INSERT INTO sources(name) VALUES(%s) ON CONFLICT (name) DO UPDATE SET name=EXCLUDED.name RETURNING id;",
                                (source_name,))
                    source_id = cur.fetchone()["id"]
                
                # Use current timestamp if published_at is None
                if published_at is None:
                    published_at = datetime.now()
                
                # Convert article_id to UUID if provided
                if article_id:
                    article_id = safe_uuid_conversion(article_id)
                
                # Use the corrected schema with id as primary key
                cur.execute(
                    """
                    INSERT INTO articles(url, source_id, title, content, language, published_at, source_name, id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (url) DO UPDATE SET
                      title=COALESCE(EXCLUDED.title, articles.title),
                      content=COALESCE(EXCLUDED.content, articles.content),
                      language=COALESCE(EXCLUDED.language, articles.language),
                      published_at=COALESCE(EXCLUDED.published_at, articles.published_at),
                      source_name=COALESCE(EXCLUDED.source_name, articles.source_name),
                      fetched_at=NOW()
                    RETURNING id;
                    """,
                    (url, source_id, title, text, lang, published_at, source_name, article_id)
                )
                article_id = cur.fetchone()["id"]
                conn.commit()
                return article_id
def insert_embedding(article_id, vec, model: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Use the corrected schema with id as primary key
            # Check if analysis_results table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = 'analysis_results'
                )
            """)
            has_analysis_table = cur.fetchone()[0]
            
            if has_analysis_table:
                # Use analysis_results table for embeddings
                cur.execute(
                    """
                    INSERT INTO analysis_results(article_id, analysis_type, results, model_name)
                    VALUES (%s, 'embedding', %s, %s)
                    ON CONFLICT (article_id, analysis_type) DO UPDATE SET
                      results=EXCLUDED.results,
                      model_name=EXCLUDED.model_name;
                    """,
                    (article_id, vec, model)
                )
            else:
                # Fallback to embeddings table if analysis_results doesn't exist
                cur.execute(
                    """
                    INSERT INTO embeddings(article_id, vec, model)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (article_id) DO UPDATE SET
                      vec=EXCLUDED.vec,
                      model=EXCLUDED.model;
                    """,
                    (article_id, vec, model)
                )
            conn.commit()
