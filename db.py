import os
from sqlalchemy import create_engine
import dotenv

dotenv.load_dotenv()

URL = (
    f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT', 3306)}/"
    f"{os.getenv('DB_NAME')}"
)

engine = create_engine(URL, pool_pre_ping=True)
