from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import jwt
import mysql.connector
from mysql.connector import Error
import json
from datetime import datetime, timedelta, date
from decimal import Decimal
from dotenv import load_dotenv
import re
from langchain.chains import create_sql_query_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError
from langchain_community.utilities import SQLDatabase
import uuid
import bcrypt
import urllib.parse
from psycopg2.errors import IntegrityError
from collections import defaultdict
import asyncio
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
import pandas as pd
import base64
from io import BytesIO
from functools import lru_cache
from time import time

# Load environment variables
load_dotenv()

# Simple in-memory cache for frequently accessed data
_cache = {}
_cache_ttl = {}  # Time-to-live for cache entries
CACHE_TTL_SECONDS = 30  # Cache for 30 seconds (adjust as needed)

# JWT_SECRET is loaded but not logged for security
jwt_secret = os.getenv("JWT_SECRET")
if not jwt_secret:
    print("[WARNING] JWT_SECRET not found in environment variables")

app = FastAPI(title="Chatbot API", version="1.0.0")

# Configure CORS from environment
frontend_origins_env = os.getenv("FRONTEND_ORIGIN", "")
allowed_origins = [origin.strip() for origin in frontend_origins_env.split(",") if origin.strip()]

localhost_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
]

all_allowed_origins = list({*allowed_origins, *localhost_origins})

app.add_middleware(
    CORSMiddleware,
    allow_origins=all_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
)

if allowed_origins:
    print(f"[CORS] Enabled for origins: {all_allowed_origins}")
else:
    print("[CORS] FRONTEND_ORIGIN not set; defaulting to localhost origins only. Set FRONTEND_ORIGIN for deployed environments.")

# Security
security = HTTPBearer()

# Database connection pool for user management (MySQL/XAMPP compatible)
# Connection pooling significantly improves performance by reusing connections
_db_pool = None

def _ensure_ist_timezone(conn):
    """Ensure the MySQL connection uses IST (Indian Standard Time, UTC+5:30) timezone"""
    try:
        cursor = conn.cursor()
        # Check current timezone first
        cursor.execute("SELECT @@session.time_zone as tz, @@global.time_zone as global_tz, @@system_time_zone as system_tz")
        result = cursor.fetchone()
        current_tz = result[0] if result else None
        global_tz = result[1] if result and len(result) > 1 else None
        system_tz = result[2] if result and len(result) > 2 else None
        
        # Only set if not already IST (+05:30)
        if current_tz != '+05:30':
            cursor.execute("SET time_zone = '+05:30'")
            # Verify timezone is set correctly
            cursor.execute("SELECT @@session.time_zone as tz")
            verify_result = cursor.fetchone()
            if verify_result and verify_result[0] != '+05:30':
                print(f"[WARNING] Timezone not set to IST. Session: {verify_result[0]}, Global: {global_tz}, System: {system_tz}")
            elif current_tz != '+05:30':
                print(f"[DB] Timezone changed from {current_tz} to IST (Global: {global_tz}, System: {system_tz})")
        cursor.close()
    except Exception as e:
        print(f"[WARNING] Failed to set IST timezone: {e}")
        import traceback
        print(f"[WARNING] Traceback: {traceback.format_exc()}")

def get_db_connection():
    global _db_pool
    
    try:
        host = os.getenv("MYSQL_HOST")
        user = os.getenv("MYSQL_USER")
        password = os.getenv("MYSQL_PASSWORD")
        database = os.getenv("MYSQL_DATABASE")
        port = int(os.getenv("MYSQL_PORT", "3306"))

        missing = [name for name, value in [
            ("MYSQL_HOST", host),
            ("MYSQL_USER", user),
            ("MYSQL_PASSWORD", password),
            ("MYSQL_DATABASE", database),
        ] if not value]

        if missing:
            print(f"[ERROR] Missing database configuration environment variables: {', '.join(missing)}")
            return None

        # Initialize connection pool if not exists
        if _db_pool is None:
            try:
                from mysql.connector import pooling
                # Create a connection to test and set timezone
                test_conn = mysql.connector.connect(
                    host=host,
                    user=user,
                    password=password,
                    database=database,
                    port=port,
                    charset='utf8mb4',
                    collation='utf8mb4_unicode_ci'
                )
                test_cursor = test_conn.cursor()
                test_cursor.execute("SET time_zone = '+05:30'")
                test_cursor.close()
                test_conn.close()
                
                _db_pool = pooling.MySQLConnectionPool(
                    pool_name="mypool",
                    pool_size=10,  # Number of connections in the pool
                    pool_reset_session=False,  # Disable reset to preserve timezone
                    host=host,
                    user=user,
                    password=password,
                    database=database,
                    port=port,
                    charset='utf8mb4',
                    collation='utf8mb4_unicode_ci',
                    autocommit=False
                )
                print("[DB] Connection pool initialized successfully")
                
                # Initialize all connections in the pool with IST timezone
                # Get and return connections to initialize them
                init_conns = []
                for _ in range(min(3, 10)):  # Initialize first few connections
                    try:
                        conn = _db_pool.get_connection()
                        _ensure_ist_timezone(conn)
                        init_conns.append(conn)
                    except:
                        break
                # Return connections to pool
                for conn in init_conns:
                    try:
                        conn.close()
                    except:
                        pass
            except ImportError:
                print("[DB] Connection pooling not available, using single connections")
                _db_pool = "no_pool"
        
        # Use connection pool if available
        if _db_pool != "no_pool" and _db_pool is not None:
            try:
                conn = _db_pool.get_connection()
                # Set session timezone to IST for this connection
                _ensure_ist_timezone(conn)
                return conn
            except Error as e:
                print(f"[DB] Error getting connection from pool: {e}")
                # Fallback to single connection
                conn = mysql.connector.connect(
                    host=host,
                    user=user,
                    password=password,
                    database=database,
                    port=port,
                    charset='utf8mb4',
                    collation='utf8mb4_unicode_ci'
                )
                # Set session timezone to IST
                _ensure_ist_timezone(conn)
                return conn
        else:
            # Fallback: single connection (original behavior)
            connection = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database,
                port=port,
                charset='utf8mb4',
                collation='utf8mb4_unicode_ci'
            )
            # Set session timezone to IST
            _ensure_ist_timezone(connection)
            return connection
    except Error as e:
        print(f"Error connecting to MySQL/XAMPP: {e}")
        return None

# JWT functions (matching JS backend)
def create_jwt_token(user_id: str, email: str):
    from datetime import timezone, timedelta
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    payload = {
        "id": user_id,  # Match JS backend structure
        "email": email,
        "exp": datetime.now(ist_tz) + timedelta(days=1)  # Match JS backend (1 day)
    }
    return jwt.encode(payload, jwt_secret, algorithm="HS256")

def verify_jwt_token(token: str):
    try:
        payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Authentication dependency (matching JS backend)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = verify_jwt_token(token)
    # Add user_id field for compatibility
    payload["user_id"] = payload.get("id")
    return payload

# Pydantic models
class UserRegister(BaseModel):
    email: str
    password: str
    name: str

class UserLogin(BaseModel):
    email: str
    password: str

class ProjectCreate(BaseModel):
    projectName: str
    projectInfo: str
    dbHost: str
    dbUser: str
    dbPassword: str
    databaseName: str
    dbPort: Optional[str] = "3306"
    dbInfo: str
    botName: str
    botAvatar: Optional[str] = None

class AnalysisRequest(BaseModel):
    naturalLanguageQuery: str
    conversationId: Optional[str] = None
    projectId: str

class MessageCreate(BaseModel):
    conversationId: str
    role: str
    content: str
    queryType: Optional[str] = None
    generatedSql: Optional[str] = None
    userId: Optional[str] = None  # Will be set from current_user in the endpoint

class MessageUpdate(BaseModel):
    isImportant: bool

class ConversationCreate(BaseModel):
    projectId: str
    title: Optional[str] = None

class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    isArchived: Optional[bool] = None

# Conversation memory cache (similar to JS version)
conversation_memory_cache = {}

# Crypto functions for sensitive data
def encrypt(text: str) -> str:
    """Encrypt sensitive text data"""
    if text is None:
        return None
    
    # Get encryption key from environment
    encryption_key = os.getenv("ENCRYPTION_KEY")
    if not encryption_key:
        # Fallback to JWT secret if no dedicated encryption key
        encryption_key = jwt_secret
    
    if not encryption_key:
        # If no encryption key available, return as-is but log warning
        print("[WARNING] No encryption key available, storing data unencrypted")
        return text
    
    try:
        # Simple base64 encoding with key (for basic obfuscation)
        import base64
        import hashlib
        
        # Create a simple hash-based encryption
        key_hash = hashlib.sha256(encryption_key.encode()).digest()
        text_bytes = text.encode('utf-8')
        
        # XOR encryption with key hash
        encrypted_bytes = bytes(a ^ b for a, b in zip(text_bytes, key_hash * (len(text_bytes) // len(key_hash) + 1)))
        
        # Return base64 encoded result
        return base64.b64encode(encrypted_bytes).decode('utf-8')
    except Exception as e:
        print(f"[WARNING] Encryption failed: {e}, storing data unencrypted")
        return text

def decrypt(encrypted_text: str) -> str:
    """Decrypt sensitive text data"""
    if encrypted_text is None:
        return None
    
    # Get encryption key from environment
    encryption_key = os.getenv("ENCRYPTION_KEY")
    if not encryption_key:
        # Fallback to JWT secret if no dedicated encryption key
        encryption_key = jwt_secret
    
    if not encryption_key:
        # If no encryption key available, return as-is
        return encrypted_text
    
    try:
        # Check if text is actually encrypted (base64 encoded)
        import base64
        import hashlib
        
        # Try to decode base64
        encrypted_bytes = base64.b64decode(encrypted_text.encode('utf-8'))
        
        # Create the same key hash
        key_hash = hashlib.sha256(encryption_key.encode()).digest()
        
        # XOR decryption
        decrypted_bytes = bytes(a ^ b for a, b in zip(encrypted_bytes, key_hash * (len(encrypted_bytes) // len(key_hash) + 1)))
        
        return decrypted_bytes.decode('utf-8')
    except Exception:
        # If decryption fails, assume it's not encrypted
        return encrypted_text

# ============================================================================
# SECURITY FUNCTIONS: Data Anonymization & Classification
# ============================================================================

def classify_sensitive_columns(schema_info):
    """
    Automatically identify columns that might contain PII or sensitive data.
    Returns a list of column names that should be anonymized.
    """
    sensitive_columns = []
    sensitive_patterns = {
        'email': r'.*email.*',
        'phone': r'.*(phone|mobile|tel|telephone).*',
        'ssn': r'.*(ssn|social.*security).*',
        'credit_card': r'.*(card|credit|payment.*card).*',
        'id': r'.*(_id|id)$',
        'name': r'.*name$',
        'address': r'.*(address|street|city|zip|postal).*',
        'account': r'.*(account|account_number).*',
        'customer': r'.*(customer_id|customer_number).*',
        'order': r'.*(order_id|order_number|invoice).*',
        'password': r'.*(password|passwd|pwd).*',
        'token': r'.*(token|api_key|secret).*',
    }
    
    # Parse schema info to extract column names
    if isinstance(schema_info, str):
        # Extract column names from schema string
        import re
        # Look for column definitions in CREATE TABLE or column listings
        column_matches = re.findall(r'(\w+)\s+(?:VARCHAR|TEXT|CHAR|INT|BIGINT|DECIMAL|DATE|DATETIME)', schema_info, re.IGNORECASE)
        for col in column_matches:
            col_lower = col.lower()
            for pattern_name, pattern in sensitive_patterns.items():
                if re.match(pattern, col_lower, re.IGNORECASE):
                    sensitive_columns.append(col)
                    break
    
    return list(set(sensitive_columns))  # Remove duplicates

def anonymize_query_results(results, sensitive_columns=None):
    """
    Anonymize query results before sending to LLM.
    - Replace PII with placeholders
    - Aggregate individual records when possible
    - Remove sensitive identifiers
    """
    if sensitive_columns is None:
        sensitive_columns = ['email', 'phone', 'ssn', 'credit_card', 
                           'customer_id', 'order_id', 'account_number',
                           'customer_number', 'order_number', 'invoice_number',
                           'account', 'password', 'token', 'api_key', 'secret']
    
    if not results:
        return results
    
    anonymized = []
    
    # Handle different result formats
    if isinstance(results, str):
        # If results are already a string, try to parse or return as-is with basic filtering
        # This is a fallback - ideally results should be structured
        return results
    
    # Handle list of dictionaries (most common format)
    if isinstance(results, list) and len(results) > 0:
        for row in results:
            if isinstance(row, dict):
                anonymized_row = {}
                for key, value in row.items():
                    key_lower = key.lower()
                    # Check if column name matches sensitive patterns
                    is_sensitive = any(
                        col.lower() in key_lower or key_lower.endswith(col.lower()) 
                        for col in sensitive_columns
                    )
                    
                    if is_sensitive:
                        # Replace with placeholder
                        anonymized_row[key] = '[REDACTED]'
                    else:
                        anonymized_row[key] = value
                anonymized.append(anonymized_row)
            else:
                # If row is not a dict, keep as-is but log
                anonymized.append(row)
    
    return anonymized

def filter_sensitive_data_before_llm(query_results, schema_info=None):
    """
    Remove or aggregate sensitive data BEFORE sending to LLM.
    This is the main pre-filtering function.
    """
    # Step 1: Identify sensitive columns from schema
    sensitive_columns = []
    if schema_info:
        sensitive_columns = classify_sensitive_columns(schema_info)
    
    # Step 2: Anonymize the results
    sanitized_results = anonymize_query_results(query_results, sensitive_columns)
    
    # Step 3: Additional filtering - remove any remaining PII patterns
    if isinstance(sanitized_results, list):
        for row in sanitized_results:
            if isinstance(row, dict):
                for key, value in row.items():
                    if isinstance(value, str):
                        # Remove email patterns
                        if re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}', value):
                            row[key] = '[REDACTED_EMAIL]'
                        # Remove phone patterns
                        if re.search(r'\b(?:\+?[\d\s\-\(\)]{10,})\b', value):
                            row[key] = '[REDACTED_PHONE]'
                        # Remove credit card patterns
                        if re.search(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b', value):
                            row[key] = '[REDACTED_CARD]'
    
    return sanitized_results

# ============================================================================
# Database Reader functionality (PostgreSQL compatible)
# ============================================================================
class DatabaseAnalyzer:
    def __init__(self, db_host: str, db_user: str, db_password: str, db_name: str, db_port: str = "3306"):
        print(f"[DEBUG] DatabaseAnalyzer: Initializing connection...")
        
        # URL encode the password to handle special characters like @
        encoded_password = urllib.parse.quote_plus(db_password)
        
        # Determine database type based on port
        if db_port == "5432" or db_port == 5432:
            # PostgreSQL connection
            connection_string = f"postgresql+psycopg2://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
            print(f"[DEBUG] DatabaseAnalyzer: Using PostgreSQL connection")
        else:
            # MySQL connection (default)
            connection_string = f"mysql+pymysql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}?charset=utf8mb4"
            print(f"[DEBUG] DatabaseAnalyzer: Using MySQL connection")
        
        # Connection string is not logged for security reasons
        print(f"[DEBUG] DatabaseAnalyzer: Connection established")
        
        self.engine = create_engine(connection_string)
        self.db = SQLDatabase(self.engine, sample_rows_in_table_info=3)
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_api_key
        )
        
        # SQL pattern cache for consistency
        self.sql_patterns = {
            "monthly_sales": "SELECT DATE_FORMAT(created_at, '%Y-%m') AS month, SUM(CAST(JSON_UNQUOTE(JSON_EXTRACT(grandtotal, '$.paid_amount')) AS DECIMAL(10,2))) AS sales_amount FROM {table} WHERE created_at IS NOT NULL GROUP BY month ORDER BY month",
            "monthly_sales_analysis": "SELECT DATE_FORMAT(created_at, '%Y-%m') AS month, SUM(CAST(JSON_UNQUOTE(JSON_EXTRACT(grandtotal, '$.paid_amount')) AS DECIMAL(10,2))) AS sales_amount FROM {table} WHERE created_at IS NOT NULL GROUP BY month ORDER BY month",
            "sales_analysis": "SELECT DATE_FORMAT(created_at, '%Y-%m') AS month, SUM(CAST(JSON_UNQUOTE(JSON_EXTRACT(grandtotal, '$.paid_amount')) AS DECIMAL(10,2))) AS sales_amount FROM {table} WHERE created_at IS NOT NULL GROUP BY month ORDER BY month",
            "daily_sales": "SELECT DATE_FORMAT(created_at, '%Y-%m-%d') AS day, SUM(CAST(JSON_UNQUOTE(JSON_EXTRACT(grandtotal, '$.paid_amount')) AS DECIMAL(10,2))) AS sales_amount FROM {table} WHERE created_at IS NOT NULL GROUP BY day ORDER BY day",
            "yearly_sales": "SELECT DATE_FORMAT(created_at, '%Y') AS year, SUM(CAST(JSON_UNQUOTE(JSON_EXTRACT(grandtotal, '$.paid_amount')) AS DECIMAL(10,2))) AS sales_amount FROM {table} WHERE created_at IS NOT NULL GROUP BY year ORDER BY year",
            "count_by_category": "SELECT category, COUNT(*) AS record_count FROM {table} GROUP BY category ORDER BY record_count DESC",
            "total_revenue": "SELECT SUM(CAST(JSON_UNQUOTE(JSON_EXTRACT(grandtotal, '$.paid_amount')) AS DECIMAL(10,2))) AS total_revenue FROM {table}",
            "average_order": "SELECT AVG(CAST(JSON_UNQUOTE(JSON_EXTRACT(grandtotal, '$.paid_amount')) AS DECIMAL(10,2))) AS avg_order_value FROM {table}",
            "top_products": "SELECT product_name, COUNT(*) AS order_count, SUM(CAST(JSON_UNQUOTE(JSON_EXTRACT(grandtotal, '$.paid_amount')) AS DECIMAL(10,2))) AS total_sales FROM {table} GROUP BY product_name ORDER BY total_sales DESC LIMIT 10"
        }
        
        # Set system-level privacy and content moderation instructions
        self.system_prompt = """You are a data analyst assistant with content moderation requirements:

SECURITY RULES:
- Do not include SQL queries or table names in responses
- Do not show raw query results or technical details
- Provide natural language explanations and insights with accurate data

CONTENT MODERATION RULES:
- Never generate, describe, or reference adult content, pornography, or sexually explicit material
- Never provide information about illegal activities, drugs, violence, or harmful content
- Never share inappropriate, offensive, or NSFW (Not Safe For Work) content
- If asked about inappropriate topics, politely redirect to project-related questions
- Maintain professional, family-friendly language at all times
- Focus on legitimate data analysis and insights relevant to the project context
C
SQL SYNTAX RULES:
- Always use MySQL/MariaDB syntax only
- Use DATE_FORMAT() instead of STRFTIME() for date formatting
- Use CURDATE() instead of CURRENT_DATE() for current date
- Use MySQL-compatible date functions and operators
- Ensure all SQL queries are compatible with MariaDB 10.4+

JSON DATA HANDLING:
- If a column contains JSON data (like grandtotal), use JSON_EXTRACT() to get specific fields
- For revenue calculations, use: JSON_UNQUOTE(JSON_EXTRACT(grandtotal, '$.paid_amount'))
- Cast JSON extracted values to appropriate data types: CAST(JSON_UNQUOTE(...) AS DECIMAL(10,2))
- Always check if columns contain JSON before using them in calculations


Your role is to provide valuable insights based on the specific project context while maintaining strict privacy standards and appropriate content guidelines."""
        self.chain = create_sql_query_chain(self.llm, self.db)

    def _parse_query_result(self, result_str):
        """
        Parse query result string into structured format (list of dicts).
        Handles different result formats from SQLDatabase.run()
        """
        if not result_str:
            return []
        
        try:
            # Try to parse as JSON first (if result is JSON string)
            import json
            if result_str.strip().startswith('[') or result_str.strip().startswith('{'):
                return json.loads(result_str)
        except:
            pass
        
        # Parse tabular format (common SQLDatabase output)
        lines = result_str.strip().split('\n')
        if len(lines) < 2:
            return []
        
        # First line is usually headers
        headers = [h.strip() for h in lines[0].split('\t') if h.strip()]
        if not headers:
            # Try comma-separated
            headers = [h.strip() for h in lines[0].split(',') if h.strip()]
        
        result_list = []
        for line in lines[1:]:
            if not line.strip():
                continue
            values = [v.strip() for v in line.split('\t') if v.strip() or v == '']
            if not values:
                # Try comma-separated
                values = [v.strip() for v in line.split(',') if v.strip() or v == '']
            
            if len(values) == len(headers):
                row_dict = {}
                for i, header in enumerate(headers):
                    row_dict[header] = values[i] if i < len(values) else None
                result_list.append(row_dict)
        
        return result_list if result_list else [{"raw_result": result_str}]

    def _format_results_for_llm(self, results):
        """
        Format structured results (list of dicts) back to string format for LLM.
        """
        if not results:
            return "No results"
        
        if isinstance(results, str):
            return results
        
        if isinstance(results, list):
            if len(results) == 0:
                return "No results"
            
            # Format as table-like structure
            if isinstance(results[0], dict):
                # Get all unique keys
                all_keys = set()
                for row in results:
                    all_keys.update(row.keys())
                headers = sorted(list(all_keys))
                
                # Format as rows
                formatted_lines = []
                for row in results:
                    row_values = [str(row.get(h, '')) for h in headers]
                    formatted_lines.append('\t'.join(row_values))
                
                header_line = '\t'.join(headers)
                return header_line + '\n' + '\n'.join(formatted_lines)
            else:
                # If list of non-dicts, join them
                return '\n'.join(str(r) for r in results)
        
        return str(results)

    def detect_question_pattern(self, question: str):
        """Detect the type of question to use consistent SQL patterns"""
        question_lower = question.lower()
        
        # Sales analysis patterns (highest priority)
        if any(word in question_lower for word in ['sales analysis', 'sales data', 'monthly sales', 'sales breakdown']):
            return "monthly_sales_analysis"
        
        # Monthly sales patterns
        if any(word in question_lower for word in ['monthly', 'month', 'by month', 'each month']):
            if any(word in question_lower for word in ['sales', 'revenue', 'income', 'earnings']):
                return "monthly_sales"
        
        # Sales analysis patterns
        if any(word in question_lower for word in ['analysis', 'breakdown', 'trends', 'performance']):
            if any(word in question_lower for word in ['sales', 'revenue', 'income', 'earnings']):
                return "sales_analysis"
        
        # Daily sales patterns
        if any(word in question_lower for word in ['daily', 'day', 'by day', 'each day']):
            if any(word in question_lower for word in ['sales', 'revenue', 'income', 'earnings']):
                return "daily_sales"
        
        # Yearly sales patterns
        if any(word in question_lower for word in ['yearly', 'year', 'by year', 'each year', 'annual']):
            if any(word in question_lower for word in ['sales', 'revenue', 'income', 'earnings']):
                return "yearly_sales"
        
        # Count patterns
        if any(word in question_lower for word in ['count', 'number of', 'how many', 'total count']):
            if any(word in question_lower for word in ['category', 'type', 'group']):
                return "count_by_category"
        
        # Total revenue patterns
        if any(word in question_lower for word in ['total', 'sum', 'overall', 'grand total']):
            if any(word in question_lower for word in ['revenue', 'sales', 'income', 'earnings']):
                return "total_revenue"
        
        # Average patterns
        if any(word in question_lower for word in ['average', 'avg', 'mean']):
            if any(word in question_lower for word in ['order', 'sale', 'transaction']):
                return "average_order"
        
        # Top products patterns
        if any(word in question_lower for word in ['top', 'best', 'highest', 'most popular', 'leading']):
            if any(word in question_lower for word in ['product', 'item', 'service']):
                return "top_products"
        
        return None

    def execute_query(self, question: str, project_context: str = ""):
        # Validate user question for inappropriate content
        def validate_user_question(text):
            # General data analysis terms - these are always allowed
            analysis_whitelist = [
                'data', 'analysis', 'query', 'search', 'find', 'show', 'get',
                'count', 'sum', 'average', 'total', 'list', 'display', 'report',
                'compare', 'trend', 'pattern', 'insight', 'statistic', 'metric',
                'table', 'record', 'entry', 'item', 'result', 'information',
                'what', 'how', 'when', 'where', 'which', 'who', 'why'
            ]
            
            text_lower = text.lower()
            
            # If the question contains analysis terms, it's likely legitimate
            for term in analysis_whitelist:
                if term in text_lower:
                    return True
            
            # More specific patterns to avoid false positives
            inappropriate_patterns = [
                r'\b(?:porn|xxx|adult\s+content|nsfw|explicit\s+content)\b',
                r'\b(?:cocaine|heroin|marijuana|weed)\b',
                r'\b(?:kill\s+someone|murder|assault\s+someone)\b',
                r'\b(?:hack\s+into|crack\s+password|steal\s+data)\b',
                r'\b(?:illegal\s+activities|unauthorized\s+access)\b'
            ]
            
            for pattern in inappropriate_patterns:
                if re.search(pattern, text_lower):
                    return False
            return True
        
        # Check if user question is appropriate
        if not validate_user_question(question):
            return None, (
                "I apologize, but I cannot assist with that type of request. "
                "I'm designed to help with data analysis and insights based on your project. "
                "Please ask me about your data, request analysis, or ask questions about "
                "the information in your database."
            )
        
        try:
            # Step 1: Check if we can use a predefined pattern for consistency
            pattern = self.detect_question_pattern(question)
            schema_info = self.db.table_info
            
            if pattern and pattern in self.sql_patterns:
                # Use predefined pattern for maximum consistency
                print(f"[DEBUG] Using predefined pattern: {pattern}")
                prompt = (
                    f"You are a senior data analyst with expertise in MySQL/MariaDB. Here is the database schema:\n{schema_info}\n\n"
                    f"User question: {question}\n\n"
                    f"For this type of analysis, use this proven SQL approach:\n"
                    f"{self.sql_patterns[pattern]}\n\n"
                    "Instructions:\n"
                    "1. ALWAYS use aggregate functions (COUNT, SUM, AVG, MAX, MIN, GROUP BY)\n"
                    "2. NEVER return individual records - only aggregated statistics\n"
                    "3. NEVER include columns that might contain PII in SELECT statements (email, phone, customer_id, order_id, etc.)\n"
                    "4. Replace {table} with the appropriate table name from the schema\n"
                    "5. Execute the query and analyze the results\n"
                    "6. Return the query in a ```sql ... ``` block\n"
                    "7. Provide a comprehensive analysis based on the actual data\n"
                    "8. Focus on trends, patterns, and key insights from the results"
                )
            else:
                # Use the detailed prompt for other questions
                prompt = (
                    f"You are a senior data analyst with expertise in MySQL/MariaDB. Here is the database schema:\n{schema_info}\n\n"
                    f"User question: {question}\n\n"
                    "Generate SQL queries to answer the user's question:\n"
                    "1. If the user asks for a list, details, or specific records, use SELECT queries to return individual records\n"
                    "2. If the user asks for summaries, statistics, or aggregations, use aggregate functions (COUNT, SUM, AVG, MAX, MIN, GROUP BY)\n"
                    "3. IMPORTANT: Only SELECT the columns that the user specifically requested. If the user asks for 'employees with department and manager', only include employee name/ID, department, and manager - do NOT include email, phone, or other columns unless explicitly requested.\n"
                    "4. For monetary values, use: CAST(JSON_UNQUOTE(JSON_EXTRACT(grandtotal, '$.paid_amount')) AS DECIMAL(10,2))\n"
                    "5. For date grouping, use DATE_FORMAT(date_column, '%Y-%m') for monthly, '%Y-%m-%d' for daily\n"
                    "6. Always include ORDER BY for predictable results\n"
                    "7. Use descriptive column aliases when needed\n"
                    "8. Include WHERE clauses to filter data appropriately\n"
                    "9. Use JOINs when you need to combine data from multiple tables (e.g., employees with departments, managers)\n"
                    "10. Return each query in a separate ```sql ... ``` block\n"
                    "11. After the queries, analyze the results and provide insights\n"
                    "12. Base your analysis strictly on the query results\n"
                    "13. Only include columns in the SELECT statement that match what the user asked for"
                )
            
            messages = []
            # Add system-level privacy instructions
            messages.append({"role": "system", "content": self.system_prompt})
            if project_context:
                messages.append({"role": "system", "content": project_context})
            messages.append({"role": "user", "content": prompt})
            response = self.llm.invoke(messages)

            # Step 2: Extract SQL queries (same as Database Reader)
            content_str = response.content if isinstance(response.content, str) else str(response.content)
            sql_blocks = re.findall(r"```sql\s*(.*?)```", content_str, re.DOTALL | re.IGNORECASE)
            
            if not sql_blocks:
                return None, "Please try rephrasing your question or provide more specific details about what you'd like to analyze."

            results = []
            successful_queries = 0
            raw_results_list = []  # Store raw results for anonymization
            
            for i, sql in enumerate(sql_blocks, 1):
                sql = sql.strip()
                if sql:
                    try:
                        print(f"[SQL GENERATED] Query {i}: {sql}")
                        print(f"[DEBUG] Executing query {i}...")
                        result = self.db.run(sql)
                        
                        # Parse result string into structured format for anonymization
                        parsed_result = self._parse_query_result(result)
                        raw_results_list.append(parsed_result)
                        
                        results.append({
                            "query": sql,
                            "result": result,
                            "status": "success"
                        })
                        successful_queries += 1
                    except Exception as e:
                        print(f"[DEBUG] Query {i} failed: {str(e)}")
                        # Do not append failed queries or errors to results
                        continue

            if successful_queries == 0:
                return None, "I generated SQL queries but none of them executed successfully. This might be due to database structure differences or query syntax issues. Please try asking a simpler question or contact your database administrator."

            # ========================================================================
            # Format raw results for LLM analysis
            # ========================================================================
            print(f"[DEBUG] Formatting {len(raw_results_list)} query result sets for LLM")
            
            # Use raw results without any filtering or anonymization
            formatted_results_strings = []
            for item in results:
                formatted_results_strings.append(f"**Query:**\n{item['query']}\n**Result:**\n{item['result']}")
            
            formatted_results = "\n".join(formatted_results_strings)

            # Step 3: Ask LLM to analyze the results
            analysis_prompt = (
                f"Given the following SQL results (as Python lists):\n{formatted_results}\n"
                f"Original user question: {question}\n\n"
                "Write a detailed, data-driven answer to the original question. "
                "Reference specific numbers and findings. Do not invent or assume data. "
                "IMPORTANT: Only display the columns that the user specifically requested in their question. "
                "If the user asked for 'employees with department and manager', only show employee name/ID, department, and manager columns - do NOT show email, phone, or other columns unless the user explicitly asked for them. "
                "If the user asked for a list, show the complete list but only with the columns they requested. "
                "If the user's question is simple and only requires a direct answer, provide only the answer. Only provide suggestions or next steps if the question is analytical or open-ended. "
                "Provide accurate information based on the query results, showing only what was requested. "
                "When showing lists or tables, format them as properly formatted markdown tables with aligned columns. "
                "Use markdown table syntax with proper alignment (| column1 | column2 | column3 |) and separator rows (|--------|--------|--------|). "
                "Ensure all data is clearly visible and the table is easy to read. "
                "Only include columns in the table that match what the user asked for in their original question."
            )
            
            analysis = self.llm.invoke([
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": analysis_prompt}
            ])
            
            # Remove SQL code blocks and table names from the final response
            def remove_sql_blocks_and_tables(text, table_names=None):
                # Remove SQL code blocks (```sql ... ```) but preserve markdown tables
                text = re.sub(r"```sql[\s\S]*?```", "", text, flags=re.IGNORECASE)
                
                # Only remove table names that are not part of markdown tables
                # Markdown tables use | for columns, so we need to be careful
                # Split text into lines to preserve markdown table structure
                lines = text.split('\n')
                processed_lines = []
                in_markdown_table = False
                
                for line in lines:
                    # Check if this line is part of a markdown table (contains |)
                    if '|' in line and not line.strip().startswith('```'):
                        in_markdown_table = True
                        # For markdown table lines, only remove SQL code blocks, not table names
                        # Remove any backticks that might contain SQL table names, but preserve the table structure
                        line = re.sub(r"`([^`]+)`", r"\1", line)  # Remove backticks but keep content
                        processed_lines.append(line)
                    else:
                        in_markdown_table = False
                        # For non-table lines, remove table names more aggressively
                        # Remove any table names in backticks (but not in markdown tables)
                        line = re.sub(r"`[^`]+`", "", line)
                        # Remove known table names (case-insensitive) but not if they're part of data
                        if table_names:
                            for tbl in table_names:
                                # Only remove if it's a standalone word, not part of other text
                                line = re.sub(rf"\b{re.escape(tbl)}\b", "", line, flags=re.IGNORECASE)
                        processed_lines.append(line)
                
                text = '\n'.join(processed_lines)
                
                # Remove any '[REDACTED_TABLE]' tokens
                text = text.replace('[REDACTED_TABLE]', '')
                
                # Only collapse whitespace outside of markdown tables
                # Preserve markdown table formatting
                lines = text.split('\n')
                final_lines = []
                for line in lines:
                    if '|' in line:
                        # This is a markdown table line, preserve it as-is
                        final_lines.append(line)
                    else:
                        # For non-table lines, normalize whitespace
                        line = re.sub(r'\s+', ' ', line).strip()
                        if line:
                            final_lines.append(line)
                
                return '\n'.join(final_lines)

            # Use the response as-is without any data hiding
            final_response = remove_sql_blocks_and_tables(analysis.content)
            
            # Content validation - check if response contains inappropriate content
            def validate_content_safety(text):
                # More specific patterns to avoid false positives
                inappropriate_patterns = [
                    r'\b(?:porn|xxx|adult\s+content|nsfw|explicit\s+content)\b',
                    r'\b(?:cocaine|heroin|marijuana|weed)\b',
                    r'\b(?:kill\s+someone|murder|assault\s+someone)\b',
                    r'\b(?:hack\s+into|crack\s+password|steal\s+data)\b',
                    r'\b(?:illegal\s+activities|unauthorized\s+access)\b'
                ]
                
                text_lower = text.lower()
                for pattern in inappropriate_patterns:
                    if re.search(pattern, text_lower):
                        return False
                return True
            
            # If content is inappropriate, replace with safe message
            if not validate_content_safety(final_response):
                final_response = (
                    "I apologize, but I cannot provide information about that topic. "
                    "I'm designed to help with data analysis and insights based on your project. "
                    "Please ask me about your data, request analysis, or ask questions about "
                    "the information in your database."
                )
            
            # if successful_queries < len(sql_blocks):
            #     final_response += f"\n\n*Note: Some parts of the analysis couldn't be completed due to data access issues, but I've provided insights based on the available information.*"
            
            return results, final_response
            
        except Exception as e:
            error_message = (
                f"I encountered an error while analyzing your data: {str(e)}\n\n"
                f"**Troubleshooting Tips**:\n"
                f"• Make sure your question is clear and specific\n"
                f"• Try rephrasing your question in simpler terms\n"
                f"• Check if the database contains the data you're looking for\n"
                f"• Contact your system administrator if the problem persists"
            )
            return None, error_message

    def generate_visualization(self, question: str, project_context: str = ""):
        """Generate data visualization based on the question"""
        try:
            # First, get the data using the existing query logic
            results, analysis = self.execute_query(question, project_context)
            
            if not results or len(results) == 0:
                return None, "No data available for visualization"
            
            # Find the most suitable result for visualization
            print(f"[DEBUG] Visualization: Found {len(results)} results")
            for i, result in enumerate(results):
                print(f"[DEBUG] Result {i}: status={result.get('status')}, has_result={bool(result.get('result'))}")
                if result.get('result'):
                    query_result = result['result']
                    print(f"[DEBUG] Result {i} query_result type: {type(query_result)}, length: {len(query_result) if isinstance(query_result, list) else 'N/A'}")
            
            best_result = None
            for result in results:
                if result.get('status') == 'success' and result.get('result'):
                    query_result = result['result']
                    # Handle both list and string results
                    if isinstance(query_result, list) and len(query_result) > 0:
                        best_result = result
                        print(f"[DEBUG] Found suitable list result for visualization")
                        break
                    elif isinstance(query_result, str) and query_result.strip():
                        # Try to parse string result as JSON or Python data structure
                        try:
                            # If it's a JSON string, parse it
                            if query_result.strip().startswith('[') or query_result.strip().startswith('{'):
                                parsed_data = json.loads(query_result)
                                if isinstance(parsed_data, list) and len(parsed_data) > 0:
                                    result['result'] = parsed_data  # Update the result with parsed data
                                    best_result = result
                                    print(f"[DEBUG] Found suitable parsed JSON string result for visualization")
                                    break
                        except json.JSONDecodeError:
                            # If not JSON, try to parse as Python data structure using eval (safe for data)
                            try:
                                if query_result.strip().startswith('[') and query_result.strip().endswith(']'):
                                    # Use eval to parse Python data structures like [(Decimal('1910454.76'),)]
                                    # Provide necessary classes in the eval namespace
                                    eval_namespace = {
                                        'Decimal': Decimal,
                                        'datetime': datetime,
                                        'date': date,
                                        '__builtins__': {}
                                    }
                                    parsed_data = eval(query_result, eval_namespace)
                                    if isinstance(parsed_data, list) and len(parsed_data) > 0:
                                        # Convert to a more standard format for pandas
                                        converted_data = []
                                        for row in parsed_data:
                                            if isinstance(row, (tuple, list)):
                                                # Convert tuple/list to dict with column names
                                                row_dict = {}
                                                for i, value in enumerate(row):
                                                    # Convert Decimal and datetime objects to standard types
                                                    if hasattr(value, 'to_eng_string'):  # Decimal
                                                        row_dict[f'column_{i}'] = float(value)
                                                    elif hasattr(value, 'isoformat'):  # datetime
                                                        row_dict[f'column_{i}'] = str(value)
                                                    else:
                                                        row_dict[f'column_{i}'] = value
                                                converted_data.append(row_dict)
                                            else:
                                                # Single value row
                                                converted_data.append({'value': row})
                                        
                                        result['result'] = converted_data
                                        best_result = result
                                        print(f"[DEBUG] Found suitable parsed Python data result for visualization")
                                        break
                            except Exception as e:
                                print(f"[DEBUG] Skipping unparseable string result: {query_result[:100]}... (Error: {str(e)})")
                                continue
            
            if not best_result:
                print(f"[DEBUG] No suitable result found for visualization")
                return None, "No suitable data found for visualization"
            
            # Convert result to DataFrame
            df = pd.DataFrame(best_result['result'])
            
            if df.empty:
                return None, "No data available for visualization"
            
            # Improve column names for better chart labels
            df = self._improve_dataframe_columns(df, question)
            
            # Determine the best chart type based on data
            chart_type = self._determine_chart_type(df, question)
            
            # Generate the visualization
            chart_data = self._create_chart(df, chart_type, question)
            
            return chart_data, analysis
            
        except Exception as e:
            print(f"[ERROR] Visualization generation error: {str(e)}")
            return None, f"Failed to generate visualization: {str(e)}"
    
    def _improve_dataframe_columns(self, df, question):
        """Improve DataFrame column names for better chart labels"""
        try:
            # Get current column names
            current_cols = list(df.columns)
            new_cols = []
            
            # First, check the question context to determine appropriate column names
            question_lower = question.lower()
            
            # For sales analysis questions, use specific naming
            if any(word in question_lower for word in ['sales', 'revenue', 'income']):
                if any(word in question_lower for word in ['monthly', 'month', 'by month']):
                    # Sales + monthly = Month and Sales Amount
                    for i, col in enumerate(current_cols):
                        if i == 0:  # First column is date
                            new_cols.append('Month')
                        elif i == 1:  # Second column is sales
                            new_cols.append('Sales Amount')
                        else:
                            new_cols.append(f'Column {i+1}')
                else:
                    # Sales but not monthly
                    for i, col in enumerate(current_cols):
                        if i == 0:  # First column
                            new_cols.append('Category')
                        elif i == 1:  # Second column is sales
                            new_cols.append('Sales Amount')
                        else:
                            new_cols.append(f'Column {i+1}')
            else:
                # Analyze the data to determine appropriate column names
                for i, col in enumerate(current_cols):
                    col_lower = str(col).lower()
                    
                    # Check if it's a date/time column
                    if any(keyword in col_lower for keyword in ['date', 'time', 'month', 'year', 'day', 'created_at', 'updated_at']):
                        new_cols.append('Date/Time')
                    # Check if it's a sales/revenue column
                    elif any(keyword in col_lower for keyword in ['sales', 'revenue', 'amount', 'total', 'sum', 'paid_amount', 'grandtotal']):
                        new_cols.append('Sales Amount')
                    # Check if it's a count column
                    elif any(keyword in col_lower for keyword in ['count', 'number', 'quantity', 'records']):
                        new_cols.append('Count')
                    # Check if it's a category column
                    elif any(keyword in col_lower for keyword in ['category', 'type', 'name', 'product', 'item']):
                        new_cols.append('Category')
                    # Check if it's a numeric value
                    elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        new_cols.append('Value')
                    # Default fallback
                    else:
                        new_cols.append(f'Column {i+1}')
            
            # Apply the new column names
            df.columns = new_cols
            
            return df
            
        except Exception as e:
            print(f"[DEBUG] Error improving column names: {str(e)}")
            return df
    
    def _determine_chart_type(self, df, question):
        """Determine the best chart type based on data and question"""
        import numpy as np
        question_lower = question.lower()
        
        # Check for specific chart type requests
        if any(word in question_lower for word in ['pie', 'pie chart']):
            return 'pie'
        elif any(word in question_lower for word in ['bar', 'bar chart']):
            return 'bar'
        elif any(word in question_lower for word in ['line', 'line chart', 'trend']):
            return 'line'
        elif any(word in question_lower for word in ['scatter', 'scatter plot']):
            return 'scatter'
        elif any(word in question_lower for word in ['histogram', 'distribution']):
            return 'histogram'
        
        # Auto-determine based on data structure
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) >= 2:
            return 'scatter'
        elif len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            if len(df) <= 10:
                return 'pie'
            else:
                return 'bar'
        elif len(numeric_cols) >= 1:
            return 'histogram'
        else:
            return 'bar'
    
    def _create_chart(self, df, chart_type, question):
        """Create the actual chart"""
        try:
            import numpy as np
            plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get numeric and categorical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            # Set better colors and styling
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            if chart_type == 'pie':
                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    # Use first categorical as labels, first numeric as values
                    label_col = categorical_cols[0]
                    value_col = numeric_cols[0]
                    
                    # Create pie chart with enhanced styling
                    wedges, texts, autotexts = ax.pie(
                        df[value_col], 
                        labels=df[label_col], 
                        autopct='%1.1f%%', 
                        startangle=90,
                        colors=colors[:len(df)],
                        explode=[0.05] * len(df),  # Slight separation
                        shadow=True,
                        textprops={'fontsize': 10, 'fontweight': 'bold'}
                    )
                    
                    # Enhance title and add legend
                    title = f'{question[:60]}...' if len(question) > 60 else question
                    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
                    
                    # Add legend with better positioning
                    ax.legend(wedges, df[label_col], 
                             title=f"{label_col.replace('_', ' ').title()}",
                             loc="center left", 
                             bbox_to_anchor=(1, 0, 0.5, 1),
                             fontsize=10)
                    
                    # Make percentage text more readable
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                        autotext.set_fontsize(9)
                else:
                    return None
                    
            elif chart_type == 'bar':
                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    label_col = categorical_cols[0]
                    value_col = numeric_cols[0]
                    
                    # Create bar chart with enhanced styling
                    bars = ax.bar(df[label_col], df[value_col], 
                                 color=colors[:len(df)], 
                                 alpha=0.8,
                                 edgecolor='black',
                                 linewidth=0.5)
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{height:,.0f}' if height >= 1 else f'{height:.2f}',
                               ha='center', va='bottom', fontweight='bold', fontsize=9)
                    
                    # Enhanced labels and title
                    ax.set_xlabel(f"{label_col.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
                    ax.set_ylabel(f"{value_col.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
                    title = f'{question[:60]}...' if len(question) > 60 else question
                    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
                    
                    # Rotate x-axis labels for better readability
                    plt.xticks(rotation=45, ha='right')
                    
                    # Add grid for better readability
                    ax.grid(True, alpha=0.3, axis='y')
                    ax.set_axisbelow(True)
                else:
                    return None
                    
            elif chart_type == 'line':
                if len(numeric_cols) >= 2:
                    x_col = numeric_cols[0]
                    y_col = numeric_cols[1]
                    
                    # Create line chart with enhanced styling
                    line = ax.plot(df[x_col], df[y_col], 
                                  marker='o', 
                                  linewidth=2.5, 
                                  markersize=6,
                                  color=colors[0],
                                  markerfacecolor=colors[1],
                                  markeredgecolor='black',
                                  markeredgewidth=1)
                    
                    # Add data point labels
                    for i, (x, y) in enumerate(zip(df[x_col], df[y_col])):
                        ax.annotate(f'{y:,.0f}' if y >= 1 else f'{y:.2f}', 
                                   (x, y), 
                                   textcoords="offset points", 
                                   xytext=(0,10), 
                                   ha='center',
                                   fontsize=8,
                                   fontweight='bold')
                    
                    # Enhanced labels and title
                    ax.set_xlabel(f"{x_col.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
                    ax.set_ylabel(f"{y_col.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
                    title = f'{question[:60]}...' if len(question) > 60 else question
                    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
                    
                    # Add grid and legend
                    ax.grid(True, alpha=0.3)
                    ax.legend([f"{y_col.replace('_', ' ').title()}"], 
                             loc='upper right', fontsize=10)
                else:
                    return None
                    
            elif chart_type == 'scatter':
                if len(numeric_cols) >= 2:
                    x_col = numeric_cols[0]
                    y_col = numeric_cols[1]
                    
                    # Create scatter plot with enhanced styling
                    scatter = ax.scatter(df[x_col], df[y_col], 
                                        c=colors[0], 
                                        alpha=0.7, 
                                        s=60,
                                        edgecolors='black',
                                        linewidth=0.5)
                    
                    # Add trend line if there are enough points
                    if len(df) > 2:
                        z = np.polyfit(df[x_col], df[y_col], 1)
                        p = np.poly1d(z)
                        ax.plot(df[x_col], p(df[x_col]), 
                               color=colors[1], 
                               linestyle='--', 
                               alpha=0.8, 
                               linewidth=2,
                               label='Trend Line')
                    
                    # Enhanced labels and title
                    ax.set_xlabel(f"{x_col.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
                    ax.set_ylabel(f"{y_col.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
                    title = f'{question[:60]}...' if len(question) > 60 else question
                    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
                    
                    # Add grid and legend
                    ax.grid(True, alpha=0.3)
                    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                                 markerfacecolor=colors[0], markersize=8,
                                                 label=f"{y_col.replace('_', ' ').title()}")]
                    if len(df) > 2:
                        legend_elements.append(plt.Line2D([0], [0], color=colors[1], 
                                                         linestyle='--', linewidth=2,
                                                         label='Trend Line'))
                    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
                else:
                    return None
                    
            elif chart_type == 'histogram':
                if len(numeric_cols) > 0:
                    value_col = numeric_cols[0]
                    
                    # Create histogram with enhanced styling
                    n, bins, patches = ax.hist(df[value_col], 
                                              bins=min(20, len(df)//2 + 1), 
                                              alpha=0.7, 
                                              color=colors[0],
                                              edgecolor='black',
                                              linewidth=0.5)
                    
                    # Color bars based on height
                    for i, (bar_height, patch) in enumerate(zip(n, patches)):
                        if bar_height > 0:
                            patch.set_facecolor(colors[i % len(colors)])
                    
                    # Add statistics text box
                    mean_val = df[value_col].mean()
                    std_val = df[value_col].std()
                    median_val = df[value_col].median()
                    
                    stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nMedian: {median_val:.2f}'
                    ax.text(0.02, 0.98, stats_text, 
                           transform=ax.transAxes, 
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                           fontsize=9, fontweight='bold')
                    
                    # Enhanced labels and title
                    ax.set_xlabel(f"{value_col.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
                    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                    title = f'{question[:60]}...' if len(question) > 60 else question
                    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
                    
                    # Add grid and legend
                    ax.grid(True, alpha=0.3, axis='y')
                    ax.legend([f"{value_col.replace('_', ' ').title()} Distribution"], 
                             loc='upper right', fontsize=10)
                else:
                    return None
            
            # Ensure proper layout with extra padding for legends
            plt.tight_layout()
            plt.subplots_adjust(right=0.85)  # Make room for legends
            
            # Convert to base64 with high quality
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            # Create descriptive title
            chart_type_names = {
                'pie': 'Pie Chart',
                'bar': 'Bar Chart', 
                'line': 'Line Chart',
                'scatter': 'Scatter Plot',
                'histogram': 'Histogram'
            }
            
            chart_title = f"{chart_type_names.get(chart_type, 'Chart')}: {question[:80]}{'...' if len(question) > 80 else ''}"
            
            return {
                'type': chart_type,
                'data': image_base64,
                'title': chart_title
            }
            
        except Exception as e:
            print(f"[ERROR] Chart creation error: {str(e)}")
            return None

# --- WebSocket Real-Time Updates ---
# Track connected websockets per project
project_ws_connections = defaultdict(set)  # project_id: set of WebSocket

async def broadcast_project_update(project_id: str, event: dict):
    """Broadcast an event to all connected clients for a project."""
    websockets = list(project_ws_connections[project_id])
    for ws in websockets:
        try:
            await ws.send_json(event)
        except Exception:
            # Remove broken connection
            project_ws_connections[project_id].remove(ws)

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await websocket.accept()
    project_ws_connections[project_id].add(websocket)
    try:
        while True:
            # Keep the connection alive; optionally handle pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        project_ws_connections[project_id].remove(websocket)
    except Exception:
        project_ws_connections[project_id].remove(websocket)

# --- In your message and importance endpoints, call broadcast_project_update ---
# Example usage after creating/updating a message:
# await broadcast_project_update(project_id, {"type": "chat_update"})
# await broadcast_project_update(project_id, {"type": "important_update"})

# You will need to make the relevant endpoints async and call the broadcast function after DB changes.

# API Routes

@app.get("/")
async def root():
    return {"message": "Python Backend is running!", "status": "healthy"}

@app.get("/api/health")
async def health_check():
    from datetime import timezone, timedelta
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    return {"status": "healthy", "backend": "python", "timestamp": datetime.now(ist_tz).isoformat()}

@app.post("/api/test-analyze")
async def test_analyze():
    """Simple test endpoint to verify the backend is working"""
    from datetime import timezone, timedelta
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    return {
        "success": True,
        "message": "Test endpoint working",
        "timestamp": datetime.now(ist_tz).isoformat()
    }

@app.post("/api/auth/register")
async def register(user_data: UserRegister):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    cursor = conn.cursor(dictionary=True)
    
    # Check if user already exists
    cursor.execute("SELECT id FROM users WHERE email = %s", (user_data.email,))
    if cursor.fetchone():
        raise HTTPException(status_code=409, detail="A user with this email already exists.")
    
    # Generate UUID for user ID (matching JS backend)
    user_id = str(uuid.uuid4())
    
    # Hash password with bcrypt (matching JS backend)
    hashed_password = bcrypt.hashpw(user_data.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    # Insert user with UUID
    cursor.execute(
        "INSERT INTO users (id, name, email, password) VALUES (%s, %s, %s, %s)",
        (user_id, user_data.name, user_data.email, hashed_password)
    )
    conn.commit()
    
    token = create_jwt_token(user_id, user_data.email)
    
    cursor.close()
    conn.close()
    
    return {
        "success": True, 
        "message": "User registered successfully.",
        "user": {"id": user_id, "email": user_data.email, "name": user_data.name},
        "token": token
    }

@app.post("/api/auth/login")
async def login(user_data: UserLogin):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    cursor = conn.cursor(dictionary=True)
    
    # Check user credentials
    cursor.execute("SELECT id, email, name, password FROM users WHERE email = %s", (user_data.email,))
    user = cursor.fetchone()
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify password with bcrypt (matching JS backend)
    try:
        is_password_match = bcrypt.checkpw(user_data.password.encode('utf-8'), user["password"].encode('utf-8'))
        if not is_password_match:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    except Exception as e:
        print(f"Password verification error: {e}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_jwt_token(user["id"], user["email"])
    
    cursor.close()
    conn.close()
    
    return {
        "success": True, 
        "message": "Login successful.",
        "user": {"id": user["id"], "email": user["email"], "name": user["name"]},
        "token": token
    }

@app.post("/api/projects")
async def create_project(project_data: ProjectCreate, current_user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    cursor = conn.cursor(dictionary=True)
    try:
        # Insert project with actual database structure
        project_id = str(uuid.uuid4())  # Generate UUID for project ID
        cursor.execute(
            """INSERT INTO projects (id, user_id, name, project_info, db_host, db_user, db_password, db_name, db_port, db_info, bot_name, bot_avatar) 
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (project_id, current_user["user_id"], project_data.projectName, project_data.projectInfo,
             encrypt(project_data.dbHost), encrypt(project_data.dbUser), 
             encrypt(project_data.dbPassword), encrypt(project_data.databaseName),
             project_data.dbPort, project_data.dbInfo, project_data.botName, project_data.botAvatar)
        )
        conn.commit()
        
        # Invalidate cache for this user's projects
        cache_key = f"projects_{current_user['user_id']}"
        if cache_key in _cache:
            del _cache[cache_key]
        if cache_key in _cache_ttl:
            del _cache_ttl[cache_key]
        
        return {"success": True, "project_id": project_id, "message": "Project created successfully"}
    except IntegrityError as e:
        if "Duplicate entry" in str(e):
            raise HTTPException(status_code=400, detail="A project with this name already exists for this user.")
        else:
            raise
    finally:
        cursor.close()
        conn.close()

@app.get("/api/projects")
async def get_projects(current_user: dict = Depends(get_current_user)):
    # Check cache first
    cache_key = f"projects_{current_user['user_id']}"
    current_time = time()
    
    if cache_key in _cache and cache_key in _cache_ttl:
        if current_time < _cache_ttl[cache_key]:
            return {"success": True, "projects": _cache[cache_key], "cached": True}
    
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Optimized query - only select needed columns
        cursor.execute("SELECT id, user_id, name, project_info, db_info, bot_name, bot_avatar, created_at, updated_at FROM projects WHERE user_id = %s ORDER BY updated_at DESC", (current_user["user_id"],))
        projects = cursor.fetchall()
        
        # Map column names but DO NOT decrypt sensitive data
        for project in projects:
            project["project_name"] = project["name"]  # Map 'name' to 'project_name' for frontend
            project["db_port"] = "3306"  # Default port since it's not in your table
        
        cursor.close()
        
        # Cache the result
        _cache[cache_key] = projects
        _cache_ttl[cache_key] = current_time + CACHE_TTL_SECONDS
        
        return {"success": True, "projects": projects, "cached": False}
    finally:
        conn.close()

@app.get("/api/projects/{project_id}")
async def get_project(project_id: str, current_user: dict = Depends(get_current_user)):
    # Check cache first
    cache_key = f"project_{project_id}_{current_user['user_id']}"
    current_time = time()
    
    if cache_key in _cache and cache_key in _cache_ttl:
        if current_time < _cache_ttl[cache_key]:
            return {"success": True, "project": _cache[cache_key], "cached": True}
    
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Optimized query with explicit column selection
        cursor.execute("SELECT * FROM projects WHERE id = %s AND user_id = %s", (project_id, current_user["user_id"]))
        project = cursor.fetchone()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Map column names but DO NOT decrypt sensitive data
        project["project_name"] = project["name"]  # Map 'name' to 'project_name' for frontend
        project["db_port"] = project.get("db_port", "3306")  # Use actual port from database
        
        cursor.close()
        
        # Cache the result
        _cache[cache_key] = project
        _cache_ttl[cache_key] = current_time + CACHE_TTL_SECONDS
        
        return {"success": True, "project": project, "cached": False}
    finally:
        conn.close()

@app.put("/api/projects/{project_id}")
async def update_project(project_id: str, project_data: ProjectCreate, current_user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Check if project exists and belongs to user, and get current values
        cursor.execute("SELECT * FROM projects WHERE id = %s AND user_id = %s", (project_id, current_user["user_id"]))
        existing_project = cursor.fetchone()
        if not existing_project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Prepare update values - use new values if provided, otherwise keep existing values
        update_values = {
            'name': project_data.projectName if project_data.projectName and project_data.projectName.strip() else existing_project['name'],
            'project_info': project_data.projectInfo if project_data.projectInfo and project_data.projectInfo.strip() else existing_project['project_info'],
            'db_host': encrypt(project_data.dbHost) if project_data.dbHost and project_data.dbHost.strip() else existing_project['db_host'],
            'db_user': encrypt(project_data.dbUser) if project_data.dbUser and project_data.dbUser.strip() else existing_project['db_user'],
            'db_password': encrypt(project_data.dbPassword) if project_data.dbPassword and project_data.dbPassword.strip() else existing_project['db_password'],
            'db_name': encrypt(project_data.databaseName) if project_data.databaseName and project_data.databaseName.strip() else existing_project['db_name'],
            'db_port': project_data.dbPort if project_data.dbPort and project_data.dbPort.strip() else existing_project.get('db_port', '3306'),
            'db_info': project_data.dbInfo if project_data.dbInfo and project_data.dbInfo.strip() else existing_project['db_info'],
            'bot_name': project_data.botName if project_data.botName and project_data.botName.strip() else existing_project['bot_name'],
            'bot_avatar': project_data.botAvatar if project_data.botAvatar and project_data.botAvatar.strip() else existing_project['bot_avatar']
        }
        
        # Update project with preserved values
        cursor.execute(
            """UPDATE projects SET name = %s, project_info = %s, db_host = %s, db_user = %s, 
               db_password = %s, db_name = %s, db_port = %s, db_info = %s, bot_name = %s, bot_avatar = %s 
               WHERE id = %s AND user_id = %s""",
            (update_values['name'], update_values['project_info'],
             update_values['db_host'], update_values['db_user'], 
             update_values['db_password'], update_values['db_name'], update_values['db_port'],
             update_values['db_info'], update_values['bot_name'], update_values['bot_avatar'],
             project_id, current_user["user_id"])
        )
        conn.commit()
        
        # Invalidate cache for this user's projects and this specific project
        projects_cache_key = f"projects_{current_user['user_id']}"
        project_cache_key = f"project_{project_id}_{current_user['user_id']}"
        for key in [projects_cache_key, project_cache_key]:
            if key in _cache:
                del _cache[key]
            if key in _cache_ttl:
                del _cache_ttl[key]
        
        cursor.close()
        return {"success": True, "message": "Project updated successfully"}
    finally:
        conn.close()

@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str, current_user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Check if project exists and belongs to user
        cursor.execute("SELECT id FROM projects WHERE id = %s AND user_id = %s", (project_id, current_user["user_id"]))
        project = cursor.fetchone()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Delete project
        cursor.execute("DELETE FROM projects WHERE id = %s AND user_id = %s", (project_id, current_user["user_id"]))
        conn.commit()
        
        # Invalidate cache for this user's projects and this specific project
        projects_cache_key = f"projects_{current_user['user_id']}"
        project_cache_key = f"project_{project_id}_{current_user['user_id']}"
        for key in [projects_cache_key, project_cache_key]:
            if key in _cache:
                del _cache[key]
            if key in _cache_ttl:
                del _cache_ttl[key]
        
        cursor.close()
        return {"success": True, "message": "Project deleted successfully"}
    finally:
        conn.close()

def is_greeting(prompt):
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings']
    prompt_lower = prompt.strip().lower()
    return any(prompt_lower.startswith(greet) for greet in greetings)

def is_thanks_message(prompt):
    """Detect if the message is a thanks/thank you message"""
    thanks_patterns = [
        r'\b(?:thanks?|thank\s+you|thx|ty|grateful|appreciate)\b',
        r'\b(?:good|great|excellent|awesome|perfect|nice|wonderful|amazing)\s+(?:job|work|help|assistance|support)\b',
        r'\b(?:that\s+)?(?:was|is)\s+(?:helpful|useful|great|good|perfect)\b',
        r'\b(?:exactly|precisely|perfectly)\s+(?:what\s+)?(?:i\s+)?(?:needed|wanted|was\s+looking\s+for)\b'
    ]
    prompt_lower = prompt.strip().lower()
    return any(re.search(pattern, prompt_lower) for pattern in thanks_patterns)

def is_empty_or_meaningless(prompt):
    """Detect if the message is empty, very short, or meaningless"""
    prompt_clean = prompt.strip()
    
    # Check if empty or very short
    if len(prompt_clean) < 3:
        return True
    
    # Check for meaningless patterns
    meaningless_patterns = [
        r'^\s*[.!?,\s]*\s*$',  # Only punctuation and spaces
        r'^\s*(?:ok|k|yes|no|maybe|sure|alright|fine|good|bad|nice|cool|wow|omg|lol|haha|hehe)\s*$',
        r'^\s*(?:test|testing|hello|hi|hey)\s*$',  # Simple test messages
        r'^\s*[a-zA-Z]{1,2}\s*$',  # Single letters or very short words
    ]
    
    prompt_lower = prompt_clean.lower()
    return any(re.search(pattern, prompt_lower) for pattern in meaningless_patterns)

def generate_follow_up_response(project_context, user_message, llm):
    """Generate a contextual follow-up response using the LLM"""
    try:
        # Create a prompt for generating follow-up responses
        follow_up_prompt = f"""
You are a helpful AI business consultant. The user has just said: "{user_message}"

Based on this interaction, generate a friendly, professional follow-up response that:
1. Acknowledges their message appropriately
2. Offers to help with further assistance
3. Suggests specific ways you can help with their business data analysis
4. Maintains a professional yet warm tone
5. Is concise (1-2 sentences maximum)

Project Context: {project_context}

Generate only the response text, no additional formatting or explanations.
"""
        
        response = llm.invoke([
            {"role": "system", "content": "You are a helpful AI business consultant. Provide concise, professional responses."},
            {"role": "user", "content": follow_up_prompt}
        ])
        
        # Clean up the response
        response_text = response.content.strip()
        if response_text.startswith('"') and response_text.endswith('"'):
            response_text = response_text[1:-1]
        
        return response_text
        
    except Exception as e:
        print(f"[WARNING] Failed to generate follow-up response: {str(e)}")
        # Fallback response
        return "Is there anything else I can help you with regarding your data analysis?"

# Add this helper function near your DB helpers

def get_last_10_messages(conversation_id: str, limit: int = 10):
    """Optimized function to get last messages with connection pooling"""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(dictionary=True)
        # Ensure IST timezone
        _ensure_ist_timezone(conn)
        # Use UNIX_TIMESTAMP with CONVERT_TZ to get IST timestamps
        from datetime import timezone, timedelta
        ist_tz = timezone(timedelta(hours=5, minutes=30))
        cursor.execute("""
            SELECT id, role, content, UNIX_TIMESTAMP(CONVERT_TZ(created_at, @@session.time_zone, '+05:30')) as created_at_ts, query_type, generated_sql
            FROM messages
            WHERE conversation_id = %s AND is_archived = FALSE
            ORDER BY created_at ASC
            LIMIT %s
        """, (conversation_id, limit))
        messages = cursor.fetchall()
        
        # Convert UNIX_TIMESTAMP to IST datetime with timezone
        for msg in messages:
            if msg.get('created_at_ts'):
                dt = datetime.fromtimestamp(msg['created_at_ts'], tz=ist_tz)
                msg['created_at'] = dt.isoformat()
                msg.pop('created_at_ts', None)
        
        cursor.close()
        # Already in chronological order (oldest to newest)
        return messages
    finally:
        conn.close()

@app.post("/api/analyze-data")
async def analyze_data(request: AnalysisRequest, current_user: dict = Depends(get_current_user)):
    try:
        print(f"[DEBUG] Analyze request received: {request.naturalLanguageQuery}")
        print(f"[DEBUG] Project ID: {request.projectId}")
        print(f"[DEBUG] User ID: {current_user['user_id']}")
        
        conversation_id = request.conversationId or str(uuid.uuid4())
        is_first_message = not request.conversationId

        # Fetch project details early for dynamic greeting - use cache if available
        cache_key = f"project_{request.projectId}_{current_user['user_id']}"
        current_time = time()
        
        if cache_key in _cache and cache_key in _cache_ttl and current_time < _cache_ttl[cache_key]:
            project = _cache[cache_key]
        else:
            conn = get_db_connection()
            if not conn:
                raise HTTPException(status_code=500, detail="Database connection failed")
            try:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT * FROM projects WHERE id = %s AND user_id = %s", (request.projectId, current_user["user_id"]))
                project = cursor.fetchone()
                cursor.close()
                if project:
                    # Cache the project
                    _cache[cache_key] = project
                    _cache_ttl[cache_key] = current_time + CACHE_TTL_SECONDS
            finally:
                conn.close()
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        project_name = project.get('name', 'your project')
        project_info = project.get('project_info', '')
        project_ref = f" (Project: {project_name})" if project_name else ""
        project_info_ref = f"\nProject Info: {project_info}" if project_info else ""

        # Check if this is a data analysis question (even on first message)
        is_data_question = not (
            is_greeting(request.naturalLanguageQuery) or
            is_thanks_message(request.naturalLanguageQuery) or
            is_empty_or_meaningless(request.naturalLanguageQuery)
        )
        
        greeting = f"👋 Hello! Welcome to{project_ref}!"
        general_response = f"How can I assist you today? You can ask me about your data, request insights."
        
        # Always store the user message (even for greetings/thanks)
        def store_user_and_ai_message(user_content, ai_content, ai_query_type="GENERAL"):
            user_message_id = None
            ai_message_id = None
            user_created_at = None
            ai_created_at = None
            try:
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor(dictionary=True)
                    # Create conversation if it doesn't exist
                    cursor.execute("SELECT id FROM conversations WHERE id = %s", (conversation_id,))
                    existing_conversation = cursor.fetchone()
                    if not existing_conversation:
                        first_message = user_content.strip() if user_content else ''
                        from datetime import timezone, timedelta
                        ist_tz = timezone(timedelta(hours=5, minutes=30))
                        title = (first_message[:60] + '...') if len(first_message) > 60 else first_message or f"Conversation {datetime.now(ist_tz).strftime('%Y-%m-%d %H:%M')}"
                        cursor.execute(
                            "INSERT INTO conversations (id, user_id, project_id, title) VALUES (%s, %s, %s, %s)",
                            (conversation_id, current_user["user_id"], request.projectId, title)
                        )
                    # Ensure IST timezone before inserting
                    _ensure_ist_timezone(conn)
                    
                    # Store user message with explicit IST timestamp using MySQL's NOW() (respects session timezone)
                    user_message_id = str(uuid.uuid4())
                    cursor.execute(
                        "INSERT INTO messages (id, conversation_id, user_id, role, content, created_at) VALUES (%s, %s, %s, %s, %s, NOW())",
                        (user_message_id, conversation_id, current_user["user_id"], 'human', user_content)
                    )
                    # Store AI response with explicit IST timestamp using MySQL's NOW() (respects session timezone)
                    ai_message_id = str(uuid.uuid4())
                    cursor.execute(
                        "INSERT INTO messages (id, conversation_id, user_id, role, content, query_type, created_at) VALUES (%s, %s, %s, %s, %s, %s, NOW())",
                        (ai_message_id, conversation_id, current_user["user_id"], 'ai', ai_content, ai_query_type)
                    )
                    # Update conversation timestamp with IST
                    cursor.execute(
                        "UPDATE conversations SET updated_at = NOW() WHERE id = %s",
                        (conversation_id,)
                    )
                    conn.commit()
                    # Fetch created_at timestamps for user and AI messages BEFORE closing cursor/conn
                    # Use UNIX_TIMESTAMP with CONVERT_TZ to get IST timestamp
                    from datetime import timezone, timedelta
                    ist_tz = timezone(timedelta(hours=5, minutes=30))
                    _ensure_ist_timezone(conn)
                    cursor.execute("SELECT UNIX_TIMESTAMP(CONVERT_TZ(created_at, @@session.time_zone, '+05:30')) as ts FROM messages WHERE id = %s", (user_message_id,))
                    user_created_at_row = cursor.fetchone()
                    if user_created_at_row and user_created_at_row.get("ts"):
                        dt = datetime.fromtimestamp(user_created_at_row["ts"], tz=ist_tz)
                        user_created_at = dt.isoformat()
                    else:
                        user_created_at = datetime.now(ist_tz).isoformat()
                    cursor.execute("SELECT UNIX_TIMESTAMP(CONVERT_TZ(created_at, @@session.time_zone, '+05:30')) as ts FROM messages WHERE id = %s", (ai_message_id,))
                    ai_created_at_row = cursor.fetchone()
                    if ai_created_at_row and ai_created_at_row.get("ts"):
                        dt = datetime.fromtimestamp(ai_created_at_row["ts"], tz=ist_tz)
                        ai_created_at = dt.isoformat()
                    else:
                        ai_created_at = datetime.now(ist_tz).isoformat()
                    cursor.close()
                    conn.close()
            except Exception as e:
                print(f"[WARNING] Failed to store messages in database: {str(e)}")
                user_message_id = None
                ai_message_id = None
                user_created_at = None
                ai_created_at = None
            return user_message_id, ai_message_id, user_created_at, ai_created_at

        # If it's a data question, analyze it even on first message
        if is_first_message and is_data_question:
            prepend_greeting = True  # We'll prepend the greeting to the analysis
            # Do NOT store user and AI message here; will store after analysis below
        elif is_first_message:
            # Only show generic greeting for non-data questions on first message
            prepend_greeting = False
            # Store user message and generic response (for greetings, thanks, etc.)
            user_message_id, ai_message_id, user_created_at, ai_created_at = store_user_and_ai_message(request.naturalLanguageQuery, general_response, "GENERAL")
            return {
                "success": True,
                "data": {
                    "conversationId": conversation_id,
                    "analysis": general_response,
                    "queryType": "GENERAL",
                    "generatedSql": None,
                    "userMessageId": user_message_id,
                    "aiMessageId": ai_message_id,
                    "userCreatedAt": user_created_at,
                    "aiCreatedAt": ai_created_at
                }
            }
        else:
            prepend_greeting = False

        # Check for thanks messages and empty/meaningless messages
        if is_thanks_message(request.naturalLanguageQuery) or is_empty_or_meaningless(request.naturalLanguageQuery):
            # Initialize LLM for generating follow-up response
            google_api_key = os.environ.get("GOOGLE_API_KEY")
            if google_api_key:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash",
                        google_api_key=google_api_key
                    )
                    project_context = f"Project: {project_name}\nProject Info: {project_info}"
                    follow_up_response = generate_follow_up_response(project_context, request.naturalLanguageQuery, llm)
                except Exception as e:
                    print(f"[WARNING] Failed to generate follow-up response: {str(e)}")
                    follow_up_response = "Is there anything else I can help you with regarding your data analysis?"
            else:
                follow_up_response = "Is there anything else I can help you with regarding your data analysis?"
            # Store both user and AI message
            user_message_id, ai_message_id, user_created_at, ai_created_at = store_user_and_ai_message(request.naturalLanguageQuery, follow_up_response, "GENERAL")
            return {
                "success": True,
                "data": {
                    "conversationId": conversation_id,
                    "analysis": follow_up_response,
                    "queryType": "GENERAL",
                    "generatedSql": None,
                    "userMessageId": user_message_id,
                    "aiMessageId": ai_message_id,
                    "userCreatedAt": user_created_at,
                    "aiCreatedAt": ai_created_at
                }
            }

        if is_greeting(request.naturalLanguageQuery):
            # Store both user and AI message
            user_message_id, ai_message_id, user_created_at, ai_created_at = store_user_and_ai_message(request.naturalLanguageQuery, general_response, "GENERAL")
            return {
                "success": True,
                "data": {
                    "conversationId": conversation_id,
                    "analysis": general_response,
                    "queryType": "GENERAL",
                    "generatedSql": None,
                    "userMessageId": user_message_id,
                    "aiMessageId": ai_message_id,
                    "userCreatedAt": user_created_at,
                    "aiCreatedAt": ai_created_at
                }
            }

        # --- MEMORY-ENABLED GEMINI CALL FOR DATA QUESTIONS ---
        # Only for data analysis questions (not greetings/thanks/empty)
        # Fetch last 10 messages for this conversation
        chat_history = get_last_10_messages(conversation_id, limit=10)
        # Build LLM history with a system message for project context
        project_context = (
            f"You are a helpful assistant for project '{project_name}'. "
            f"Project info: {project_info}. "
            "Always answer with specific names, numbers, and details. "
            "IMPORTANT: When users ask follow-up questions like 'What about X?' or 'How about Y?', "
            "you MUST use the context from the previous conversation to understand what they're asking about. "
            "For example, if they previously asked about 'August 2024 sales' and now ask 'What about September 2024?', "
            "they want to know about September 2024 sales. Always maintain context and provide specific answers. "
            "Never ask users to rephrase unless absolutely necessary."
        )
        llm_history = [("system", project_context)]
        # Add previous messages (should be up to 10, oldest to newest)
        prev_msgs = [(msg["role"], msg["content"]) for msg in chat_history]
        llm_history.extend(prev_msgs)

        # Add the new user message, with follow-up rephrasing if needed
        last_bot_answer = chat_history[-1]["content"] if chat_history and chat_history[-1]["role"] == "ai" else ""
        user_message = request.naturalLanguageQuery.strip().lower()
        
        # Enhanced follow-up detection patterns
        vague_followups = ["which candidate", "who", "who was it", "who is it", "which leader", "who won", "who is the leader"]
        contextual_followups = ["what about", "how about", "and", "also", "then", "next", "after that", "similarly", "likewise"]
        
        # Check if this is a contextual follow-up question
        is_contextual_followup = any(pattern in user_message for pattern in contextual_followups)
        is_vague_followup = user_message in vague_followups
        
        if (is_vague_followup or is_contextual_followup) and last_bot_answer:
            # Explicit context chaining: combine last user question and last bot answer
            last_user_question = chat_history[-2]["content"] if len(chat_history) >= 2 and chat_history[-2]["role"] == "human" else ""
            explicit_prompt = (
                f"Context from previous conversation:\n"
                f"User asked: '{last_user_question}'\n"
                f"You answered: '{last_bot_answer}'\n\n"
                f"Now the user asks: '{request.naturalLanguageQuery}'\n\n"
                f"Please provide a specific answer using the context from the previous question and answer. "
                f"If the user is asking about a similar topic (like a different time period, category, or entity), "
                f"apply the same analysis approach to their new question."
            )
            llm_history.append(("human", explicit_prompt))
        else:
            llm_history.append(("human", request.naturalLanguageQuery))

        # Debug: print the full context sent to Gemini
        print(f"[DEBUG] LLM history sent to Gemini: {len(llm_history)-1} previous messages + system + new user message")
        for idx, (role, content) in enumerate(llm_history):
            print(f"  [{idx}] {role}: {content}")
        if len(prev_msgs) < 10:
            print(f"[WARNING] Only {len(prev_msgs)} previous messages found for this conversation. If you expect more, check message storage.")

        # Call Gemini LLM with memory context
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
            print("[ERROR] GOOGLE_API_KEY environment variable is not set")
            return {
                "success": True,
                "data": {
                    "conversationId": conversation_id,
                    "analysis": (
                        "## AI Analysis Service Temporarily Unavailable\n\n"
                        "I'm currently unable to perform AI-powered data analysis because the Google AI API key is not configured.\n\n"
                        "**What you can do:**\n"
                        "• Contact your system administrator to configure the Google AI API key\n"
                        "• Try asking simpler questions about your data\n"
                        "• Use the database connection to run manual queries\n\n"
                        "**For immediate assistance:**\n"
                        "• Check your database connection settings\n"
                        "• Verify that your project's database credentials are correct\n"
                        "• Ensure your database contains the data you're looking for"
                    ),
                    "queryType": "ERROR",
                    "generatedSql": None
                }
            }
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_api_key
        )
        response = llm.invoke(llm_history)
        analysis = response.content

        # --- Post-processing: Ensure candidate name is present if party and votes are mentioned ---
        if (
            re.search(r"belonged to the [A-Za-z ]+ party", analysis)
            and re.search(r"\d{1,3}(,\d{3})* votes", analysis)
            and not re.search(r"candidate (named|was|is|:) [A-Za-z ]+", analysis, re.IGNORECASE)
        ):
            # Try to extract year and party from the answer
            year_match = re.search(r"(\d{4})", analysis)
            party_match = re.search(r"belonged to the ([A-Za-z ]+) party", analysis)
            year = year_match.group(1) if year_match else ""
            party = party_match.group(1) if party_match else ""
            # Query the database for the candidate name
            if year and party:
                conn = get_db_connection()
                cursor = conn.cursor(dictionary=True)
                # You may need to adjust this query to match your schema
                cursor.execute(
                    """
                    SELECT candidate_name FROM election_results
                    WHERE year = %s AND party = %s
                    ORDER BY votes DESC LIMIT 1
                    """, (year, party)
                )
                row = cursor.fetchone()
                cursor.close()
                conn.close()
                if row and row.get("candidate_name"):
                    candidate_name = row["candidate_name"]
                    analysis += f" The candidate was {candidate_name}."

        # ... rest of your analysis logic ...
        # (move DB connection and project fetch above, so remove duplicate code below)
        db_host = decrypt(project["db_host"])
        db_user = decrypt(project["db_user"])
        db_password = decrypt(project["db_password"])
        db_name = decrypt(project["db_name"])
        db_port = "3306"  # Default port since it's not stored in your table
        print(f"[DEBUG] Database credentials retrieved successfully")
        # Check if Google API key is available
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
            print("[ERROR] GOOGLE_API_KEY environment variable is not set")
            return {
                "success": True,
                "data": {
                    "conversationId": conversation_id,
                    "analysis": (
                        "## AI Analysis Service Temporarily Unavailable\n\n"
                        "I'm currently unable to perform AI-powered data analysis because the Google AI API key is not configured.\n\n"
                        "**What you can do:**\n"
                        "• Contact your system administrator to configure the Google AI API key\n"
                        "• Try asking simpler questions about your data\n"
                        "• Use the database connection to run manual queries\n\n"
                        "**For immediate assistance:**\n"
                        "• Check your database connection settings\n"
                        "• Verify that your project's database credentials are correct\n"
                        "• Ensure your database contains the data you're looking for"
                    ),
                    "queryType": "ERROR",
                    "generatedSql": None
                }
            }
        print(f"[DEBUG] Google API key found and loaded successfully")
        # Initialize database analyzer with PostgreSQL settings
        try:
            analyzer = DatabaseAnalyzer(db_host, db_user, db_password, db_name, db_port)
            print("[DEBUG] DatabaseAnalyzer initialized successfully")
        except Exception as e:
            print(f"[ERROR] Failed to initialize DatabaseAnalyzer: {str(e)}")
            return {
                "success": True,
                "data": {
                    "conversationId": conversation_id,
                    "analysis": f"Failed to connect to the target database: {str(e)}",
                    "queryType": "ERROR",
                    "generatedSql": None
                }
            }
        # Execute analysis
        try:
            project_context = f"Project Goal: {project['project_info']}\nTechnical Schema Details: {project['db_info']}"
            results, analysis = analyzer.execute_query(request.naturalLanguageQuery, project_context)
            print(f"[DEBUG] Analysis completed successfully")
        except Exception as e:
            print(f"[ERROR] Analysis failed: {str(e)}")
            # Provide generic but helpful responses based on the error
            if "connection" in str(e).lower() or "connect" in str(e).lower():
                generic_response = (
                    "## Database Connection Issue\n\n"
                    "I'm unable to connect to your database for analysis.\n\n"
                    "**Possible causes:**\n"
                    "• Database server is not running\n"
                    "• Incorrect database credentials\n"
                    "• Network connectivity issues\n"
                    "• Database permissions\n\n"
                    "**What to check:**\n"
                    "• Verify your PostgreSQL server is running\n"
                    "• Check your project's database connection settings\n"
                    "• Ensure the database user has proper permissions\n"
                    "• Try connecting to the database manually to verify credentials"
                )
            elif "table" in str(e).lower() or "column" in str(e).lower():
                generic_response = (
                    "## Database Schema Issue\n\n"
                    "I found a problem with the database structure while analyzing your data.\n\n"
                    "**Possible causes:**\n"
                    "• Missing tables or columns\n"
                    "• Incorrect table names\n"
                    "• Schema changes not reflected\n\n"
                    "**What to check:**\n"
                    "• Verify that all required tables exist in your database\n"
                    "• Check if table or column names match your project configuration\n"
                    "• Ensure your database schema is up to date"
                )
            else:
                generic_response = (
                    "## Analysis Error\n\n"
                    "I encountered an unexpected error while analyzing your data.\n\n"
                    "**Error details:** " + str(e) + "\n\n"
                    "**What you can try:**\n"
                    "• Rephrase your question in simpler terms\n"
                    "• Check if your database contains the relevant data\n"
                    "• Try asking a different question\n"
                    "• Contact your system administrator if the problem persists"
                )
            analysis = generic_response
        # Fallback if analysis is empty
        if not analysis or not analysis.strip():
            analysis = "I'm sorry, I couldn't generate an answer for your question. Please try rephrasing or provide more details."
        print(f"[DEBUG] Final analysis to return: {analysis}")
        
        # Prepend greeting if this is the first message and not a general prompt
        if prepend_greeting:
            analysis = greeting + ' ' + analysis
        
        # Store messages in database
        try:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor(dictionary=True)
                
                # Create conversation if it doesn't exist
                cursor.execute("SELECT id FROM conversations WHERE id = %s", (conversation_id,))
                existing_conversation = cursor.fetchone()
                
                if not existing_conversation:
                    # Create new conversation
                    # Use the first user message as the title, truncated to 60 chars
                    first_message = request.naturalLanguageQuery.strip() if hasattr(request, 'naturalLanguageQuery') else ''
                    title = (first_message[:60] + '...') if len(first_message) > 60 else first_message or f"Conversation {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
                    cursor.execute(
                        "INSERT INTO conversations (id, user_id, project_id, title) VALUES (%s, %s, %s, %s)",
                        (conversation_id, current_user["user_id"], request.projectId, title)
                    )
                
                # Ensure IST timezone before inserting
                _ensure_ist_timezone(conn)
                
                # Store user message with explicit IST timestamp using MySQL's NOW() (respects session timezone)
                user_message_id = str(uuid.uuid4())
                cursor.execute(
                    "INSERT INTO messages (id, conversation_id, user_id, role, content, created_at) VALUES (%s, %s, %s, %s, %s, NOW())",
                    (user_message_id, conversation_id, current_user["user_id"], 'human', request.naturalLanguageQuery)
                )
                
                # Store AI response with explicit IST timestamp using MySQL's NOW() (respects session timezone)
                ai_message_id = str(uuid.uuid4())
                cursor.execute(
                    "INSERT INTO messages (id, conversation_id, user_id, role, content, query_type, created_at) VALUES (%s, %s, %s, %s, %s, %s, NOW())",
                    (ai_message_id, conversation_id, current_user["user_id"], 'ai', analysis, "DATABASE")
                )
                
                # Update conversation timestamp with IST
                cursor.execute(
                    "UPDATE conversations SET updated_at = NOW() WHERE id = %s",
                    (conversation_id,)
                )
                
                conn.commit()
        except Exception as e:
            print(f"[WARNING] Failed to store messages in database: {str(e)}")
            # Continue without failing the request
        
        # Fetch created_at timestamps for user and AI messages BEFORE closing cursor/conn
        # Ensure IST timezone and use CONVERT_TZ to get IST timestamp
        from datetime import timezone, timedelta
        ist_tz = timezone(timedelta(hours=5, minutes=30))
        _ensure_ist_timezone(conn)
        cursor.execute("SELECT UNIX_TIMESTAMP(CONVERT_TZ(created_at, @@session.time_zone, '+05:30')) as ts FROM messages WHERE id = %s", (user_message_id,))
        user_created_at_row = cursor.fetchone()
        if user_created_at_row and user_created_at_row.get("ts"):
            dt = datetime.fromtimestamp(user_created_at_row["ts"], tz=ist_tz)
            user_created_at = dt.isoformat()
        else:
            print(f"[WARNING] Could not fetch created_at for user message {user_message_id}, using now().")
            user_created_at = datetime.now(ist_tz).isoformat()

        cursor.execute("SELECT UNIX_TIMESTAMP(CONVERT_TZ(created_at, @@session.time_zone, '+05:30')) as ts FROM messages WHERE id = %s", (ai_message_id,))
        ai_created_at_row = cursor.fetchone()
        if ai_created_at_row and ai_created_at_row.get("ts"):
            dt = datetime.fromtimestamp(ai_created_at_row["ts"], tz=ist_tz)
            ai_created_at = dt.isoformat()
        else:
            print(f"[WARNING] Could not fetch created_at for AI message {ai_message_id}, using now().")
            ai_created_at = datetime.now(ist_tz).isoformat()

        cursor.close()
        conn.close()

        return {
            "success": True,
            "data": {
                "conversationId": conversation_id,
                "analysis": analysis,
                "queryType": "DATABASE",
                "generatedSql": None,
                "userMessageId": user_message_id,
                "aiMessageId": ai_message_id,
                "userCreatedAt": user_created_at,
                "aiCreatedAt": ai_created_at
            }
        }
    except Exception as e:
        print(f"[ERROR] Analyze data error: {str(e)}")
        import traceback
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/generate-visualization")
async def generate_visualization(request: AnalysisRequest, current_user: dict = Depends(get_current_user)):
    """Generate data visualization based on user query and store it in database"""
    try:
        print(f"[DEBUG] Visualization request received: {request.naturalLanguageQuery}")
        print(f"[DEBUG] Project ID: {request.projectId}")
        
        # Fetch project details
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM projects WHERE id = %s AND user_id = %s", (request.projectId, current_user["user_id"]))
        project = cursor.fetchone()
        
        if not project:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Get database connection details
        db_host = decrypt(project.get('db_host', ''))
        db_user = decrypt(project.get('db_user', ''))
        db_password = decrypt(project.get('db_password', ''))
        db_name = decrypt(project.get('db_name', ''))
        db_port = project.get('db_port', '3306')
        
        # Initialize database analyzer
        try:
            analyzer = DatabaseAnalyzer(db_host, db_user, db_password, db_name, db_port)
            print("[DEBUG] DatabaseAnalyzer initialized for visualization")
        except Exception as e:
            print(f"[ERROR] Failed to initialize DatabaseAnalyzer: {str(e)}")
            cursor.close()
            conn.close()
            raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")
        
        # Generate visualization
        try:
            project_context = f"Project Goal: {project['project_info']}\nTechnical Schema Details: {project['db_info']}"
            chart_data, analysis = analyzer.generate_visualization(request.naturalLanguageQuery, project_context)
            print(f"[DEBUG] Visualization generated successfully")
        except Exception as e:
            print(f"[ERROR] Visualization generation failed: {str(e)}")
            cursor.close()
            conn.close()
            raise HTTPException(status_code=500, detail=f"Visualization generation failed: {str(e)}")
        
        # Store both user and AI messages with visualization
        user_message_id = None
        ai_message_id = None
        visualization_id = None
        
        # Generate conversation_id if not provided
        if not request.conversationId:
            import uuid
            request.conversationId = str(uuid.uuid4())
            print(f"[DEBUG] Generated new conversation_id: {request.conversationId}")
        
        try:
            import uuid
            user_message_id = str(uuid.uuid4())
            ai_message_id = str(uuid.uuid4())
            visualization_id = str(uuid.uuid4())
            
            # Check if conversation exists, create if not
            cursor.execute("SELECT id FROM conversations WHERE id = %s", (request.conversationId,))
            if not cursor.fetchone():
                # Create new conversation
                title = f"Visualization: {request.naturalLanguageQuery[:50]}..."
                cursor.execute(
                    "INSERT INTO conversations (id, user_id, project_id, title) VALUES (%s, %s, %s, %s)",
                    (request.conversationId, current_user["user_id"], request.projectId, title)
                )
                print(f"[DEBUG] Created new conversation: {request.conversationId}")
            
            # First, ensure the visualizations table exists
            create_table_query = """
            CREATE TABLE IF NOT EXISTS visualizations (
                id VARCHAR(36) NOT NULL PRIMARY KEY,
                conversation_id VARCHAR(36) NOT NULL,
                message_id VARCHAR(36) NOT NULL,
                user_id VARCHAR(36) NOT NULL,
                project_id VARCHAR(36) NOT NULL,
                title VARCHAR(255) NOT NULL,
                chart_type VARCHAR(50) NOT NULL,
                chart_data LONGTEXT NOT NULL,
                query_used TEXT NOT NULL,
                data_summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                is_archived BOOLEAN DEFAULT FALSE,
                is_favorite BOOLEAN DEFAULT FALSE,
                INDEX idx_conversation (conversation_id),
                INDEX idx_user (user_id),
                INDEX idx_project (project_id),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            cursor.execute(create_table_query)
            print("[DEBUG] Visualizations table created/verified")
            
            # Ensure IST timezone before inserting
            _ensure_ist_timezone(conn)
            
            # Store the user message first with explicit IST timestamp using MySQL's NOW() (respects session timezone)
            cursor.execute(
                "INSERT INTO messages (id, conversation_id, user_id, role, content, query_type, created_at) VALUES (%s, %s, %s, %s, %s, %s, NOW())",
                (user_message_id, request.conversationId, current_user["user_id"], 'human', request.naturalLanguageQuery, "VISUALIZATION_REQUEST")
            )
            print(f"[DEBUG] User message stored with ID: {user_message_id}")
            
            # Store the AI message with visualization with explicit IST timestamp using MySQL's NOW() (respects session timezone)
            cursor.execute(
                "INSERT INTO messages (id, conversation_id, user_id, role, content, query_type, created_at) VALUES (%s, %s, %s, %s, %s, %s, NOW())",
                (ai_message_id, request.conversationId, current_user["user_id"], 'ai', analysis, "VISUALIZATION")
            )
            print(f"[DEBUG] AI message with visualization stored with ID: {ai_message_id}")
            
            # Insert visualization record linked to the AI message
            if chart_data and chart_data.get('data'):
                try:
                    insert_query = """
                    INSERT INTO visualizations 
                    (id, conversation_id, message_id, user_id, project_id, title, chart_type, 
                     chart_data, query_used, data_summary, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    """
                    
                    cursor.execute(insert_query, (
                        visualization_id,
                        request.conversationId,
                        ai_message_id,  # Link to the AI message
                        current_user["user_id"],
                        request.projectId,
                        chart_data.get('title', 'Data Visualization'),
                        chart_data.get('type', 'chart'),
                        chart_data.get('data', ''),
                        request.naturalLanguageQuery,
                        analysis[:500] if analysis else ''  # Truncate analysis for storage
                    ))
                    
                    print(f"[DEBUG] Visualization stored in database with ID: {visualization_id}")
                except Exception as viz_error:
                    print(f"[ERROR] Failed to store visualization: {str(viz_error)}")
                    # Continue without failing the entire request
            
            conn.commit()
            print(f"[DEBUG] All messages and visualization committed to database")
            
            # Verify the storage worked
            cursor.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = %s", (request.conversationId,))
            message_count = cursor.fetchone()[0]
            print(f"[DEBUG] Total messages in conversation: {message_count}")
            
            if visualization_id:
                cursor.execute("SELECT COUNT(*) FROM visualizations WHERE id = %s", (visualization_id,))
                viz_count = cursor.fetchone()[0]
                print(f"[DEBUG] Visualization records found: {viz_count}")
            
        except Exception as e:
            print(f"[ERROR] Failed to store messages/visualization: {str(e)}")
            print(f"[ERROR] Full error details: {str(e)}")
            # Don't fail the request if storage fails, just log the error
            try:
                conn.rollback()
            except:
                pass
        
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "data": {
                "visualization": chart_data,
                "analysis": analysis,
                "queryType": "VISUALIZATION",
                "visualizationId": visualization_id,
                "userMessageId": user_message_id,
                "aiMessageId": ai_message_id
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Visualization endpoint error: {str(e)}")
        import traceback
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

# Visualization Management Endpoints

@app.get("/api/visualizations")
async def get_visualizations(
    project_id: str = None,
    conversation_id: str = None,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """Get stored visualizations for a user, optionally filtered by project or conversation"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = conn.cursor(dictionary=True)
        
        # Build query with filters
        base_query = """
        SELECT v.id, v.title, v.chart_type, v.query_used, v.data_summary, 
               v.created_at, v.is_favorite, c.title as conversation_title,
               p.name as project_name
        FROM visualizations v
        JOIN conversations c ON v.conversation_id = c.id
        JOIN projects p ON v.project_id = p.id
        WHERE v.user_id = %s AND v.is_archived = FALSE
        """
        
        params = [current_user["user_id"]]
        
        if project_id:
            base_query += " AND v.project_id = %s"
            params.append(project_id)
            
        if conversation_id:
            base_query += " AND v.conversation_id = %s"
            params.append(conversation_id)
        
        base_query += " ORDER BY v.created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        cursor.execute(base_query, params)
        visualizations = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "data": visualizations,
            "count": len(visualizations)
        }
        
    except Exception as e:
        print(f"[ERROR] Get visualizations error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve visualizations: {str(e)}")

@app.get("/api/visualizations/{visualization_id}")
async def get_visualization(visualization_id: str, current_user: dict = Depends(get_current_user)):
    """Get a specific visualization by ID"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT v.*, c.title as conversation_title, p.name as project_name
            FROM visualizations v
            JOIN conversations c ON v.conversation_id = c.id
            JOIN projects p ON v.project_id = p.id
            WHERE v.id = %s AND v.user_id = %s AND v.is_archived = FALSE
        """, (visualization_id, current_user["user_id"]))
        
        visualization = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not visualization:
            raise HTTPException(status_code=404, detail="Visualization not found")
        
        return {
            "success": True,
            "data": visualization
        }
        
    except Exception as e:
        print(f"[ERROR] Get visualization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve visualization: {str(e)}")

@app.put("/api/visualizations/{visualization_id}/favorite")
async def toggle_favorite_visualization(visualization_id: str, current_user: dict = Depends(get_current_user)):
    """Toggle favorite status of a visualization"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = conn.cursor(dictionary=True)
        
        # Check if visualization exists and belongs to user
        cursor.execute("""
            SELECT is_favorite FROM visualizations 
            WHERE id = %s AND user_id = %s AND is_archived = FALSE
        """, (visualization_id, current_user["user_id"]))
        
        result = cursor.fetchone()
        if not result:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Visualization not found")
        
        # Toggle favorite status
        new_favorite_status = not result['is_favorite']
        cursor.execute("""
            UPDATE visualizations 
            SET is_favorite = %s, updated_at = NOW()
            WHERE id = %s AND user_id = %s
        """, (new_favorite_status, visualization_id, current_user["user_id"]))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "data": {
                "visualizationId": visualization_id,
                "isFavorite": new_favorite_status
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Toggle favorite error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update favorite status: {str(e)}")

@app.delete("/api/visualizations/{visualization_id}")
async def delete_visualization(visualization_id: str, current_user: dict = Depends(get_current_user)):
    """Soft delete a visualization (mark as archived)"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            UPDATE visualizations 
            SET is_archived = TRUE, updated_at = NOW()
            WHERE id = %s AND user_id = %s
        """, (visualization_id, current_user["user_id"]))
        
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Visualization not found")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "message": "Visualization deleted successfully"
        }
        
    except Exception as e:
        print(f"[ERROR] Delete visualization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete visualization: {str(e)}")

# Chat Storage and History Endpoints

@app.post("/api/conversations")
async def create_conversation(conversation_data: ConversationCreate, current_user: dict = Depends(get_current_user)):
    """Create a new conversation"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = conn.cursor(dictionary=True)
        
        # Verify project exists and belongs to user
        cursor.execute("SELECT id FROM projects WHERE id = %s AND user_id = %s", 
                      (conversation_data.projectId, current_user["user_id"]))
        project = cursor.fetchone()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        conversation_id = str(uuid.uuid4())
        # Use the first user message as the title if provided, truncated to 60 chars
        if hasattr(conversation_data, 'firstMessage') and conversation_data.firstMessage:
            first_message = conversation_data.firstMessage.strip()
            title = (first_message[:60] + '...') if len(first_message) > 60 else first_message
        else:
            from datetime import timezone, timedelta
            ist_tz = timezone(timedelta(hours=5, minutes=30))
            title = f"Conversation {datetime.now(ist_tz).strftime('%Y-%m-%d %H:%M')}"
        
        cursor.execute(
            "INSERT INTO conversations (id, user_id, project_id, title) VALUES (%s, %s, %s, %s)",
            (conversation_id, current_user["user_id"], conversation_data.projectId, title)
        )
        conn.commit()
        
        cursor.close()
        conn.close()
        
        from datetime import timezone, timedelta
        ist_tz = timezone(timedelta(hours=5, minutes=30))
        return {
            "success": True,
            "data": {
                "id": conversation_id,
                "title": title,
                "projectId": conversation_data.projectId,
                "createdAt": datetime.now(ist_tz).isoformat()
            }
        }
    except Exception as e:
        print(f"[ERROR] Create conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")

@app.get("/api/conversations")
async def get_conversations(project_id: Optional[str] = None, current_user: dict = Depends(get_current_user)):
    """Get all active conversations for a user, optionally filtered by project"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        cursor = conn.cursor(dictionary=True)
        query = "SELECT c.*, COUNT(m.id) as message_count FROM conversations c " \
                "LEFT JOIN messages m ON c.id = m.conversation_id " \
                "WHERE c.user_id = %s AND c.status = 'active' "
        params = [current_user["user_id"]]
        if project_id:
            query += " AND c.project_id = %s"
            params.append(project_id)
        query += " GROUP BY c.id"
        cursor.execute(query, tuple(params))
        conversations = cursor.fetchall()
        cursor.close()
        conn.close()
        return {"data": conversations}
    except Exception as e:
        print(f"[ERROR] Get conversations error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversations: {str(e)}")

@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, current_user: dict = Depends(get_current_user)):
    """Get a single active conversation by ID, including its messages"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT c.*, p.name as project_name FROM conversations c "
                       "LEFT JOIN projects p ON c.project_id = p.id "
                       "WHERE c.id = %s AND c.user_id = %s AND c.status = 'active'",
                       (conversation_id, current_user["user_id"]))
        conversation = cursor.fetchone()
        if not conversation:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Conversation not found")
        # Ensure IST timezone for this query (double-check)
        _ensure_ist_timezone(conn)
        
        # Fetch messages for this conversation with visualizations
        # Use UNIX_TIMESTAMP with CONVERT_TZ to ensure we get IST regardless of how MySQL stored it
        cursor.execute("""
            SELECT m.id as id, m.conversation_id, m.user_id, m.role, m.content, m.query_type, m.generated_sql, 
                   UNIX_TIMESTAMP(CONVERT_TZ(m.created_at, @@session.time_zone, '+05:30')) as created_at_timestamp,
                   v.id as visualization_id, v.title as visualization_title, v.chart_type, v.chart_data, v.query_used as visualization_query
            FROM messages m
            LEFT JOIN visualizations v ON m.id = v.message_id AND v.is_archived = FALSE
            WHERE m.conversation_id = %s AND m.is_archived = FALSE
            ORDER BY m.created_at ASC
        """, (conversation_id,))
        messages = cursor.fetchall()
        
        # Convert UNIX_TIMESTAMP to IST datetime with timezone info
        from datetime import timezone, timedelta
        ist_tz = timezone(timedelta(hours=5, minutes=30))
        for message in messages:
            if message.get('created_at_timestamp'):
                # UNIX_TIMESTAMP returns seconds since epoch, convert to IST
                dt = datetime.fromtimestamp(message['created_at_timestamp'], tz=ist_tz)
                message['created_at'] = dt.isoformat()
            # Remove the temporary timestamp field
            message.pop('created_at_timestamp', None)
        
        # Process messages and add visualization data
        processed_messages = []
        for msg in messages:
            # Ensure every message has a 'createdAt' field in ISO format
            if 'created_at' in msg and msg['created_at']:
                if not isinstance(msg['created_at'], str):
                    msg['createdAt'] = msg['created_at'].isoformat()
                else:
                    msg['createdAt'] = msg['created_at']
            
            # Add visualization data if it exists
            if msg.get('visualization_id') and msg.get('chart_data'):
                msg['visualization'] = {
                    'id': msg['visualization_id'],
                    'title': msg['visualization_title'],
                    'type': msg['chart_type'],
                    'data': msg['chart_data'],
                    'query': msg['visualization_query']
                }
            
            # Remove visualization fields from the main message object
            msg.pop('visualization_id', None)
            msg.pop('visualization_title', None)
            msg.pop('chart_type', None)
            msg.pop('chart_data', None)
            msg.pop('visualization_query', None)
            
            processed_messages.append(msg)
        
        messages = processed_messages
        cursor.close()
        conn.close()
        return {"data": {"conversation": conversation, "messages": messages}}
    except Exception as e:
        print(f"[ERROR] Get conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation: {str(e)}")

@app.put("/api/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, conversation_data: ConversationUpdate, 
                            current_user: dict = Depends(get_current_user)):
    """Update conversation details"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = conn.cursor(dictionary=True)
        
        # Verify conversation belongs to user
        cursor.execute("SELECT id FROM conversations WHERE id = %s AND user_id = %s", 
                      (conversation_id, current_user["user_id"]))
        conversation = cursor.fetchone()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Build update query
        update_fields = []
        update_values = []
        
        if conversation_data.title is not None:
            update_fields.append("title = %s")
            update_values.append(conversation_data.title)
        
        if conversation_data.isArchived is not None:
            update_fields.append("is_archived = %s")
            update_values.append(conversation_data.isArchived)
        
        if update_fields:
            update_values.append(conversation_id)
            cursor.execute(
                f"UPDATE conversations SET {', '.join(update_fields)} WHERE id = %s",
                update_values
            )
            conn.commit()
        
        cursor.close()
        conn.close()
        
        return {"success": True, "message": "Conversation updated successfully"}
    except Exception as e:
        print(f"[ERROR] Update conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update conversation: {str(e)}")

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, current_user: dict = Depends(get_current_user)):
    """Soft delete a conversation by setting status to 'deleted'"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = conn.cursor(dictionary=True)
        
        # Verify conversation belongs to user
        cursor.execute("SELECT id FROM conversations WHERE id = %s AND user_id = %s", 
                      (conversation_id, current_user["user_id"]))
        conversation = cursor.fetchone()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Soft delete: set status to 'deleted'
        cursor.execute("UPDATE conversations SET status = 'deleted' WHERE id = %s", (conversation_id,))
        # Soft delete all related messages
        cursor.execute("UPDATE messages SET is_archived = 1 WHERE conversation_id = %s", (conversation_id,))
        # Soft delete all related important messages
        cursor.execute("UPDATE important_messages SET is_archived = 1 WHERE conversation_id = %s", (conversation_id,))
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return {"success": True, "message": "Conversation deleted (soft) successfully"}
    except Exception as e:
        print(f"[ERROR] Delete conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {str(e)}")

@app.post("/api/messages")
async def create_message(message_data: MessageCreate, current_user: dict = Depends(get_current_user)):
    """Create a new message in a conversation"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = conn.cursor(dictionary=True)
        
        # Verify conversation belongs to user
        cursor.execute(
            "SELECT c.id FROM conversations c WHERE c.id = %s AND c.user_id = %s",
            (message_data.conversationId, current_user["user_id"])
        )
        conversation = cursor.fetchone()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Ensure IST timezone before inserting
        _ensure_ist_timezone(conn)
        
        message_id = str(uuid.uuid4())
        
        cursor.execute(
            "INSERT INTO messages (id, conversation_id, user_id, role, content, query_type, generated_sql, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())",
            (message_id, message_data.conversationId, current_user["user_id"], message_data.role, message_data.content,
             message_data.queryType, message_data.generatedSql)
        )
        
        # Update conversation's updated_at timestamp with IST
        cursor.execute(
            "UPDATE conversations SET updated_at = NOW() WHERE id = %s",
            (message_data.conversationId,)
        )
        
        conn.commit()
        
        cursor.close()
        conn.close()
        
        # After message creation, broadcast chat update
        await broadcast_project_update(message_data.conversationId, {"type": "chat_update"})
        from datetime import timezone, timedelta
        ist_tz = timezone(timedelta(hours=5, minutes=30))
        return {
            "success": True,
            "data": {
                "id": message_id,
                "conversationId": message_data.conversationId,
                "role": message_data.role,
                "content": message_data.content,
                "createdAt": datetime.now(ist_tz).isoformat()
            }
        }
    except Exception as e:
        print(f"[ERROR] Create message error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create message: {str(e)}")

# This endpoint is deprecated - use POST/DELETE /api/messages/{message_id}/important instead
# @app.put("/api/messages/{message_id}")
# async def update_message(message_id: str, message_data: MessageUpdate, current_user: dict = Depends(get_current_user)):
#     """Update message importance flag - DEPRECATED"""
#     raise HTTPException(status_code=410, detail="This endpoint is deprecated. Use POST/DELETE /api/messages/{message_id}/important instead")

@app.delete("/api/messages/{message_id}")
async def delete_message(message_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a specific message"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = conn.cursor(dictionary=True)
        
        # Verify message belongs to user's conversation
        cursor.execute(
            "SELECT m.id FROM messages m "
            "JOIN conversations c ON m.conversation_id = c.id "
            "WHERE m.id = %s AND c.user_id = %s",
            (message_id, current_user["user_id"])
        )
        message = cursor.fetchone()
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        # Delete the message
        cursor.execute("DELETE FROM messages WHERE id = %s", (message_id,))
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return {"success": True, "message": "Message deleted successfully"}
    except Exception as e:
        print(f"[ERROR] Delete message error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete message: {str(e)}")

@app.post("/api/messages/{message_id}/important")
async def mark_message_important(message_id: str, current_user: dict = Depends(get_current_user)):
    """Mark a message as important for the current user."""
    try:
        print(f"[DEBUG] Marking message {message_id} as important for user {current_user['user_id']}")
        
        conn = get_db_connection()
        if not conn:
            print("[ERROR] Database connection failed")
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cursor = conn.cursor(dictionary=True)
        
        # Check if message exists and get conversation_id
        print(f"[DEBUG] Checking if message {message_id} exists")
        cursor.execute("SELECT id, conversation_id FROM messages WHERE id = %s", (message_id,))
        message = cursor.fetchone()
        
        if not message:
            print(f"[ERROR] Message {message_id} not found in database")
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Message not found")
        
        print(f"[DEBUG] Message found: {message}")
        print(f"[DEBUG] Conversation ID: {message['conversation_id']}")

        # Check if already marked as important (only if is_deleted=0)
        cursor.execute(
            "SELECT id, is_archived FROM important_messages WHERE message_id = %s AND user_id = %s",
            (message_id, current_user["user_id"])
        )
        existing = cursor.fetchone()
        if existing:
            if existing.get('is_archived', 0) == 0:
                print(f"[DEBUG] Message already marked as important (is_archived=0)")
                cursor.close()
                conn.close()
                return {"success": True, "message": "Message already marked as important."}
            else:
                # If soft-deleted, update is_archived=0
                cursor.execute(
                    "UPDATE important_messages SET is_archived = 0 WHERE id = %s",
                    (existing['id'],)
                )
                conn.commit()
                print(f"[DEBUG] Restored soft-deleted important message: {existing['id']}")
                cursor.close()
                conn.close()
                return {"success": True, "message": "Message re-marked as important."}

        # Insert into important_messages with conversation_id
        import uuid
        important_id = str(uuid.uuid4())
        print(f"[DEBUG] Inserting into important_messages: id={important_id}, conversation_id={message['conversation_id']}, message_id={message_id}, user_id={current_user['user_id']}")
        
        try:
            cursor.execute(
                "INSERT INTO important_messages (id, conversation_id, message_id, user_id) VALUES (%s, %s, %s, %s)",
                (important_id, message['conversation_id'], message_id, current_user["user_id"])
            )
            conn.commit()
            print(f"[DEBUG] Successfully marked message as important")
        except Exception as e:
            print(f"[ERROR] Database error during insert: {str(e)}")
            cursor.close()
            conn.close()
            raise HTTPException(status_code=500, detail=f"Failed to mark as important: {str(e)}")
        
        cursor.close()
        conn.close()
        return {"success": True, "message": "Message marked as important."}
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"[ERROR] Unexpected error in mark_message_important: {str(e)}")
        import traceback
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to mark as important: {str(e)}")

@app.delete("/api/messages/{message_id}/important")
async def unmark_message_important(message_id: str, current_user: dict = Depends(get_current_user)):
    """Unmark a message as important for the current user (soft delete)."""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "UPDATE important_messages SET is_archived = 1 WHERE message_id = %s AND user_id = %s AND is_archived = 0",
            (message_id, current_user["user_id"])
        )
        conn.commit()
        cursor.close()
        conn.close()
        return {"success": True, "message": "Message unmarked as important (soft deleted)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unmark as important: {str(e)}")

@app.patch("/api/messages/{message_id}/important-title")
async def update_important_message_title(message_id: str, title: str = Body(..., embed=True), current_user: dict = Depends(get_current_user)):
    """Update the title of an important message for the current user (only if not archived)."""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "UPDATE important_messages SET title = %s WHERE message_id = %s AND user_id = %s AND is_archived = 0",
            (title, message_id, current_user["user_id"])
        )
        conn.commit()
        cursor.close()
        conn.close()
        return {"success": True, "title": title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update important message title: {str(e)}")

@app.get("/api/important-messages")
async def get_important_messages_api(
    project_id: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """Get all important messages for the current user, optionally filtered by project. Only return not-deleted."""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        cursor = conn.cursor(dictionary=True)
        if project_id:
            cursor.execute(
                """
                SELECT m.*, im.title, c.title as conversation_title, p.name as project_name,
                       v.id as visualization_id, v.title as visualization_title, v.chart_type, v.chart_data, v.query_used as visualization_query
                FROM important_messages im
                JOIN messages m ON im.message_id = m.id
                JOIN conversations c ON m.conversation_id = c.id
                JOIN projects p ON c.project_id = p.id
                LEFT JOIN visualizations v ON m.id = v.message_id AND v.is_archived = FALSE
                WHERE im.user_id = %s AND c.project_id = %s AND im.is_archived = 0
                ORDER BY m.created_at ASC
                """,
                (current_user["user_id"], project_id)
            )
        else:
            cursor.execute(
                """
                SELECT m.*, im.title, c.title as conversation_title, p.name as project_name,
                       v.id as visualization_id, v.title as visualization_title, v.chart_type, v.chart_data, v.query_used as visualization_query
                FROM important_messages im
                JOIN messages m ON im.message_id = m.id
                JOIN conversations c ON m.conversation_id = c.id
                JOIN projects p ON c.project_id = p.id
                LEFT JOIN visualizations v ON m.id = v.message_id AND v.is_archived = FALSE
                WHERE im.user_id = %s AND im.is_archived = 0
                ORDER BY m.created_at ASC
                """,
                (current_user["user_id"],)
            )
        messages = cursor.fetchall()
        
        # Process messages and add visualization data
        processed_messages = []
        for msg in messages:
            # Add visualization data if it exists
            if msg.get('visualization_id') and msg.get('chart_data'):
                msg['visualization'] = {
                    'id': msg['visualization_id'],
                    'title': msg['visualization_title'],
                    'type': msg['chart_type'],
                    'data': msg['chart_data'],
                    'query': msg['visualization_query']
                }
            
            # Remove visualization fields from the main message object
            msg.pop('visualization_id', None)
            msg.pop('visualization_title', None)
            msg.pop('chart_type', None)
            msg.pop('chart_data', None)
            msg.pop('visualization_query', None)
            
            processed_messages.append(msg)
        
        messages = processed_messages
        print(f"[DEBUG] get_important_messages_api: user_id={current_user['user_id']} project_id={project_id} -> {len(messages)} messages")
        cursor.close()
        conn.close()
        return {"success": True, "data": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get important messages: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 