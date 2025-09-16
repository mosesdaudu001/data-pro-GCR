from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from werkzeug.security import check_password_hash, generate_password_hash
import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import google.generativeai as genai
from google.generativeai import types
from typing import Dict, Any
import os
from dotenv import load_dotenv
import plotly.graph_objs as go
import plotly.utils

load_dotenv()

# # Configure Gemini AI
# genai.configure(api_key=os.getenv('GEMINI_API_KEY', 'your-api-key-here'))
# client = genai.Client()

try:
    from google import genai
    from google.genai import types
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "tranquil-rite-467211-c6-604a3e9c8a6b.json"

    # Initialize Vertex AI client
    client = genai.Client(
        vertexai=True,
        project="tranquil-rite-467211-c6",
        location="us-central1",
    )
    
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    raise

app = Flask(__name__)
app.config['SECRET_KEY'] = 'demo-secret-key-for-grc-platform'

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username, email, role):
        self.id = id
        self.username = username
        self.email = email
        self.role = role

# Demo users (in production this would be database)
demo_users = {
    'admin': {'password': generate_password_hash('admin123'), 'email': 'admin@cbl.lr', 'role': 'Administrator'},
    'risk_manager': {'password': generate_password_hash('risk123'), 'email': 'risk@cbl.lr', 'role': 'Risk Manager'},
    'compliance': {'password': generate_password_hash('comp123'), 'email': 'compliance@cbl.lr', 'role': 'Compliance Officer'},
    'auditor': {'password': generate_password_hash('audit123'), 'email': 'auditor@cbl.lr', 'role': 'Internal Auditor'}
}

@login_manager.user_loader
def load_user(user_id):
    for username, user_data in demo_users.items():
        if username == user_id:
            return User(username, username, user_data['email'], user_data['role'])
    return None

# Database initialization (SQLite for demo)
def init_db():
    conn = sqlite3.connect('grc_demo.db')
    cursor = conn.cursor()
    
    # Risk assessments table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS risk_assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            risk_type TEXT,
            risk_category TEXT,
            risk_description TEXT,
            likelihood INTEGER,
            impact INTEGER,
            risk_score INTEGER,
            mitigation_status TEXT,
            owner TEXT,
            created_date TEXT,
            updated_date TEXT
        )
    ''')
    
    # Compliance tracking table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS compliance_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            regulation TEXT,
            requirement TEXT,
            status TEXT,
            due_date TEXT,
            owner TEXT,
            evidence TEXT,
            created_date TEXT,
            updated_date TEXT
        )
    ''')
    
    # Audit trails table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_trails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT,
            action TEXT,
            module TEXT,
            details TEXT,
            timestamp TEXT
        )
    ''')
    
    # Insert demo data
    cursor.execute("DELETE FROM risk_assessments")
    demo_risks = [
        ('Credit Risk', 'Credit', 'High concentration in government securities', 4, 5, 20, 'In Progress', 'Risk Manager', datetime.now().isoformat(), datetime.now().isoformat()),
        ('Market Risk', 'Market', 'Currency exchange rate fluctuations', 3, 4, 12, 'Identified', 'Risk Manager', datetime.now().isoformat(), datetime.now().isoformat()),
        ('Operational Risk', 'Operational', 'System downtime during critical operations', 2, 5, 10, 'Mitigated', 'IT Manager', datetime.now().isoformat(), datetime.now().isoformat()),
        ('Liquidity Risk', 'Liquidity', 'Insufficient liquid assets during stress periods', 3, 4, 12, 'Monitored', 'Treasury', datetime.now().isoformat(), datetime.now().isoformat()),
    ]
    cursor.executemany('INSERT INTO risk_assessments (risk_type, risk_category, risk_description, likelihood, impact, risk_score, mitigation_status, owner, created_date, updated_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', demo_risks)
    
    cursor.execute("DELETE FROM compliance_items")
    demo_compliance = [
        ('Basel III', 'Capital Adequacy Ratio maintenance', 'Compliant', '2024-12-31', 'Compliance Officer', 'CAR Report Q3 2024', datetime.now().isoformat(), datetime.now().isoformat()),
        ('AML/CFT', 'Customer Due Diligence updates', 'In Progress', '2024-11-30', 'AML Officer', 'Pending 15% of accounts', datetime.now().isoformat(), datetime.now().isoformat()),
        ('IFRS 9', 'Expected Credit Loss provisioning', 'Compliant', '2024-10-31', 'Finance Manager', 'ECL Model Q3 2024', datetime.now().isoformat(), datetime.now().isoformat()),
        ('GDPR', 'Data protection impact assessments', 'Overdue', '2024-09-30', 'Data Protection Officer', 'Outstanding DPIA for 3 systems', datetime.now().isoformat(), datetime.now().isoformat()),
    ]
    cursor.executemany('INSERT INTO compliance_items (regulation, requirement, status, due_date, owner, evidence, created_date, updated_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)', demo_compliance)
    
    conn.commit()
    conn.close()

def log_audit_trail(user, action, module, details):
    """Log user actions for audit purposes"""
    conn = sqlite3.connect('grc_demo.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO audit_trails (user, action, module, details, timestamp) VALUES (?, ?, ?, ?, ?)',
                  (user, action, module, details, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def analyze_risk_with_ai(risk_data) -> Dict[str, Any]:
    """Analyze risk using Gemini AI"""
    prompt = f"""
    Analyze the following risk data for a central bank and provide structured insights:
    
    Risk Type: {risk_data['risk_type']}
    Risk Category: {risk_data['risk_category']}
    Description: {risk_data['risk_description']}
    Likelihood: {risk_data['likelihood']}/5
    Impact: {risk_data['impact']}/5
    Current Status: {risk_data['mitigation_status']}
    
    Provide analysis including risk assessment, mitigation recommendations, and monitoring strategies.
    """
    
    model = "gemini-2.0-flash-001"
    
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ]
        )
    ]
    
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "risk_assessment": {
                "type": "STRING",
                "description": "Overall assessment of the risk"
            },
            "severity_level": {
                "type": "STRING",
                "description": "Risk severity: Low, Medium, High, Critical"
            },
            "mitigation_recommendations": {
                "type": "STRING", 
                "description": "Recommended mitigation strategies"
            },
            "monitoring_strategy": {
                "type": "STRING",
                "description": "Suggested monitoring and review approach"
            },
            "regulatory_considerations": {
                "type": "STRING",
                "description": "Relevant regulatory requirements and considerations"
            }
        },
        "required": ["risk_assessment", "severity_level"]
    }
    
    generate_content_config = types.GenerateContentConfig(
        temperature=0.1,
        top_p=1.0,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        response_mime_type="application/json",
        response_schema=response_schema
    )
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        result = json.loads(response.text)
        return result
    except Exception as e:
        print(f"AI Analysis Error: {e}")
        return {
            "risk_assessment": "AI analysis temporarily unavailable",
            "severity_level": "Medium",
            "mitigation_recommendations": "Standard risk management procedures apply",
            "monitoring_strategy": "Regular review recommended",
            "regulatory_considerations": "Follow applicable regulations"
        }

# Routes
@app.route('/')
@login_required
def dashboard():
    log_audit_trail(current_user.username, 'View', 'Dashboard', 'Accessed main dashboard')
    
    # Get summary data
    conn = sqlite3.connect('grc_demo.db')
    cursor = conn.cursor()
    
    # Risk summary
    cursor.execute('SELECT COUNT(*) FROM risk_assessments')
    total_risks = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM risk_assessments WHERE risk_score >= 15')
    high_risks = cursor.fetchone()[0]
    
    # Compliance summary
    cursor.execute('SELECT COUNT(*) FROM compliance_items')
    total_compliance = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM compliance_items WHERE status = "Overdue"')
    overdue_compliance = cursor.fetchone()[0]
    
    # Recent activities
    cursor.execute('SELECT * FROM audit_trails ORDER BY timestamp DESC LIMIT 5')
    recent_activities = cursor.fetchall()
    
    conn.close()
    
    # Create risk chart
    risk_chart = create_risk_distribution_chart()
    
    return render_template('dashboard.html', 
                         total_risks=total_risks,
                         high_risks=high_risks,
                         total_compliance=total_compliance,
                         overdue_compliance=overdue_compliance,
                         recent_activities=recent_activities,
                         risk_chart=risk_chart)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in demo_users and check_password_hash(demo_users[username]['password'], password):
            user = User(username, username, demo_users[username]['email'], demo_users[username]['role'])
            login_user(user)
            log_audit_trail(username, 'Login', 'Authentication', 'User logged in successfully')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    log_audit_trail(current_user.username, 'Logout', 'Authentication', 'User logged out')
    logout_user()
    return redirect(url_for('login'))

@app.route('/risk-management')
@login_required
def risk_management():
    log_audit_trail(current_user.username, 'View', 'Risk Management', 'Accessed risk management module')
    
    conn = sqlite3.connect('grc_demo.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM risk_assessments ORDER BY risk_score DESC')
    risks = cursor.fetchall()
    conn.close()
    
    return render_template('risk_management.html', risks=risks)

@app.route('/risk-analysis/<int:risk_id>')
@login_required
def risk_analysis(risk_id):
    conn = sqlite3.connect('grc_demo.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM risk_assessments WHERE id = ?', (risk_id,))
    risk = cursor.fetchone()
    conn.close()
    
    if risk:
        risk_data = {
            'risk_type': risk[1],
            'risk_category': risk[2],
            'risk_description': risk[3],
            'likelihood': risk[4],
            'impact': risk[5],
            'mitigation_status': risk[7]
        }
        
        ai_analysis = analyze_risk_with_ai(risk_data)
        log_audit_trail(current_user.username, 'Analyze', 'Risk Management', f'AI analysis performed for risk ID {risk_id}')
        
        return render_template('risk_analysis.html', risk=risk, ai_analysis=ai_analysis)
    
    return redirect(url_for('risk_management'))

@app.route('/compliance')
@login_required
def compliance():
    log_audit_trail(current_user.username, 'View', 'Compliance', 'Accessed compliance management module')
    
    conn = sqlite3.connect('grc_demo.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM compliance_items ORDER BY due_date ASC')
    compliance_items = cursor.fetchall()
    conn.close()
    
    return render_template('compliance.html', compliance_items=compliance_items)

@app.route('/reporting')
@login_required
def reporting():
    log_audit_trail(current_user.username, 'View', 'Reporting', 'Accessed reporting module')
    
    # Generate sample reports data
    reports = [
        {'name': 'Basel III Capital Adequacy Report', 'type': 'Regulatory', 'status': 'Generated', 'date': '2024-10-15'},
        {'name': 'Risk Assessment Summary', 'type': 'Risk', 'status': 'Pending', 'date': '2024-10-20'},
        {'name': 'AML/CFT Compliance Report', 'type': 'Compliance', 'status': 'In Progress', 'date': '2024-10-25'},
        {'name': 'IFRS 9 ECL Report', 'type': 'Financial', 'status': 'Generated', 'date': '2024-10-10'}
    ]
    
    return render_template('reporting.html', reports=reports)

@app.route('/audit-trail')
@login_required
def audit_trail():
    log_audit_trail(current_user.username, 'View', 'Audit Trail', 'Accessed audit trail module')
    
    conn = sqlite3.connect('grc_demo.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM audit_trails ORDER BY timestamp DESC LIMIT 100')
    audit_logs = cursor.fetchall()
    conn.close()
    
    return render_template('audit_trail.html', audit_logs=audit_logs)

@app.route('/data-analytics')
@login_required
def data_analytics():
    log_audit_trail(current_user.username, 'View', 'Data Analytics', 'Accessed data analytics module')
    
    # Create sample analytics charts
    risk_trend_chart = create_risk_trend_chart()
    compliance_status_chart = create_compliance_status_chart()
    
    return render_template('data_analytics.html', 
                         risk_trend_chart=risk_trend_chart,
                         compliance_status_chart=compliance_status_chart)

def create_risk_distribution_chart():
    """Create risk distribution pie chart"""
    conn = sqlite3.connect('grc_demo.db')
    df = pd.read_sql_query('SELECT risk_category, COUNT(*) as count FROM risk_assessments GROUP BY risk_category', conn)
    conn.close()
    
    fig = go.Figure(data=[go.Pie(labels=df['risk_category'], values=df['count'])])
    fig.update_layout(title='Risk Distribution by Category', height=400)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_risk_trend_chart():
    """Create risk trend line chart"""
    # Simulate trend data
    dates = pd.date_range(start='2024-01-01', end='2024-10-01', freq='M')
    risk_scores = np.random.randint(8, 25, len(dates))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=risk_scores, mode='lines+markers', name='Average Risk Score'))
    fig.update_layout(title='Risk Score Trends', xaxis_title='Date', yaxis_title='Risk Score')
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_compliance_status_chart():
    """Create compliance status bar chart"""
    conn = sqlite3.connect('grc_demo.db')
    df = pd.read_sql_query('SELECT status, COUNT(*) as count FROM compliance_items GROUP BY status', conn)
    conn.close()
    
    fig = go.Figure(data=[go.Bar(x=df['status'], y=df['count'])])
    fig.update_layout(title='Compliance Status Overview', xaxis_title='Status', yaxis_title='Count')
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/api/risk-summary')
@login_required
def api_risk_summary():
    """API endpoint for risk summary data"""
    conn = sqlite3.connect('grc_demo.db')
    cursor = conn.cursor()
    cursor.execute('SELECT risk_category, AVG(risk_score) as avg_score FROM risk_assessments GROUP BY risk_category')
    data = cursor.fetchall()
    conn.close()
    
    return jsonify([{'category': row[0], 'average_score': round(row[1], 2)} for row in data])

if __name__ == '__main__':
    init_db()
    app.run(debug=True)