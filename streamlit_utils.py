"""
Utility functions for Streamlit RAG application
"""
import streamlit as st
from datetime import datetime
import pandas as pd
import os


def format_message_time(timestamp_str):
    """Format timestamp for display"""
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
        return timestamp.strftime("%I:%M %p")
    except:
        return ""


def export_chat_to_excel(metrics, filename='exports/chat_export.xlsx'):
    """Export chat metrics to Excel file"""
    try:
        os.makedirs('exports', exist_ok=True)
        
        df_new = pd.DataFrame(metrics)
        
        # If file exists, append (without duplicating previous rows)
        if os.path.exists(filename):
            df_existing = pd.read_excel(filename)
            combined = pd.concat([df_existing, df_new], ignore_index=True)
            # Drop duplicates, keeping the first occurrence
            combined = combined.drop_duplicates(
                subset=['query', 'output', 'response_time'], 
                keep='first'
            )
            combined.to_excel(filename, index=False)
        else:
            df_new.to_excel(filename, index=False)
        
        return True, f"Exported to {filename}"
    except Exception as e:
        return False, f"Error exporting: {str(e)}"


def probation_checker(current_date, profile_dict):
    """Check if user is under probation"""
    try:
        probation_end_date_str = profile_dict.get('probationEndDate')
        if not probation_end_date_str:
            return False

        probation_end_date = datetime.strptime(probation_end_date_str, '%Y-%m-%d').date()
        return probation_end_date >= current_date.date()

    except ValueError as e:
        print(f"Error parsing date: {e}")
        return False


def apply_custom_theme():
    """Apply custom theme and styling to Streamlit app"""
    st.markdown("""
    <style>
    /* Additional custom styles can be added here */
    .reportview-container {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 100%);
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Animated gradient text */
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
    }
    
    /* Card hover effect */
    .hover-card {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .hover-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.3);
    }
    
    /* Loading animation */
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }
    
    .loading {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    </style>
    """, unsafe_allow_html=True)


def create_feature_card(icon, title, description):
    """Create a feature card with icon, title, and description"""
    return f"""
    <div class="metric-card hover-card" style="text-align: center;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
        <h3 style="color: white; margin-bottom: 0.5rem;">{title}</h3>
        <p style="color: rgba(255, 255, 255, 0.7);">{description}</p>
    </div>
    """


def display_chat_message(role, content, avatar=""):
    """Display a chat message with proper formatting"""
    if role == "user":
        avatar = avatar or "👤"
        bg_color = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    else:
        avatar = avatar or "🧠"
        bg_color = "rgba(255, 255, 255, 0.05)"
    
    return f"""
    <div class="chat-message {role}" style="background: {bg_color};">
        <div class="chat-avatar">{avatar}</div>
        <div style="flex: 1;">{content}</div>
    </div>
    """


def get_example_queries():
    """Return a list of example queries for the chatbot"""
    return [
        "What is the minimum CGPA required for B.E. graduation?",
        "How does the re-evaluation process work?",
        "What are the credit requirements for B.E. programs?",
        "Explain the grading system and CGPA calculation",
        "What are the rules for vertical progression?",
        "What is the policy on internships?",
        "How many credits can I register for in a semester?",
        "What is the process for course registration?"
    ]


def display_metrics_table(metrics_data):
    """Display metrics in a formatted table"""
    if not metrics_data:
        return st.info("No metrics available yet. Start a conversation to see performance data.")
    
    df = pd.DataFrame(metrics_data)
    
    # Format columns
    if 'response_time' in df.columns:
        # Remove 's' suffix and convert to float for sorting
        df['response_time_numeric'] = df['response_time'].str.replace('s', '').astype(float)
        df = df.sort_values('response_time_numeric', ascending=False)
        df = df.drop('response_time_numeric', axis=1)
    
    # Display with custom styling
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "query": st.column_config.TextColumn(
                "Query",
                width="large",
            ),
            "output": st.column_config.TextColumn(
                "Output",
                width="large",
            ),
            "response_time": st.column_config.TextColumn(
                "Response Time",
                width="small",
            ),
            "retrieval_count": st.column_config.NumberColumn(
                "Retrieved Docs",
                width="small",
            ),
        }
    )
    
    return df


def create_hero_section():
    """Create an animated hero section for landing page"""
    return """
    <div style="text-align: center; padding: 4rem 0;">
        <div style="margin-bottom: 2rem;">
            <span style="font-size: 6rem;">🧠</span>
        </div>
        <h1 class="hero-title">The Future of Academic AI</h1>
        <p class="hero-subtitle">
            RAG revolutionizes how students and faculty interact with academic regulations.
            Get instant, accurate answers with our next-generation AI assistant.
        </p>
    </div>
    """


def display_stats(stat_data):
    """Display statistics in a visually appealing format"""
    cols = st.columns(len(stat_data))
    
    for i, (label, value) in enumerate(stat_data):
        with cols[i]:
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: rgba(255, 255, 255, 0.05); 
                        border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.1);">
                <h2 class="gradient-text" style="font-size: 3rem; margin: 0;">{value}</h2>
                <p style="color: rgba(255, 255, 255, 0.7); margin: 0.5rem 0 0 0;">{label}</p>
            </div>
            """, unsafe_allow_html=True)


def show_notification(message, type="info"):
    """Show a notification to the user"""
    if type == "success":
        st.success(message)
    elif type == "error":
        st.error(message)
    elif type == "warning":
        st.warning(message)
    else:
        st.info(message)


def validate_email(email):
    """Simple email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_password(password):
    """Validate password strength"""
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    return True, "Password is valid"


def clear_chat_history():
    """Clear the chat history from session state"""
    if 'chat_history' in st.session_state:
        st.session_state.chat_history = []
    if 'metrics' in st.session_state:
        st.session_state.metrics = []


def get_user_initials(email):
    """Get user initials from email"""
    if not email:
        return "?"
    name = email.split('@')[0]
    parts = name.split('.')
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).upper()
    return name[0].upper()

