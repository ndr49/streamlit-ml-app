import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import string
import re
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page Configuration
st.set_page_config(
    page_title="Textective",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .ai-result {
        background: #ffebee;
        border: 2px solid #f44336;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .human-result {
        background: #e3f2fd;
        border: 2px solid #2196f3;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load Models Function
@st.cache_resource
def load_models():
    """Load AI detection models"""
    models = {}
    model_files = {
        'vectorizer': 'models/tfidf_vectorizer.pkl',
        'svm': 'models/svm_model.pkl',
        'decision_tree': 'models/decision_tree_model.pkl',
        'adaboost': 'models/adaboost_model.pkl'
    }
    
    loaded_models = []
    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    models[name] = pickle.load(f)
                loaded_models.append(name)
            except:
                continue
    
    return models if 'vectorizer' in loaded_models and len(loaded_models) > 1 else None

# Prediction Function
def predict_text(text, models):
    """Get predictions from all available models"""
    if not models or not text.strip():
        return None
    
    try:
        # Simple text preprocessing without NLTK
        def text_process(essay):
            """Simple text preprocessing without NLTK"""
            if not isinstance(essay, str):
                return ""
            
            # Simple stopwords list
            STOPWORDS = [
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'could', 'should', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
                'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
                'her', 'its', 'our', 'their', 'not', 'no', 'yes', 'can', 'may', 'might', 'must',
                'shall', 'from', 'up', 'out', 'down', 'off', 'over', 'under', 'again', 'further',
                'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so',
                'than', 'too', 'very', 'just', 'now'
            ]
            
            # Remove punctuation
            nopunc = ''.join([char for char in essay if char not in string.punctuation])
            
            # Remove numbers and URLs
            cleaned_text = re.sub(r'\S*\d\S*', '', nopunc)
            cleaned_text = re.sub(r'http\S+|www.\S+', '', cleaned_text)
            cleaned_text = re.sub(r'\S*@\S*\s?', '', cleaned_text)
            
            # Simple word processing without lemmatization
            words = [word.lower().strip() for word in cleaned_text.split() 
                    if word.lower().strip() not in STOPWORDS and len(word.strip()) > 2]
            
            return ' '.join(words)
        
        # Preprocess the text
        processed_text = text_process(text)
        
        if not processed_text.strip():
            return None
        
        # Make predictions with all available models
        results = {}
        model_names = {'svm': 'SVM', 'decision_tree': 'Decision Tree', 'adaboost': 'AdaBoost'}
        
        for model_key, display_name in model_names.items():
            if model_key in models:
                try:
                    # Use the processed text for pipeline models
                    prediction = models[model_key].predict([processed_text])[0]
                    
                    # Get probabilities if available
                    if hasattr(models[model_key], 'predict_proba'):
                        proba = models[model_key].predict_proba([processed_text])[0]
                    else:
                        proba = [0.75, 0.25] if prediction == 0 else [0.25, 0.75]
                    
                    results[display_name] = {
                        'prediction': 'AI' if prediction == 0 else 'Human',
                        'confidence': max(proba),
                        'ai_prob': proba[0],
                        'human_prob': proba[1]
                    }
                except Exception as e:
                    print(f"Error with {model_key}: {e}")
                    continue
        
        return results
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

# Text Analysis Function
def analyze_text_features(text):
    """Analyze text characteristics"""
    words = text.split()
    sentences = text.split('.')
    
    # Basic statistics
    features = {
        'word_count': len(words),
        'char_count': len(text),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        'avg_sentence_length': len(words) / max(len([s for s in sentences if s.strip()]), 1)
    }
    
    # AI indicators
    ai_phrases = ['furthermore', 'moreover', 'therefore', 'consequently', 'implementation', 
                  'optimization', 'comprehensive', 'systematic', 'methodology']
    features['ai_indicators'] = sum(1 for phrase in ai_phrases if phrase.lower() in text.lower())
    
    # Human indicators  
    human_phrases = ["i'm", "you're", "can't", "don't", "honestly", "literally", 
                     "amazing", "awesome", "omg", "lol"]
    features['human_indicators'] = sum(1 for phrase in human_phrases if phrase.lower() in text.lower())
    
    return features

# Create visualization
def create_results_viz(results):
    """Create beautiful results visualization"""
    if not results:
        return None
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Model Predictions', 'Confidence Levels', 'AI vs Human Probabilities', 'Consensus Analysis'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "pie"}]]
    )
    
    models = list(results.keys())
    predictions = [results[m]['prediction'] for m in models]
    confidences = [results[m]['confidence'] for m in models]
    ai_probs = [results[m]['ai_prob'] for m in models]
    human_probs = [results[m]['human_prob'] for m in models]
    
    # Predictions bar
    colors = ['#ff6b6b' if pred == 'AI' else '#4ecdc4' for pred in predictions]
    fig.add_trace(go.Bar(x=models, y=[1]*len(models), marker_color=colors, 
                    text=predictions, name="Predictions", showlegend=False), row=1, col=1)
    
    # Confidence levels
    fig.add_trace(go.Bar(x=models, y=confidences, marker_color='#95a5a6', name="Confidence"), row=1, col=2)
    
    # Probabilities
    fig.add_trace(go.Bar(x=models, y=ai_probs, name="AI Probability", marker_color='#ff6b6b'), row=2, col=1)
    fig.add_trace(go.Bar(x=models, y=human_probs, name="Human Probability", marker_color='#4ecdc4'), row=2, col=1)
    
    # Consensus pie
    ai_votes = sum(1 for pred in predictions if pred == 'AI')
    human_votes = len(predictions) - ai_votes
    fig.add_trace(go.Pie(labels=['AI', 'Human'], values=[ai_votes, human_votes], 
                        marker_colors=['#ff6b6b', '#4ecdc4']), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False, title_text="Analysis Results")
    return fig

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🕵️ Textective </h1>
        <p>A machine learning web application that classifies text as AI-generated or human-written using Support Vector Machine (SVM), Decision Tree, and AdaBoost algorithms. 
        This project was developed for the Large Language Models course to create a comprehensive text classification system that distinguishes between AI-generated and human-written content using multiple machine learning approaches.</p>
        <small>Large Language Models Course Project</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    
    # Sidebar
    with st.sidebar:
        st.title("🔧 Controls")
        
        # Model status
        if models:
            st.success("✅ Models Loaded Successfully")
            available_models = [k for k in ['svm', 'decision_tree', 'adaboost'] if k in models]
            for model in available_models:
                icon = "⚙️" if model == 'svm' else "🌳" if model == 'decision_tree' else "🚀"
                st.text(f"{icon} {model.upper().replace('_', ' ')}")
        else:
            st.error("❌ Models not found!")
            st.info("Please ensure model files are in the 'models/' directory")
        
        st.markdown("---")
        
        # Analysis options
        st.subheader("📊 Analysis Options")
        show_details = st.checkbox("Show detailed analysis", value=True)
        show_features = st.checkbox("Show text features", value=True)
        show_viz = st.checkbox("Show visualizations", value=True)
        
        st.markdown("---")
        st.markdown("""
        ### 💡 About
        This app uses three ML models:
        - **SVM**: Pattern recognition
        - **Decision Tree**: Rule-based logic  
        - **AdaBoost**: Ensemble learning
        
        Upload text to detect if it's AI or human-written!
        """)
    
    # Main content
    if models:
        # Text input tabs
        tab1, tab2 = st.tabs(["📝 Text Input", "📄 File Upload"])
        
        with tab1:
            st.subheader("Enter Text for Analysis")
            
            # Example buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🤖 Try AI Example"):
                    st.session_state.sample_text = "The implementation of artificial intelligence in modern healthcare systems represents a significant advancement in medical technology. This development enables more accurate diagnostic procedures and streamlined patient care processes through comprehensive data analysis."
            
            with col2:
                if st.button("👤 Try Human Example"):
                    st.session_state.sample_text = "OMG, I can't believe how amazing this new coffee shop is! The barista was super friendly and made the most incredible latte art. I'm definitely going back tomorrow - this place is going to be my new addiction!"
            
            # Text area
            text_input = st.text_area(
                "Paste your text here:",
                value=st.session_state.get('sample_text', ''),
                height=200,
                placeholder="Enter text to analyze whether it was written by AI or human..."
            )
            
            # Quick stats
            if text_input:
                words = len(text_input.split())
                chars = len(text_input)
                col1, col2, col3 = st.columns(3)
                col1.metric("Words", words)
                col2.metric("Characters", chars)
                col3.metric("Sentences", len([s for s in text_input.split('.') if s.strip()]))
        
        with tab2:
            st.subheader("Upload Document")
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['txt', 'pdf', 'docx'],
                help="Upload text files for batch analysis"
            )
            
            if uploaded_file:
                # File processing would go here
                st.info("📄 File uploaded successfully! (Processing not implemented in this demo)")
                text_input = ""  # Reset for demo
        
        # Analysis button
        if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
            if text_input and text_input.strip():
                with st.spinner("🤖 Analyzing text with AI detection models..."):
            
                    # Get predictions
                    results = predict_text(text_input, models)
                    
                    if results:
                        st.success("✅ Analysis Complete!")
                        
                        # Main results
                        st.subheader("🎯 Detection Results")
                        
                        # Show individual model results
                        cols = st.columns(len(results))
                        for i, (model_name, result) in enumerate(results.items()):
                            with cols[i]:
                                prediction = result['prediction']
                                confidence = result['confidence']
                                
                                if prediction == "AI":
                                    st.markdown(f"""
                                    <div class="ai-result">
                                        <h4>🤖 {model_name}</h4>
                                        <h2>AI Generated</h2>
                                        <p>Confidence: {confidence:.1%}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="human-result">
                                        <h4>👤 {model_name}</h4>
                                        <h2>Human Written</h2>
                                        <p>Confidence: {confidence:.1%}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Consensus
                        ai_votes = sum(1 for r in results.values() if r['prediction'] == 'AI')
                        total_votes = len(results)
                        consensus = "AI Generated" if ai_votes > total_votes/2 else "Human Written"
                        consensus_pct = max(ai_votes, total_votes - ai_votes) / total_votes
                        
                        st.subheader("🏛️ Final Verdict")
                        if consensus == "AI Generated":
                            st.error(f"🤖 **{consensus}** ({ai_votes}/{total_votes} models agree, {consensus_pct:.1%} consensus)")
                        else:
                            st.success(f"👤 **{consensus}** ({total_votes-ai_votes}/{total_votes} models agree, {consensus_pct:.1%} consensus)")
                        
                        # Detailed analysis
                        if show_details:
                            st.subheader("📊 Detailed Analysis")
                            
                            # Create dataframe for detailed view
                            detail_data = []
                            for model_name, result in results.items():
                                detail_data.append({
                                    'Model': model_name,
                                    'Prediction': result['prediction'],
                                    'Confidence': f"{result['confidence']:.1%}",
                                    'AI Probability': f"{result['ai_prob']:.1%}",
                                    'Human Probability': f"{result['human_prob']:.1%}"
                                })
                            
                            st.dataframe(pd.DataFrame(detail_data), use_container_width=True)
                        
                        # Text features
                        if show_features:
                            st.subheader("📈 Text Characteristics")
                            features = analyze_text_features(text_input)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Word Count", features['word_count'])
                            col2.metric("Avg Word Length", f"{features['avg_word_length']:.1f}")
                            col3.metric("AI Indicators", features['ai_indicators'])
                            col4.metric("Human Indicators", features['human_indicators'])
                        
                        # Visualizations
                        if show_viz:
                            st.subheader("📊 Visual Analysis")
                            fig = create_results_viz(results)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Explanation
                        st.subheader("💡 What This Means")
                        if consensus == "AI Generated":
                            st.info("🤖 **AI Generated**: This text shows patterns typical of AI-generated content, such as formal language structure, technical vocabulary, and consistent stylistic patterns.")
                        else:
                            st.info("👤 **Human Written**: This text displays characteristics of human writing, including personal expression, varied sentence structure, emotional language, or informal elements.")
                        
                    else:
                        st.error("❌ Failed to analyze text. Please try again.")
            else:
                st.warning("⚠️ Please enter some text to analyze!")
    
    else:
        # No models loaded - show help
        st.error("❌ No models found!")
        st.markdown("""
        ### 🔧 Setup Instructions:
        
        1. **Create models folder**: `mkdir models`
        2. **Train and save your models** as:
           - `models/svm_model.pkl`
           - `models/decision_tree_model.pkl`
           - `models/adaboost_model.pkl`
           - `models/tfidf_vectorizer.pkl`
        3. **Restart the app**
        
        ### 📚 Model Training:
        Use scikit-learn to train your models and save them with pickle:
        ```python
        import pickle
        
        # After training your models...
        with open('models/svm_model.pkl', 'wb') as f:
            pickle.dump(svm_model, f)
        ```
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🕵️ AI vs Human Text Detective | Built with Streamlit | Large Language Models Course</p>
        <p><a href="https://github.com/ndr49/ai-human-detection-project" target="_blank">🔗 View on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()