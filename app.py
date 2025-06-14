import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Custom CSS to apply Satoshi font only
st.markdown("""
<style>
    /* Import Satoshi font */
    @import url('https://api.fontshare.com/v2/css?f[]=satoshi@400,500,700,900&display=swap');
    
    /* Apply Satoshi font to all elements */
    html, body, [class*="css"] {
        font-family: 'Satoshi', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }
    
    /* Ensure all text elements use Satoshi */
    h1, h2, h3, h4, h5, h6, p, div, span, li, a, button, input, textarea, label {
        font-family: 'Satoshi', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

# Define the emotion analysis models
class CircumplexEmotion(BaseModel):
    valence: float = Field(
        description="Valence score from -1 (negative) to 1 (positive)",
        ge=-1.0,
        le=1.0
    )
    arousal: float = Field(
        description="Arousal score from -1 (low) to 1 (high)",
        ge=-1.0,
        le=1.0
    )
    emotions: List[Dict[str, float]] = Field(
        description="List of emotions detected with their confidence scores"
    )
    explanation: str = Field(
        description="Brief explanation of the emotional state and why these scores were assigned. The paragraph should be written out of the user's perspective with directly addressing the user, Don't use any 'paragraph' or 'paragraphs' in the explanation."
    )

class ParagraphEmotion(BaseModel):
    paragraph: str = Field(description="The text content of the paragraph")
    circumplex: CircumplexEmotion = Field(description="Circumplex model analysis of the paragraph")
    dominant_emotion: str = Field(description="The most prominent emotion in this paragraph")

class ArticleAnalysis(BaseModel):
    paragraphs: List[ParagraphEmotion] = Field(description="List of analyzed paragraphs with their emotions")
    overall_emotion_flow: str = Field(description="Description of how emotions change throughout the article")
    key_emotional_points: List[str] = Field(description="Key points where significant emotional changes occur")

# Initialize the parsers
paragraph_parser = PydanticOutputParser(pydantic_object=ParagraphEmotion)
article_parser = PydanticOutputParser(pydantic_object=ArticleAnalysis)

# Create the prompt templates
segmentation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at analyzing emotional content in text using the Circumplex Model of Affect. 
    Your task is to:
    1. Read the entire article
    2. Identify natural breaks in emotional content
    3. Split the article into paragraphs based on emotional changes
    4. For each paragraph, analyze the emotional content using the Circumplex Model:
       - Valence: Score from -1 (negative) to 1 (positive)
       - Arousal: Score from -1 (low) to 1 (high)
    5. Identify the dominant emotion in each paragraph
    
    {format_instructions}
    """),
    ("user", "{text}")
])

paragraph_analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at analyzing emotions in text using the Circumplex Model of Affect.
    Analyze the given paragraph and provide:
    1. Valence score (-1 to 1): How positive or negative the emotional tone is
    2. Arousal score (-1 to 1): How activated or deactivated the emotional state is
    3. List of specific emotions with confidence scores
    4. Detailed explanation of the emotional analysis
    
    {format_instructions}
    """),
    ("user", "{text}")
])

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Create the chains
segmentation_chain = segmentation_prompt | llm | article_parser
paragraph_chain = paragraph_analysis_prompt | llm | paragraph_parser

# Streamlit UI
st.title("Article Emotion Analysis with Circumplex Model")
st.write("Enter your article below for emotion analysis:")

# Text input
text_input = st.text_area("Input Article", height=300, key="text_input")

if st.button("Analyze Article"):
    if text_input:
        with st.spinner("Analyzing article structure and emotions..."):
            try:
                # First stage: Analyze and segment the article
                article_analysis = segmentation_chain.invoke({
                    "text": text_input,
                    "format_instructions": article_parser.get_format_instructions()
                })
                
                # Display overall emotion flow
                st.subheader("Overall Emotion Flow")
                st.write(article_analysis.overall_emotion_flow)
                
                # Display key emotional points
                st.subheader("Key Emotional Points")
                for point in article_analysis.key_emotional_points:
                    st.write(f"• {point}")
                
                # Display detailed paragraph analysis
                st.subheader("Detailed Paragraph Analysis")
                for i, paragraph in enumerate(article_analysis.paragraphs, 1):
                    with st.expander(f"Paragraph {i} - {paragraph.dominant_emotion}"):
                        st.write("**Content:**")
                        st.write(paragraph.paragraph)
                        
                        # Display Circumplex Model scores
                        st.write("**Circumplex Model Analysis:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Valence", f"{paragraph.circumplex.valence:.2f}", 
                                    delta=None, delta_color="normal")
                        with col2:
                            st.metric("Arousal", f"{paragraph.circumplex.arousal:.2f}", 
                                    delta=None, delta_color="normal")
                        
                        # Display emotions breakdown
                        st.write("**Emotion Breakdown:**")
                        for emotion in paragraph.circumplex.emotions:
                            for emotion_name, confidence in emotion.items():
                                st.write(f"- {emotion_name}: {confidence:.2%}")
                        
                        # Display explanation
                        st.write("**Explanation:**")
                        st.write(paragraph.circumplex.explanation)

                st.session_state['analysis_results'] = {
                    'timestamp': datetime.now().isoformat(),
                    'input_text': text_input,
                    'analysis': {
                        "overall_emotion_flow": article_analysis.overall_emotion_flow,
                        "key_emotional_points": article_analysis.key_emotional_points,
                        "paragraphs": [
                            {
                                "number": i + 1,
                                "content": p.paragraph,
                                "dominant_emotion": p.dominant_emotion,
                                "valence": p.circumplex.valence,
                                "arousal": p.circumplex.arousal,
                                "emotions": p.circumplex.emotions,
                                "explanation": p.circumplex.explanation
                            }
                            for i, p in enumerate(article_analysis.paragraphs)
                        ]
                    }
                }
                if 'analysis_results' in st.session_state:
                    # Access the nested analysis data
                    results = st.session_state['analysis_results']['analysis']
                    
                    # Transform results into Circumplex Model format
                    total_paragraphs = len(results['paragraphs'])
                    circumplex_results = {
                        "paragraphs": []
                    }
                    
                    # Calculate proportions
                    for i, paragraph in enumerate(results['paragraphs']):
                        start = i / total_paragraphs
                        end = (i + 1) / total_paragraphs
                        proportion = end - start
                        
                        circumplex_results["paragraphs"].append({
                            "paragraph_number": paragraph['number'],
                            "proportion": {
                                "start": round(start, 3),
                                "end": round(end, 3),
                                "proportion": round(proportion, 3)
                            },
                            "emotions": {
                                "valence": paragraph['valence'],
                                "arousal": paragraph['arousal']
                            }
                        })
                    
                    # Display the results in a structured way
                    st.subheader("Analysis Results in JSON")
                    st.json(circumplex_results)
                    print("Circumplex Model Results: " + str(circumplex_results))
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

        