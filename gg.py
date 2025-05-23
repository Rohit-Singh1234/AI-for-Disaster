import streamlit as st
import json
from collections import defaultdict
import requests
import os
from st_audiorec import st_audiorec
import assemblyai as aai
from elevenlabs import  play
from elevenlabs.client import ElevenLabs


   

# Configuration
GEMINI_API_KEY = "AIzaSyCyn7jLbnixPUvPs9lJMEhiFTXhlvqQA9c"  # Replace with your actual key
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
DATA_DIR = "F:\KU\hcf25\Streamlit\json_clean"  # Update this path

# Set page config   
st.set_page_config(
    page_title="Nepal Disaster Knowledge Base",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

audio_value = st.audio_input("Record a voice message")

if audio_value:
    st.audio(audio_value)
    with open("audio.wav", "wb") as f:
        f.write(audio_value.getvalue())

aai.settings.api_key = "01593ecce93243c299a595f26c4ece04"

# audio_file = "./local_file.mp3"
audio_file = "audio.wav"

config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best)

transcript = aai.Transcriber(config=config).transcribe(audio_file)

if transcript.status == "error":
  raise RuntimeError(f"Transcription failed: {transcript.error}")

st.write(transcript.text)
if transcript.text=="NULL":
    st.write("write your question in box below")
else:
    pathaune = transcript.text

class DisasterKnowledgeBase:    
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.load_all_data()
        self.build_relationships()
       
    def load_all_data(self):
        self.datasets = {
            'districts': self.load_json('district_clean.json'),
            'municipalities': self.load_json('municipality_clean.json'),
            'earthquakes': self.load_json('earthquake_clean.json'),
            'earthquake_risks': self.load_json('earthquake_risk_score_clean.json'),
            'fires': self.load_json('fire_clean.json'),
            'pollution': self.load_json('pollution_clean.json'),
            'rain': self.load_json('rain_clean.json'),
            'rivers': self.load_json('river_clean.json')
        }

    def load_json(self, filename):
        try:
            with open(os.path.join(self.data_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('results', [])
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            return []

    def build_relationships(self):
        self.district_province = {d['id']: d['province'] for d in self.datasets['districts']}
        self.municipality_district = {m['id']: m['district'] for m in self.datasets['municipalities']}
        self.province_districts = defaultdict(list)
        for d in self.datasets['districts']:
            self.province_districts[d['province']].append(d['id'])
        self.district_municipalities = defaultdict(list)
        for m in self.datasets['municipalities']:
            self.district_municipalities[m['district']].append(m['id'])
        self.district_names = {d['id']: d['title'] for d in self.datasets['districts']}
        self.municipality_names = {m['id']: m['title'] for m in self.datasets['municipalities']}

    def get_context_for_query(self, query):
        focus_areas = self._detect_geographic_focus(query)
        context_parts = [
            self._get_admin_hierarchy_context(focus_areas),
            self._get_earthquake_context(focus_areas),
            self._get_fire_context(focus_areas),
            self._get_pollution_context(focus_areas),
            self._get_rain_context(focus_areas),
            self._get_river_context(focus_areas),
            self._get_risk_score_context(focus_areas)
        ]
        return "\n\n".join([p for p in context_parts if p])

    def _detect_geographic_focus(self, query):
        focus_areas = {'provinces': set(), 'districts': set(), 'municipalities': set()}
        
        for p in set(d['province'] for d in self.datasets['districts']):
            if f"province {p}".lower() in query.lower():
                focus_areas['provinces'].add(p)
                
        for d in self.datasets['districts']:
            if d['title'].lower() in query.lower():
                focus_areas['districts'].add(d['id'])
                focus_areas['provinces'].add(d['province'])

        for m in self.datasets['municipalities']:
            if m['title'].lower() in query.lower():
                focus_areas['municipalities'].add(m['id'])
                focus_areas['districts'].add(m['district'])
                if m['district'] in self.district_province:
                    focus_areas['provinces'].add(self.district_province[m['district']])
        
        return focus_areas

    def _get_admin_hierarchy_context(self, focus_areas):
        if not any(focus_areas.values()):
            return None
        context = ["### Administrative Hierarchy"]
        for p in focus_areas['provinces']:
            districts = self.province_districts.get(p, [])
            context.append(f"Province {p} contains {len(districts)} districts")
            for d in districts:
                if d in focus_areas['districts'] or not focus_areas['districts']:
                    district_name = self.district_names.get(d, f"District {d}")
                    municipalities = self.district_municipalities.get(d, [])
                    context.append(f"- {district_name} (ID: {d}) has {len(municipalities)} municipalities")
        return "\n".join(context)

    def _get_earthquake_context(self, focus_areas):
        relevant_quakes = []
        for eq in self.datasets['earthquakes']:
            if (eq.get('district') in focus_areas['districts'] or 
                eq.get('municipality') in focus_areas['municipalities'] or
                (not focus_areas['districts'] and not focus_areas['municipalities'])):
                district_name = self.district_names.get(eq.get('district'), eq.get('district'))
                municipality_name = self.municipality_names.get(eq.get('municipality'), eq.get('municipality'))
                relevant_quakes.append(
                    f"Magnitude {eq['magnitude']} earthquake on {eq['eventOn']} "
                    f"(District: {district_name}, Municipality: {municipality_name})"
                )
        return "### Earthquake Events (last 5)\n" + "\n".join(relevant_quakes[:5]) if relevant_quakes else None

    def _get_fire_context(self, focus_areas):
        relevant_fires = []
        for fire in self.datasets['fires']:
            if fire.get('municipality') in focus_areas['municipalities'] or not focus_areas['municipalities']:
                relevant_fires.append(
                    f"{fire['title']} on {fire['eventOn']} "
                    f"(Brightness: {fire['brightness']}, Land Cover: {fire['landCover']})"
                )
        return "### Fire Events (last 3)\n" + "\n".join(relevant_fires[:3]) if relevant_fires else None

    def _get_pollution_context(self, focus_areas):
        relevant_pollution = []
        for poll in self.datasets['pollution']:
            if poll.get('municipality') in focus_areas['municipalities'] or not focus_areas['municipalities']:
                municipality_name = self.municipality_names.get(poll.get('municipality'), poll.get('municipality'))
                aqi_values = []
                for obs in poll.get('observations', []):
                    try:
                        aqi = float(obs['data']['aqi'])
                        aqi_values.append(aqi)
                    except (ValueError, TypeError):
                        continue
                if aqi_values:
                    avg_aqi = sum(aqi_values) / len(aqi_values)
                    relevant_pollution.append(
                        f"{poll['title']} in {municipality_name}: Average AQI {avg_aqi:.1f} "
                        f"on {poll['datetime']}"
                    )
        return "### Pollution Data (last 3)\n" + "\n".join(relevant_pollution[:3]) if relevant_pollution else None

    def _get_rain_context(self, focus_areas):
        relevant_rain = []
        for rain in self.datasets['rain']:
            if rain.get('municipality') in focus_areas['municipalities'] or not focus_areas['municipalities']:
                municipality_name = self.municipality_names.get(rain.get('municipality'), rain.get('municipality'))
                relevant_rain.append(
                    f"{rain['title']} in {municipality_name}: Status {rain['status']} "
                    f"on {rain['measuredOn']}"
                )
        return "### Rainfall Data (last 3)\n" + "\n".join(relevant_rain[:3]) if relevant_rain else None

    def _get_river_context(self, focus_areas):
        relevant_rivers = []
        for river in self.datasets['rivers']:
            if river.get('municipality') in focus_areas['municipalities'] or not focus_areas['municipalities']:
                municipality_name = self.municipality_names.get(river.get('municipality'), river.get('municipality'))
                relevant_rivers.append(
                    f"{river['title']} in {municipality_name}: Water level {river['waterLevel']} "
                    f"(Status: {river['status']})"
                )
        return "### River Data (last 3)\n" + "\n".join(relevant_rivers[:3]) if relevant_rivers else None

    def _get_risk_score_context(self, focus_areas):
        relevant_risks = []
        for risk in self.datasets['earthquake_risks']:
            if risk['district'] in focus_areas['districts'] or not focus_areas['districts']:
                district_name = self.district_names.get(risk['district'], risk['district'])
                relevant_risks.append(
                    f"{district_name} has earthquake risk score: {risk['riskScore']:.2f}"
                )
        return "### Earthquake Risk Scores\n" + "\n".join(relevant_risks) if relevant_risks else None

def query_gemini(api_key, prompt):
    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={api_key}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error querying Gemini API: {str(e)}")
        return None

def main():
    # Initialize knowledge base
    if 'kb' not in st.session_state:
        st.session_state.kb = DisasterKnowledgeBase()

    # Custom CSS for better styling
    st.markdown("""
    <style>
        .stTextInput input, .stTextArea textarea {
            border-radius: 8px;
            padding: 12px;
        }
        .stButton button {
            width: 100%;
            border-radius: 8px;
            padding: 12px;
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .answer-box {
            border-radius: 8px;
            padding: 20px;
            background-color: #2e353e;
            margin-top: 20px;
        }
        .example-question {
            cursor: pointer;
            padding: 4px;
            margin: 2px 0;
            border-radius: 4px;
            background-color: #e9f5ff;
        }
        .example-question:hover {
            background-color: #d0e6ff;
        }
    </style>
    """, unsafe_allow_html=True)

    # App header
    st.title("🌍 Nepal Disaster Knowledge Base")
    st.markdown("Ask questions about earthquakes, fires, pollution, rainfall, and rivers in Nepal")
    # Example questions
    example_questions = [
        "Recent earthquakes in Bajura",
        "Forest Fire at Surkhet",
        "Pollution levels in Achaam",
        "Rainfall data for Baglung"
    ]

    # Display example questions
    st.subheader("Example Questions")
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(question, key=f"example_{i}"):
                st.session_state.user_question = question

    # User input
    user_question = st.text_area(
        "Enter your question about Nepal disasters:",
        value=st.session_state.get('user_question', ''),
        height=100,
        key="user_question_input"
    )
    if transcript.text=="NULL":abc=user_question
    else:abc=pathaune

    # Submit button
    if st.button("Get Answer", disabled=not user_question.strip()):
        if not GEMINI_API_KEY or "AIza" in GEMINI_API_KEY and "YOUR_API_KEY_HERE" in GEMINI_API_KEY:
            st.error("Please set your Gemini API key in the GEMINI_API_KEY variable")
            return

        with st.spinner("Analyzing your question and generating answer..."):
            context = st.session_state.kb.get_context_for_query(user_question)

            prompt = f"""
            You are an expert disaster analysis assistant for Nepal. Use this context to answer the question.
            Provide a concise answer in simple language. If data isn't available, say so. also dont just return the data only, interpret it too.

            Context:
            {context}

            Question: {abc}

            Answer:
            """

            response = query_gemini(GEMINI_API_KEY, prompt)
            if response and 'candidates' in response and response['candidates']:
                global answer
                answer = response['candidates'][0]['content']['parts'][0]['text']
                st.session_state.last_answer = answer.strip()
                
           
            else:
                st.error("Failed to get response from Gemini API")

    # Display answer
    if 'last_answer' in st.session_state:
        st.markdown("### Answer")
        st.markdown(f'<div class="answer-box">{st.session_state.last_answer}</div>', unsafe_allow_html=True)
    
    
        # Show raw context (for debugging)
    if st.checkbox("Show context used for this answer"):
        context = st.session_state.kb.get_context_for_query(user_question)
        st.text_area("Context used:", value=context, height=300)
#idk


if __name__ == "__main__":
    main()

    #eleven=sk_b36681bd6fd3417c614aff4806458f9409b9b85b663c1b6e