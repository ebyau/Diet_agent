# Elder Diet Recommendation System with Chat Memory
# Requirements: pip install streamlit langchain openai

import streamlit as st
import os
from datetime import datetime
from dataclasses import dataclass
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Elder Diet Planner",
    page_icon="🍎",
    layout="wide"
)

@dataclass
class ElderProfile:
    name: str
    age: int
    gender: str
    weight: float
    dietary_restrictions: str
    country: str = "USA"
    health_goal: str = "general wellness"

class ElderDietAgent:
    def __init__(self):
        self.llm = OpenAI(temperature=0.7, max_tokens=2500)
        # Initialize memory for conversation
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 exchanges
            return_messages=True,
            memory_key="chat_history"
        )
    
    def create_diet_prompt(self, profile: ElderProfile, duration: str) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert nutritionist specializing in elderly care with cultural sensitivity.
            Your task is to create personalized diet plans for seniors based on their profile and health goals.
            
            CRITICAL REQUIREMENTS:
            1. STRICTLY follow dietary restrictions - NO exceptions (e.g., "no nuts" means absolutely NO nuts/nut products)
            2. Include foods commonly available in the specified country
            3. FOCUS specifically on the stated health goal with targeted nutrition
            4. Consider cultural food preferences and local cuisine
            5. Ensure meals are elderly-appropriate (easy to chew, digest, prepare)
            6. Provide practical, affordable recommendations
            
            Always double-check that your recommendations don't violate any dietary restrictions."""),
            
            ("human", """Create a {duration} diet plan for this senior:

            PROFILE:
            - Name: {name}
            - Age: {age} years old
            - Gender: {gender}
            - Weight: {weight} kg
            - Dietary Restrictions: {dietary_restrictions}
            - Country: {country}
            - Health Goal: {health_goal}

            Please create a comprehensive {duration} meal plan that:
            1. Is appropriate for a {age}-year-old {gender}
            2. Considers their weight of {weight} kg
            3. STRICTLY respects dietary restrictions: {dietary_restrictions}
            4. Includes proper nutrition for elderly adults
            5. Provides easy-to-prepare meals suitable for seniors
            6. Includes portion sizes and meal timing
            7. SPECIFICALLY focuses on {health_goal} with targeted nutrition
            8. Incorporates foods commonly available in {country}
            9. Considers cultural food preferences from {country}

            Format your response with:
            📊 **NUTRITIONAL OVERVIEW**
            - Daily calorie and protein recommendations
            - Key nutrients for {health_goal}

            🍽️ **DETAILED MEAL PLAN**
            - Breakfast, lunch, dinner, and snacks
            - Specific portion sizes
            - Meal timing suggestions

            🎯 **HEALTH GOAL SUPPORT**
            - How each meal supports {health_goal}
            - Specific nutrients and their benefits

            🛒 **SHOPPING LIST**
            - Organized by food groups
            - Local/cultural alternatives where applicable

            💡 **PRACTICAL TIPS**
            - Easy preparation methods
            - Cultural cooking techniques from {country}

            ⚠️ **SAFETY REMINDERS**
            - Dietary restriction compliance check
            - Any foods to avoid

            Make this plan specific, culturally appropriate, and practical for {name} to follow.""")
        ])
    
    def create_chat_prompt(self, profile: ElderProfile) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", f"""You are a friendly, expert nutritionist specializing in elderly care. 
            You are currently helping {profile.name}, a {profile.age}-year-old {profile.gender} from {profile.country}.
            
            PROFILE CONTEXT:
            - Name: {profile.name}
            - Age: {profile.age} years old
            - Gender: {profile.gender}
            - Weight: {profile.weight} kg
            - Dietary Restrictions: {profile.dietary_restrictions}
            - Country: {profile.country}
            - Health Goal: {profile.health_goal}
            
            CONVERSATION GUIDELINES:
            1. Always keep their dietary restrictions in mind: {profile.dietary_restrictions}
            2. Consider foods available in {profile.country}
            3. Focus on their health goal: {profile.health_goal}
            4. Provide elderly-appropriate advice
            5. Be encouraging and supportive
            6. Ask clarifying questions when needed
            7. Offer practical, actionable advice
            
            You can help with:
            - Modifying meal plans
            - Explaining nutritional benefits
            - Suggesting alternatives
            - Answering food preparation questions
            - Addressing concerns about the diet plan
            - Providing shopping tips
            - Cultural food adaptations"""),
            
            ("placeholder", "{chat_history}"),
            ("human", "{input}")
        ])
    
    def generate_diet_plan(self, profile: ElderProfile, duration: str) -> str:
        prompt = self.create_diet_prompt(profile, duration)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        result = chain.run(
            duration=duration,
            name=profile.name,
            age=profile.age,
            gender=profile.gender,
            weight=profile.weight,
            dietary_restrictions=profile.dietary_restrictions if profile.dietary_restrictions else "None",
            country=profile.country,
            health_goal=profile.health_goal
        )
        
        # Store the diet plan in memory context
        self.memory.save_context(
            {"input": f"Generate a {duration} diet plan"},
            {"output": result}
        )
        
        return result
    
    def chat_with_agent(self, user_message: str, profile: ElderProfile) -> str:
        prompt = self.create_chat_prompt(profile)
        chain = LLMChain(
            llm=self.llm, 
            prompt=prompt, 
            memory=self.memory,
            verbose=False
        )
        
        response = chain.run(input=user_message)
        return response
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()

def initialize_session_state():
    """Initialize session state variables"""
    if 'agent' not in st.session_state:
        st.session_state.agent = ElderDietAgent()
    if 'profile' not in st.session_state:
        st.session_state.profile = None
    if 'diet_plan_generated' not in st.session_state:
        st.session_state.diet_plan_generated = False
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

def display_chat_message(role: str, content: str):
    """Display a chat message with appropriate styling"""
    if role == "user":
        with st.chat_message("user"):
            st.write(content)
    else:
        with st.chat_message("assistant"):
            st.write(content)

def main():
    st.title("🍎 Elder Diet Planner with Chat")
    st.markdown("### Personalized nutrition planning with conversational support")
    
    # Initialize session state
    initialize_session_state()
    
    # API Key setup
    with st.sidebar:
        st.header("⚙️ Setup")
        llm_option = st.radio("Choose LLM:", ["OpenAI", "Ollama"])
        api_key = st.text_input("OpenAI API Key", type="password")

        if llm_option == "OpenAI":
            api_key = st.text_input("OpenAI API Key", type="password")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                st.success("✅ OpenAI Ready!")
                #agent = ElderDietAgent(use_ollama=False)
            else:
                st.warning("Enter your OpenAI API key to use OpenAI")
                st.stop()
        elif llm_option == "Ollama":
            ollama_base = st.text_input("Ollama Base URL", value=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
            ollama_model_name = st.text_input("Ollama Model", value=os.environ.get("OLLAMA_MODEL", "llama2"))
            os.environ["OLLAMA_BASE_URL"] = ollama_base
            os.environ["OLLAMA_MODEL"] = ollama_model_name
            st.success(f"✅ Ollama Ready! Using model: {os.environ.get('OLLAMA_MODEL')}")
            #agent = ElderDietAgent(use_ollama=True)
        else:
            st.stop()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("✅ Ready!")
        else:
            st.warning("Enter your OpenAI API key")
            st.stop()
        
        # Profile status
        if st.session_state.profile:
            st.markdown("---")
            st.subheader("👤 Current Profile")
            st.write(f"**Name:** {st.session_state.profile.name}")
            st.write(f"**Age:** {st.session_state.profile.age}")
            st.write(f"**Country:** {st.session_state.profile.country}")
            st.write(f"**Goal:** {st.session_state.profile.health_goal}")
            
            if st.button("🔄 New Profile"):
                st.session_state.profile = None
                st.session_state.diet_plan_generated = False
                st.session_state.agent.clear_memory()
                st.session_state.chat_messages = []
                st.rerun()
        
        # Clear chat button
        if st.session_state.chat_messages:
            st.markdown("---")
            if st.button("🧹 Clear Chat"):
                st.session_state.agent.clear_memory()
                st.session_state.chat_messages = []
                st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📝 Profile Setup", "🍽️ Diet Plan", "💬 Chat with Nutritionist"])
    
    with tab1:
        if st.session_state.profile is None:
            st.header("👤 Basic Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Name")
                age = st.number_input("Age", min_value=50, max_value=100, value=70)
                health_goal = st.selectbox("Primary Health Goal", [
                    "General wellness",
                    "Weight management", 
                    "Heart health",
                    "Diabetes management",
                    "Bone health",
                    "Digestive health",
                    "Cognitive health",
                    "Energy levels",
                    "Blood pressure control"
                ])
            
            with col2:
                gender = st.selectbox("Gender", ["Female", "Male"])
                weight = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=70.0)
                country = st.selectbox("Country", [
                    "Uganda", "Kenya", "Tanzania", "Rwanda", "USA", "Canada", "UK", 
                    "India", "Nigeria", "South Africa", "Ghana", "Other"
                ])
            
            dietary_restrictions = st.text_area(
                "Dietary Restrictions", 
                placeholder="e.g., diabetic, low sodium, vegetarian, no nuts, etc.",
                help="List any dietary restrictions, allergies, or special requirements"
            )
            
            if st.button("💾 Save Profile", type="primary"):
                if not name:
                    st.error("Please enter a name")
                else:
                    st.session_state.profile = ElderProfile(
                        name=name,
                        age=age,
                        gender=gender,
                        weight=weight,
                        dietary_restrictions=dietary_restrictions,
                        country=country,
                        health_goal=health_goal
                    )
                    st.success(f"✅ Profile saved for {name}!")
                    st.info("👉 Go to 'Diet Plan' tab to generate your meal plan!")
        else:
            st.success("✅ Profile already saved!")
            st.info("👉 Go to 'Diet Plan' tab or 'Chat' tab to continue!")
    
    with tab2:
        if st.session_state.profile is None:
            st.warning("⚠️ Please create a profile first!")
            return
        
        st.header(f"🍽️ Diet Plan for {st.session_state.profile.name}")
        
        # Plan duration selection
        duration = st.radio("Plan Type:", ["Daily", "Weekly"], horizontal=True)
        
        # Generate plan
        if st.button("🚀 Generate Diet Plan", type="primary"):
            with st.spinner("Creating your personalized diet plan..."):
                try:
                    diet_plan = st.session_state.agent.generate_diet_plan(
                        st.session_state.profile, duration
                    )
                    
                    st.session_state.current_plan = diet_plan
                    st.session_state.diet_plan_generated = True
                    
                    st.success("✅ Diet plan created!")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Display generated plan
        if st.session_state.diet_plan_generated and 'current_plan' in st.session_state:
            st.markdown("---")
            st.subheader(f"{duration} Diet Plan")
            st.markdown(st.session_state.current_plan)
            
            # Download button
            st.download_button(
                "📥 Download Plan",
                st.session_state.current_plan,
                f"{st.session_state.profile.name.replace(' ', '_')}_diet_plan.txt",
                "text/plain"
            )
            
            st.info("💬 Go to the 'Chat' tab to ask questions or request modifications!")
    
    with tab3:
        if st.session_state.profile is None:
            st.warning("⚠️ Please create a profile first!")
            return
        
        st.header(f"💬 Chat with Your Nutritionist")
        st.markdown(f"*Ask questions about {st.session_state.profile.name}'s nutrition plan*")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_messages:
                display_chat_message(message["role"], message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about the diet plan, request changes, or get nutrition advice..."):
            # Add user message to chat
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.agent.chat_with_agent(
                        prompt, st.session_state.profile
                    )
                    st.write(response)
            
            # Add AI response to chat
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
        
        # Suggested questions
        if not st.session_state.chat_messages:
            st.subheader("💡 Try asking:")
            suggestions = [
                "Can you modify breakfast to include more protein?",
                "What are some alternatives to the suggested snacks?",
                "How can I make these meals easier to prepare?",
                "Are there local alternatives to the suggested foods?",
                "Can you explain why these foods help with my health goal?",
                "What if I don't like some of the suggested foods?"
            ]
            
            for suggestion in suggestions:
                if st.button(f"💭 {suggestion}", key=suggestion):
                    st.session_state.chat_messages.append({"role": "user", "content": suggestion})
                    
                    with st.spinner("Thinking..."):
                        response = st.session_state.agent.chat_with_agent(
                            suggestion, st.session_state.profile
                        )
                    
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    st.rerun()

if __name__ == "__main__":
    main()
