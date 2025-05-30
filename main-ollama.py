import streamlit as st
import os
from datetime import datetime
from dataclasses import dataclass
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import os

# Load environment variables from .env file
load_dotenv()

#access environment variables
openaiapi_key = os.environ.get("OPENAI_SECRET_KEY")
tavily_key = os.environ.get("TAVILY_API_KEY")
ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_model = os.environ.get("OLLAMA_MODEL", "llama2")


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
    health_goal: str = "Maintain health"

class ElderDietAgent:
    def __init__(self, use_ollama=False):
        self.use_ollama = use_ollama
        if self.use_ollama:
            self.llm = ChatOllama(base_url=ollama_base_url, model=ollama_model)
        else:
            self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

    def create_diet_prompt(self, profile: ElderProfile, duration: str):
        template = """
        You are an expert nutritionist specializing in elderly care with cultural sensitivity.
        Your task is to create a personalized diet plan for seniors based on their profile and health goals.


        Create a {duration} diet plan for this senior:

        PROFILE:
        - Name: {name}
        - Age: {age} years old
        - Gender: {gender}
        - Weight: {weight} kg
        - Dietary Restrictions: {dietary_restrictions}
        - Country: {country}
        - Health Goal: {health_goal}

        IMPORTANT REQUIREMENTS:
        1. STRICTLY follow dietary restrictions - if "no nuts" means NO nuts/nut products
        2. Include foods commonly available in {country}
        3. FOCUS specifically on {health_goal} with targeted nutrition
        4. Consider cultural food preferences from {country}

        Please create a comprehensive {duration} meal plan that:
        1. Is appropriate for a {age}-year-old {gender}
        2. Considers their weight of {weight} kg
        3. Respects their dietary restrictions: {dietary_restrictions}
        4. Includes proper nutrition for elderly adults
        5. Provides easy-to-prepare meals
        6. Includes portion sizes and meal timing
        7. Focuses on health goals like {health_goal}
        8. Consider cultural food preferences from {country}

        Format the response with:
        - Daily calorie and protein recommendations
        - Detailed meal plan (breakfast, lunch, dinner, snacks)
        - Provide information on how suggested foods support health goals
        - Nutritional highlights

        Make it practical and easy to follow for {name}.
        """

        return PromptTemplate(
            input_variables=["duration", "name", "age", "gender", "weight", "dietary_restrictions", "country", "health_goal"],
            template=template
        )

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

        return result

def main():
    st.title("🍎 Elder Diet Planner")
    st.markdown("### Simple nutrition planning for seniors")

    # LLM Selection in the sidebar
    with st.sidebar:
        st.header("⚙️ Setup")
        llm_option = st.radio("Choose LLM:", ["OpenAI", "Ollama"])

        if llm_option == "OpenAI":
            openai_api_key = st.text_input("OpenAI API Key", type="password")
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
                st.success("✅ Using OpenAI")
                llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
            elif "OPENAI_API_KEY" in os.environ:
                st.success("✅ Using OpenAI (API key found in environment)")
                llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
            else:
                st.warning("Enter your OpenAI API key to use OpenAI")
                llm = None
        elif llm_option == "Ollama":
            ollama_base = st.text_input("Ollama Base URL", value=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
            ollama_model_name = st.text_input("Ollama Model", value=os.environ.get("OLLAMA_MODEL", "llama2"))
            os.environ["OLLAMA_BASE_URL"] = ollama_base
            os.environ["OLLAMA_MODEL"] = ollama_model_name
            st.success(f"✅ Using Ollama with model: {os.environ.get('OLLAMA_MODEL')}")
            llm = ChatOllama(base_url=os.environ.get("OLLAMA_BASE_URL"), model=os.environ.get("OLLAMA_MODEL"))
        else:
            llm = None
            st.stop()

    # Initialize agent based on selected LLM
    if 'agent_type' not in st.session_state:
        st.session_state.agent_type = llm_option
    elif st.session_state.agent_type != llm_option:
        st.session_state.agent_type = llm_option
        st.session_state.agent = None # Reset agent if LLM changes

    if 'agent' not in st.session_state:
        if llm_option == "OpenAI" and ("OPENAI_API_KEY" in os.environ or st.sidebar.get_child(1).value): # Check if API key is available
            st.session_state.agent = ElderDietAgent(use_ollama=False)
        elif llm_option == "Ollama" and llm:
            st.session_state.agent = ElderDietAgent(use_ollama=True)
        else:
            st.warning("Please configure the LLM in the sidebar.")

    # Simple form
    st.header("👤 Basic Information")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=50, max_value=100, value=70)
        health_goal = st.text_input("Health Goal",
                                    placeholder="e.g., maintain health, lose weight,improve cognitive health",
                                    )

    with col2:
        gender = st.selectbox("Gender", ["Female", "Male"])
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=70.0)
        country = st.text_input("Country", value="USA")

    dietary_restrictions = st.text_area(
        "Dietary Restrictions",
        placeholder="e.g., diabetic, low sodium, vegetarian, no nuts, etc.",
        help="List any dietary restrictions, allergies, or special requirements"
    )

    # Plan duration
    duration = st.radio("Plan Type:", ["Daily", "Weekly"], horizontal=True)

    # Generate plan
    if st.button("🍽️ Generate Diet Plan", type="primary"):
        if not name:
            st.error("Please enter a name")
            return

        if 'agent' in st.session_state and st.session_state.agent:
            profile = ElderProfile(
                name=name,
                age=age,
                gender=gender,
                weight=weight,
                dietary_restrictions=dietary_restrictions,
                country=country,
                health_goal=health_goal
            )

            with st.spinner(f"Creating your personalized diet plan using {llm_option} ..."):
                try:
                    diet_plan = st.session_state.agent.generate_diet_plan(profile, duration)

                    st.success("✅ Diet plan created!")
                    st.markdown("---")
                    st.subheader(f"{duration} Diet Plan for {name}")
                    st.markdown(diet_plan)

                    # Download button
                    st.download_button(
                        "📥 Download Plan",
                        diet_plan,
                        f"{name.replace(' ', '_')}_diet_plan.txt",
                        "text/plain"
                    )

                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please configure the LLM in the sidebar.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("💬 Chat with the LLM")
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the diet plan or elderly nutrition:"):
        if llm:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner(f"Thinking with {llm_option} ..."):
                    try:
                        full_response = ""
                        if isinstance(llm, ChatOpenAI) or isinstance(llm, ChatOllama):
                            response = llm.invoke(prompt)
                            if hasattr(response, 'content'):
                                full_response = response.content
                            else:
                                full_response = str(response)
                        else:
                            full_response = llm.predict(prompt)
                        st.markdown(full_response)
                    except Exception as e:
                        st.error(f"Error during chat: {e}")
                        full_response = f"An error occurred: {e}"
                st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.warning("Please configure the LLM in the sidebar to chat.")

if __name__ == "__main__":
    main()
