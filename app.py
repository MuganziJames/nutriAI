# 🌍 NutriAI East Africa
# Intelligent Nutrition Planning for East African Families
# SDG 2 (Zero Hunger) & SDG 3 (Good Health and Well-being)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
from textblob import TextBlob
import nltk
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="NutriAI East Africa",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .success-card {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .food-tag {
        display: inline-block;
        background: #007bff;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    .nutrient-progress {
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'meal_plan_generated' not in st.session_state:
    st.session_state.meal_plan_generated = False
if 'current_meal_plan' not in st.session_state:
    st.session_state.current_meal_plan = None
if 'nutrition_analysis' not in st.session_state:
    st.session_state.nutrition_analysis = None

# Import custom modules
from data_processor import DataProcessor
from nlp_processor import NLPProcessor
from ai_predictors import SupplyChainPredictor, NutritionDeficiencyPredictor, EconomicImpactModeler, CulturalFoodMapper
from meal_planner import MealPlanningEngine

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all AI components"""
    data_processor = DataProcessor()
    nlp_processor = NLPProcessor()
    supply_predictor = SupplyChainPredictor(data_processor.price_data)
    nutrition_predictor = NutritionDeficiencyPredictor(data_processor.nutrition_data)
    economic_modeler = EconomicImpactModeler()
    cultural_mapper = CulturalFoodMapper()
    
    meal_planner = MealPlanningEngine(
        data_processor, nlp_processor, supply_predictor, nutrition_predictor
    )
    
    return {
        'data_processor': data_processor,
        'nlp_processor': nlp_processor,
        'supply_predictor': supply_predictor,
        'nutrition_predictor': nutrition_predictor,
        'economic_modeler': economic_modeler,
        'cultural_mapper': cultural_mapper,
        'meal_planner': meal_planner
    }

# Initialize components
components = initialize_components()

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌍 NutriAI East Africa</h1>
        <p>Intelligent Nutrition Planning for East African Families</p>
        <p><strong>SDG 2: Zero Hunger | SDG 3: Good Health and Well-being</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # Country selection
    country = st.sidebar.selectbox(
        "Select Country",
        ["Kenya", "Uganda", "Tanzania", "Ethiopia", "Rwanda", "Burundi"],
        index=0
    )
    
    # Budget slider
    budget = st.sidebar.slider(
        "Weekly Budget (KES)",
        min_value=200,
        max_value=5000,
        value=1000,
        step=50,
        help="Set your weekly food budget in Kenyan Shillings"
    )
    
    # Family size
    family_size = st.sidebar.number_input(
        "Family Size",
        min_value=1,
        max_value=10,
        value=4,
        help="Number of people in your family"
    )
    
    # Dietary restrictions
    st.sidebar.subheader("Dietary Preferences")
    dietary_restrictions = []
    
    if st.sidebar.checkbox("Vegetarian"):
        dietary_restrictions.append("vegetarian")
    if st.sidebar.checkbox("No Dairy"):
        dietary_restrictions.append("no_dairy")
    if st.sidebar.checkbox("Gluten Free"):
        dietary_restrictions.append("gluten_free")
    if st.sidebar.checkbox("Diabetic Friendly"):
        dietary_restrictions.append("diabetic")
    if st.sidebar.checkbox("Pregnancy Nutrition"):
        dietary_restrictions.append("pregnancy")
    
    # Special needs
    st.sidebar.subheader("Special Needs")
    special_needs = []
    
    if st.sidebar.checkbox("Has Children"):
        special_needs.append("children")
    if st.sidebar.checkbox("Elderly Family Members"):
        special_needs.append("elderly")
    if st.sidebar.checkbox("Expecting Mother"):
        special_needs.append("pregnant")
    
    # Natural language input
    st.subheader("🗣️ Tell us about your food preferences")
    user_input = st.text_area(
        "Describe your food preferences, available ingredients, or specific needs...",
        placeholder="Example: I have maize, beans, and sukuma wiki. My family likes ugali and we need high-protein meals. Budget is tight this week.",
        height=100
    )
    
    # Generate meal plan button
    if st.button("🍽️ Generate Meal Plan", type="primary"):
        if user_input.strip():
            with st.spinner("🤖 AI is analyzing your preferences and generating optimal meal plan..."):
                
                # Prepare user preferences
                user_preferences = {
                    'country': country,
                    'budget': budget,
                    'family_size': family_size,
                    'dietary_restrictions': dietary_restrictions,
                    'special_needs': special_needs,
                    'input_text': user_input
                }
                
                # Generate meal plan
                meal_plan_result = components['meal_planner'].generate_meal_plan(user_preferences)
                
                # Store in session state
                st.session_state.current_meal_plan = meal_plan_result
                st.session_state.meal_plan_generated = True
                
                st.success("✅ Meal plan generated successfully!")
                st.rerun()
        else:
            st.warning("Please describe your food preferences or needs.")
    
    # Display meal plan if generated
    if st.session_state.meal_plan_generated and st.session_state.current_meal_plan:
        display_meal_plan(st.session_state.current_meal_plan)
    
    # AI Insights Section
    st.subheader("🤖 AI Insights & Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Supply Chain Intelligence")
        
        # Get supply chain predictions
        availability_predictions = components['supply_predictor'].predict_food_availability(country)
        
        if availability_predictions:
            current_month = datetime.now().month
            
            for month, data in list(availability_predictions.items())[:3]:
                month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                if month in month_names:
                    availability_score = data['availability_score']
                    price_trend = data['price_trend']
                    
                    if price_trend == 'low':
                        color = "#28a745"
                        icon = "📉"
                    elif price_trend == 'high':
                        color = "#dc3545"
                        icon = "📈"
                    else:
                        color = "#ffc107"
                        icon = "📊"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{icon} {month_names[month]}</strong><br>
                        Availability: {availability_score:.0f}%<br>
                        Price Trend: <span style="color: {color};">{price_trend.upper()}</span>
                    </div>
                    """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🏥 Nutrition Risk Assessment")
        
        if st.session_state.current_meal_plan:
            # Analyze nutrition risks
            meal_plan = st.session_state.current_meal_plan['meal_plan']
            user_prefs = st.session_state.current_meal_plan['user_preferences']
            
            nutrition_risks = components['nutrition_predictor'].predict_deficiency_risk(meal_plan, user_prefs)
            
            for nutrient, risk_level in nutrition_risks.items():
                if risk_level == 'high':
                    color = "#dc3545"
                    icon = "⚠️"
                    card_class = "warning-card"
                elif risk_level == 'medium':
                    color = "#ffc107"
                    icon = "⚡"
                    card_class = "warning-card"
                else:
                    color = "#28a745"
                    icon = "✅"
                    card_class = "success-card"
                
                st.markdown(f"""
                <div class="{card_class}">
                    <strong>{icon} {nutrient.title()}</strong><br>
                    Risk Level: <span style="color: {color};">{risk_level.upper()}</span>
                </div>
                """, unsafe_allow_html=True)
    
    # Cultural Food Mapping
    st.subheader("🌍 Cultural Food Intelligence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🥘 Traditional Foods")
        traditional_meal = components['cultural_mapper'].get_culturally_appropriate_meal(
            country, 'lunch', dietary_restrictions
        )
        
        st.markdown(f"""
        **Suggested Traditional Meal:**
        - **Staple:** {traditional_meal['staple']}
        - **Protein:** {traditional_meal['protein']}
        - **Vegetable:** {traditional_meal['vegetable']}
        """)
    
    with col2:
        st.markdown("### 💰 Economic Impact")
        
        if st.session_state.current_meal_plan:
            cost_breakdown = st.session_state.current_meal_plan['cost_breakdown']
            
            st.metric(
                "Weekly Cost",
                f"KES {cost_breakdown['total_weekly_cost']:.0f}",
                f"Budget: KES {budget}"
            )
            
            savings = budget - cost_breakdown['total_weekly_cost']
            if savings > 0:
                st.markdown(f"""
                <div class="success-card">
                    <strong>💰 Savings: KES {savings:.0f}</strong><br>
                    You're within budget!
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-card">
                    <strong>⚠️ Over Budget: KES {abs(savings):.0f}</strong><br>
                    Consider cheaper alternatives
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    ---
    <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
        <h4>🌍 "The right to food is a basic human right"</h4>
        <p><em>Together, we can achieve Zero Hunger and Good Health for all East African families</em></p>
        <p><strong>SDG 2 | SDG 3 | Powered by AI for Social Good</strong></p>
    </div>
    """, unsafe_allow_html=True)

def display_meal_plan(meal_plan_result):
    """Display the generated meal plan"""
    meal_plan = meal_plan_result['meal_plan']
    nutrition_analysis = meal_plan_result['nutrition_analysis']
    cost_breakdown = meal_plan_result['cost_breakdown']
    shopping_list = meal_plan_result['shopping_list']
    
    st.subheader("📋 Your 7-Day Meal Plan")
    
    # Meal plan tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Daily Meals", "📊 Nutrition Analysis", "💰 Cost Breakdown", "🛒 Shopping List"])
    
    with tab1:
        for day in meal_plan:
            with st.expander(f"Day {day['day']} - {day['date']}", expanded=day['day'] <= 2):
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 🌅 Breakfast")
                    breakfast = day['meals']['breakfast']
                    
                    st.markdown(f"**{breakfast['description']}**")
                    
                    for food in breakfast['foods']:
                        st.markdown(f"<span class='food-tag'>{food}</span>", unsafe_allow_html=True)
                    
                    st.markdown(f"💰 **Cost:** KES {breakfast['estimated_cost']:.0f}")
                    st.markdown(f"⏱️ **Prep time:** {breakfast['preparation_time']} min")
                    
                    # Nutrition score bar
                    nutrition_score = breakfast['nutrition_score']
                    st.progress(nutrition_score / 100)
                    st.caption(f"Nutrition Score: {nutrition_score:.0f}/100")
                
                with col2:
                    st.markdown("### 🌞 Lunch")
                    lunch = day['meals']['lunch']
                    
                    st.markdown(f"**{lunch['description']}**")
                    
                    for food in lunch['foods']:
                        st.markdown(f"<span class='food-tag'>{food}</span>", unsafe_allow_html=True)
                    
                    st.markdown(f"💰 **Cost:** KES {lunch['estimated_cost']:.0f}")
                    st.markdown(f"⏱️ **Prep time:** {lunch['preparation_time']} min")
                    
                    # Nutrition score bar
                    nutrition_score = lunch['nutrition_score']
                    st.progress(nutrition_score / 100)
                    st.caption(f"Nutrition Score: {nutrition_score:.0f}/100")
                
                with col3:
                    st.markdown("### 🌙 Dinner")
                    dinner = day['meals']['dinner']
                    
                    st.markdown(f"**{dinner['description']}**")
                    
                    for food in dinner['foods']:
                        st.markdown(f"<span class='food-tag'>{food}</span>", unsafe_allow_html=True)
                    
                    st.markdown(f"💰 **Cost:** KES {dinner['estimated_cost']:.0f}")
                    st.markdown(f"⏱️ **Prep time:** {dinner['preparation_time']} min")
                    
                    # Nutrition score bar
                    nutrition_score = dinner['nutrition_score']
                    st.progress(nutrition_score / 100)
                    st.caption(f"Nutrition Score: {nutrition_score:.0f}/100")
                
                # Daily summary
                daily_cost = sum(meal['estimated_cost'] for meal in day['meals'].values())
                st.markdown(f"**Daily Total Cost:** KES {daily_cost:.0f}")
    
    with tab2:
        st.markdown("### 📊 Nutritional Analysis")
        
        # Weekly nutrition averages
        weekly_nutrition = nutrition_analysis['weekly_averages']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Daily Averages")
            
            # Display key nutrients
            key_nutrients = ['calories', 'protein', 'iron', 'calcium', 'vitamin_c']
            
            for nutrient in key_nutrients:
                if nutrient in weekly_nutrition:
                    value = weekly_nutrition[nutrient]
                    
                    # Define recommended daily values
                    recommended_values = {
                        'calories': 2000,
                        'protein': 50,
                        'iron': 18,
                        'calcium': 1000,
                        'vitamin_c': 90
                    }
                    
                    recommended = recommended_values.get(nutrient, 100)
                    percentage = (value / recommended) * 100
                    
                    st.metric(
                        nutrient.replace('_', ' ').title(),
                        f"{value:.1f}",
                        f"{percentage:.0f}% of RDA"
                    )
        
        with col2:
            st.markdown("#### Weekly Nutrition Chart")
            
            # Create nutrition chart
            nutrition_data = []
            for day_data in nutrition_analysis['daily_nutrition']:
                nutrition_data.append({
                    'Day': day_data['day'],
                    'Calories': day_data['nutrition'].get('calories', 0),
                    'Protein': day_data['nutrition'].get('protein', 0),
                    'Iron': day_data['nutrition'].get('iron', 0)
                })
            
            df_nutrition = pd.DataFrame(nutrition_data)
            
            if not df_nutrition.empty:
                fig = px.line(df_nutrition, x='Day', y=['Calories', 'Protein', 'Iron'], 
                            title="Daily Nutrition Trends")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 💰 Cost Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Weekly Cost Breakdown")
            
            # Display cost metrics
            st.metric("Total Weekly Cost", f"KES {cost_breakdown['total_weekly_cost']:.0f}")
            st.metric("Average Daily Cost", f"KES {cost_breakdown['total_weekly_cost']/7:.0f}")
            
            # Meal type costs
            st.markdown("**Cost by Meal Type:**")
            for meal_type, cost in cost_breakdown['meal_type_costs'].items():
                st.markdown(f"- {meal_type.title()}: KES {cost:.0f}")
        
        with col2:
            st.markdown("#### Cost Distribution")
            
            # Create cost pie chart
            if cost_breakdown['food_category_costs']:
                fig = px.pie(
                    values=list(cost_breakdown['food_category_costs'].values()),
                    names=list(cost_breakdown['food_category_costs'].keys()),
                    title="Cost by Food Category"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### 🛒 Shopping List")
        
        # Generate shopping list by category
        shopping_categories = {}
        for item in shopping_list:
            category = item['category']
            if category not in shopping_categories:
                shopping_categories[category] = []
            shopping_categories[category].append(item)
        
        # Display shopping list
        for category, items in shopping_categories.items():
            st.markdown(f"#### {category.title()}")
            
            total_category_cost = sum(item['estimated_cost'] for item in items)
            st.caption(f"Category Total: KES {total_category_cost:.0f}")
            
            for item in items:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**{item['item']}**")
                
                with col2:
                    st.markdown(f"Qty: {item['quantity']}")
                
                with col3:
                    st.markdown(f"KES {item['estimated_cost']:.0f}")
            
            st.markdown("---")
        
        # Download options
        st.subheader("📥 Download Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Download as PDF"):
                st.info("PDF download feature coming soon!")
        
        with col2:
            if st.button("📊 Download as CSV"):
                st.info("CSV download feature coming soon!")

if __name__ == "__main__":
    main()
