# 🤖 NutriAI East Africa - AI Components Demo
# This script demonstrates the AI capabilities of the system

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import DataProcessor
from nlp_processor import NLPProcessor
from ai_predictors import SupplyChainPredictor, NutritionDeficiencyPredictor, EconomicImpactModeler, CulturalFoodMapper
from meal_planner import MealPlanningEngine
import pandas as pd

def demo_nlp_processing():
    """Demonstrate NLP processing capabilities"""
    print("🗣️ NLP Processing Demo")
    print("=" * 50)
    
    nlp_processor = NLPProcessor()
    
    # Test various user inputs
    test_inputs = [
        "I have maize, beans, and sukuma wiki. My family likes ugali and we need high-protein meals.",
        "Mimi nina mahindi, maharage, na sukuma wiki. Familia yangu inapenda ugali.",
        "We are vegetarian and need meals under 500 KES per week.",
        "My wife is pregnant and we need iron-rich foods.",
        "Nina watoto wadogo, nahitaji chakula chenye vitamini nyingi."
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n📝 Test {i}: {user_input}")
        result = nlp_processor.process_user_input(user_input)
        
        print(f"   🍽️ Foods mentioned: {result['foods_mentioned']}")
        print(f"   🚫 Dietary restrictions: {result['dietary_restrictions']}")
        print(f"   👨‍👩‍👧‍👦 Family info: {result['family_info']}")
        print(f"   💰 Budget: {result.get('budget', 'Not specified')}")
        print(f"   😊 Sentiment: {result['sentiment']:.2f}")

def demo_supply_chain_prediction():
    """Demonstrate supply chain prediction"""
    print("\n📊 Supply Chain Prediction Demo")
    print("=" * 50)
    
    # Initialize data processor and supply chain predictor
    data_processor = DataProcessor()
    supply_predictor = SupplyChainPredictor(data_processor.price_data)
    
    countries = ['Kenya', 'Uganda', 'Tanzania', 'Ethiopia']
    
    for country in countries:
        print(f"\n🌍 {country} - Food Availability Forecast:")
        
        # Get availability predictions
        predictions = supply_predictor.predict_food_availability(country, months_ahead=3)
        
        for month, data in predictions.items():
            month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            availability = data['availability_score']
            price_trend = data['price_trend']
            
            trend_icons = {'low': '📉', 'medium': '📊', 'high': '📈'}
            icon = trend_icons.get(price_trend, '📊')
            
            print(f"   {month_names[month]}: {availability:.0f}% available, {icon} {price_trend} prices")
        
        # Get best shopping recommendations
        shopping_advice = supply_predictor.get_best_shopping_days(country)
        print(f"   🛒 Best months to shop: {shopping_advice['best_months']}")
        print(f"   ⚠️  Avoid months: {shopping_advice['avoid_months']}")

def demo_nutrition_deficiency_prediction():
    """Demonstrate nutrition deficiency prediction"""
    print("\n🏥 Nutrition Deficiency Prediction Demo")
    print("=" * 50)
    
    # Initialize components
    data_processor = DataProcessor()
    nutrition_predictor = NutritionDeficiencyPredictor(data_processor.nutrition_data)
    
    # Create sample meal plans for different scenarios
    sample_meal_plans = {
        'balanced': [
            {'foods': ['ugali', 'beans', 'sukuma wiki']},
            {'foods': ['rice', 'chicken', 'spinach']},
            {'foods': ['sweet potato', 'groundnuts', 'cabbage']}
        ],
        'poor_nutrition': [
            {'foods': ['ugali', 'ugali', 'ugali']},
            {'foods': ['rice', 'rice', 'rice']},
            {'foods': ['bread', 'bread', 'bread']}
        ],
        'vegetarian': [
            {'foods': ['ugali', 'beans', 'spinach']},
            {'foods': ['rice', 'groundnuts', 'cabbage']},
            {'foods': ['sweet potato', 'beans', 'sukuma wiki']}
        ]
    }
    
    user_profiles = {
        'balanced': {'dietary_restrictions': []},
        'poor_nutrition': {'dietary_restrictions': []},
        'vegetarian': {'dietary_restrictions': ['vegetarian'], 'family_info': {'pregnant': True}}
    }
    
    for scenario, meal_plan in sample_meal_plans.items():
        print(f"\n📋 Scenario: {scenario.title()}")
        
        # Predict deficiency risks
        risks = nutrition_predictor.predict_deficiency_risk(meal_plan, user_profiles[scenario])
        
        print("   🔍 Deficiency Risk Assessment:")
        for nutrient, risk_level in risks.items():
            risk_icons = {'low': '✅', 'medium': '⚠️', 'high': '🚨'}
            icon = risk_icons.get(risk_level, '❓')
            
            print(f"     {icon} {nutrient.title()}: {risk_level.upper()} risk")
        
        # Get improvement suggestions
        suggestions = nutrition_predictor.suggest_nutrition_improvements(risks)
        
        if suggestions:
            print("   💡 Improvement Suggestions:")
            for nutrient, suggestion in suggestions.items():
                if suggestion['risk_level'] in ['high', 'medium']:
                    foods = ', '.join(suggestion['recommended_foods'][:3])
                    print(f"     • {nutrient.title()}: Add {foods}")

def demo_economic_impact_modeling():
    """Demonstrate economic impact modeling"""
    print("\n💰 Economic Impact Modeling Demo")
    print("=" * 50)
    
    economic_modeler = EconomicImpactModeler()
    
    # Sample meal plans for comparison
    original_plan = [
        {'foods': ['beef', 'rice', 'expensive_vegetables']},
        {'foods': ['chicken', 'bread', 'imported_fruits']},
        {'foods': ['fish', 'pasta', 'processed_foods']}
    ]
    
    optimized_plan = [
        {'foods': ['beans', 'ugali', 'sukuma wiki']},
        {'foods': ['eggs', 'sweet potato', 'spinach']},
        {'foods': ['groundnuts', 'cassava', 'cabbage']}
    ]
    
    # Calculate cost savings
    cost_analysis = economic_modeler.calculate_cost_savings(original_plan, optimized_plan)
    
    print("📊 Cost Analysis:")
    print(f"   Original weekly cost: KES {cost_analysis['original_cost']:.0f}")
    print(f"   Optimized weekly cost: KES {cost_analysis['optimized_cost']:.0f}")
    print(f"   💰 Savings: KES {cost_analysis['savings']:.0f} ({cost_analysis['savings_percentage']:.1f}%)")
    
    # Sample program effectiveness data
    sample_user_data = [
        {'cost_savings': 200, 'nutrition_improved': True},
        {'cost_savings': 150, 'nutrition_improved': True},
        {'cost_savings': 300, 'nutrition_improved': False},
        {'cost_savings': 250, 'nutrition_improved': True},
        {'cost_savings': 180, 'nutrition_improved': True}
    ]
    
    effectiveness = economic_modeler.measure_program_effectiveness(sample_user_data)
    
    print("\n📈 Program Effectiveness:")
    print(f"   👥 Total users served: {effectiveness['total_users_served']}")
    print(f"   💰 Total cost savings: KES {effectiveness['total_cost_savings']:.0f}")
    print(f"   📊 Average savings per user: KES {effectiveness['average_savings_per_user']:.0f}")
    print(f"   🎯 Nutrition improvement rate: {effectiveness['nutrition_improvement_rate']:.1f}%")
    print(f"   🌍 Estimated people reached: {effectiveness['estimated_people_reached']}")

def demo_cultural_food_mapping():
    """Demonstrate cultural food mapping"""
    print("\n🌍 Cultural Food Mapping Demo")
    print("=" * 50)
    
    cultural_mapper = CulturalFoodMapper()
    
    countries = ['Kenya', 'Uganda', 'Tanzania', 'Ethiopia']
    
    for country in countries:
        print(f"\n🇰🇪 {country} Traditional Foods:")
        
        # Get traditional meal
        traditional_meal = cultural_mapper.get_culturally_appropriate_meal(
            country, 'lunch', ['vegetarian']
        )
        
        print(f"   🍽️ Traditional Lunch:")
        print(f"     • Staple: {traditional_meal['staple']}")
        print(f"     • Protein: {traditional_meal['protein']}")
        print(f"     • Vegetable: {traditional_meal['vegetable']}")
        
        # Get traditional alternatives
        modern_foods = ['bread', 'pasta', 'processed_meat', 'soda']
        
        print(f"   🔄 Traditional Alternatives:")
        for modern_food in modern_foods:
            alternatives = cultural_mapper.get_traditional_alternatives(modern_food, country)
            print(f"     • {modern_food} → {', '.join(alternatives)}")

def demo_meal_planning_engine():
    """Demonstrate the complete meal planning engine"""
    print("\n🍽️ Meal Planning Engine Demo")
    print("=" * 50)
    
    # Initialize all components
    data_processor = DataProcessor()
    nlp_processor = NLPProcessor()
    supply_predictor = SupplyChainPredictor(data_processor.price_data)
    nutrition_predictor = NutritionDeficiencyPredictor(data_processor.nutrition_data)
    
    meal_planner = MealPlanningEngine(
        data_processor, nlp_processor, supply_predictor, nutrition_predictor
    )
    
    # Sample user preferences
    user_preferences = {
        'country': 'Kenya',
        'budget': 800,
        'family_size': 4,
        'dietary_restrictions': ['vegetarian'],
        'special_needs': ['children'],
        'input_text': 'We need nutritious meals for our family with children. We prefer ugali and beans.'
    }
    
    print("👨‍👩‍👧‍👦 User Profile:")
    print(f"   🌍 Country: {user_preferences['country']}")
    print(f"   💰 Budget: KES {user_preferences['budget']}/week")
    print(f"   👥 Family size: {user_preferences['family_size']}")
    print(f"   🚫 Restrictions: {user_preferences['dietary_restrictions']}")
    print(f"   📝 Input: {user_preferences['input_text']}")
    
    # Generate meal plan
    print("\n🤖 Generating AI-optimized meal plan...")
    meal_plan_result = meal_planner.generate_meal_plan(user_preferences, days=3)
    
    print("\n📋 Generated Meal Plan (3 days):")
    
    for day in meal_plan_result['meal_plan']:
        print(f"\n📅 Day {day['day']} ({day['date']}):")
        
        daily_cost = 0
        for meal_type, meal in day['meals'].items():
            daily_cost += meal['estimated_cost']
            
            print(f"   🍽️ {meal_type.title()}: {meal['description']}")
            print(f"     Foods: {', '.join(meal['foods'])}")
            print(f"     Cost: KES {meal['estimated_cost']:.0f}")
            print(f"     Nutrition Score: {meal['nutrition_score']:.0f}/100")
        
        print(f"   💰 Daily Total: KES {daily_cost:.0f}")
    
    # Display cost breakdown
    cost_breakdown = meal_plan_result['cost_breakdown']
    print(f"\n💰 Cost Summary:")
    print(f"   Total Cost: KES {cost_breakdown['total_weekly_cost']:.0f}")
    print(f"   Budget: KES {user_preferences['budget']}")
    
    savings = user_preferences['budget'] - cost_breakdown['total_weekly_cost']
    if savings > 0:
        print(f"   ✅ Savings: KES {savings:.0f}")
    else:
        print(f"   ⚠️ Over budget: KES {abs(savings):.0f}")
    
    # Display shopping list
    shopping_list = meal_plan_result['shopping_list']
    print(f"\n🛒 Shopping List:")
    
    for item in shopping_list[:10]:  # Show first 10 items
        print(f"   • {item['item']} (x{item['quantity']}) - KES {item['estimated_cost']:.0f}")

def main():
    """Main demo function"""
    print("🌍 NutriAI East Africa - AI Components Demo")
    print("=" * 60)
    print("This demo showcases the AI capabilities of the nutrition planning system")
    print("=" * 60)
    
    # Run all demos
    demo_nlp_processing()
    demo_supply_chain_prediction()
    demo_nutrition_deficiency_prediction()
    demo_economic_impact_modeling()
    demo_cultural_food_mapping()
    demo_meal_planning_engine()
    
    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    print("🚀 The system is ready for production use.")
    print("🌍 Together, we can achieve Zero Hunger in East Africa!")
    print("=" * 60)

if __name__ == "__main__":
    main()
