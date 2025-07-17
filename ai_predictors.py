# AI Prediction Components for NutriAI East Africa
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SupplyChainPredictor:
    """Predict food availability and prices using time-series analysis"""
    
    def __init__(self, price_data):
        self.price_data = price_data
        self.seasonal_patterns = {}
        self.price_predictions = {}
        self.analyze_seasonal_patterns()
    
    def analyze_seasonal_patterns(self):
        """Analyze seasonal patterns in food prices"""
        if self.price_data is not None and not self.price_data.empty:
            # Convert date column
            self.price_data['date'] = pd.to_datetime(self.price_data['date'])
            self.price_data['month'] = self.price_data['date'].dt.month
            self.price_data['year'] = self.price_data['date'].dt.year
            
            # Analyze patterns by country
            for country in self.price_data['country'].unique():
                country_data = self.price_data[self.price_data['country'] == country]
                
                # Calculate seasonal patterns
                monthly_stats = country_data.groupby('month')['Close'].agg(['mean', 'std', 'min', 'max'])
                
                # Calculate availability score (inverse of price)
                monthly_stats['availability_score'] = 100 - (monthly_stats['mean'] / monthly_stats['mean'].max() * 100)
                
                # Identify best months to buy
                monthly_stats['best_month'] = monthly_stats['availability_score'] > monthly_stats['availability_score'].mean()
                
                self.seasonal_patterns[country] = monthly_stats
    
    def predict_food_availability(self, country='Kenya', months_ahead=3):
        """Predict food availability for the next few months"""
        if country not in self.seasonal_patterns:
            return self.get_default_availability()
        
        current_month = datetime.now().month
        predictions = {}
        
        for i in range(months_ahead):
            future_month = ((current_month + i - 1) % 12) + 1
            
            if country in self.seasonal_patterns:
                pattern = self.seasonal_patterns[country]
                if future_month in pattern.index:
                    availability = pattern.loc[future_month, 'availability_score']
                    predictions[future_month] = {
                        'month': future_month,
                        'availability_score': availability,
                        'price_trend': 'low' if availability > 60 else 'high' if availability < 40 else 'medium'
                    }
        
        return predictions
    
    def get_default_availability(self):
        """Get default availability patterns for East Africa"""
        # Based on typical harvest seasons
        default_patterns = {
            1: {'availability_score': 70, 'price_trend': 'medium'},  # January
            2: {'availability_score': 65, 'price_trend': 'medium'},  # February
            3: {'availability_score': 60, 'price_trend': 'medium'},  # March
            4: {'availability_score': 55, 'price_trend': 'high'},    # April (pre-harvest)
            5: {'availability_score': 50, 'price_trend': 'high'},    # May
            6: {'availability_score': 45, 'price_trend': 'high'},    # June
            7: {'availability_score': 80, 'price_trend': 'low'},     # July (harvest)
            8: {'availability_score': 85, 'price_trend': 'low'},     # August
            9: {'availability_score': 75, 'price_trend': 'low'},     # September
            10: {'availability_score': 70, 'price_trend': 'medium'}, # October
            11: {'availability_score': 75, 'price_trend': 'medium'}, # November
            12: {'availability_score': 80, 'price_trend': 'low'}     # December
        }
        
        return default_patterns
    
    def get_best_shopping_days(self, country='Kenya'):
        """Predict best days for shopping based on price patterns"""
        if country in self.seasonal_patterns:
            pattern = self.seasonal_patterns[country]
            best_months = pattern[pattern['best_month']].index.tolist()
            
            return {
                'best_months': best_months,
                'avoid_months': pattern[~pattern['best_month']].index.tolist(),
                'peak_season': pattern['availability_score'].idxmax(),
                'lean_season': pattern['availability_score'].idxmin()
            }
        
        return {
            'best_months': [7, 8, 9, 12],  # Harvest seasons
            'avoid_months': [4, 5, 6],      # Pre-harvest
            'peak_season': 8,
            'lean_season': 5
        }

class NutritionDeficiencyPredictor:
    """Predict nutritional deficiencies and at-risk populations"""
    
    def __init__(self, nutrition_data):
        self.nutrition_data = nutrition_data
        self.deficiency_patterns = {}
        self.risk_thresholds = self.setup_risk_thresholds()
    
    def setup_risk_thresholds(self):
        """Setup risk thresholds for different nutrients"""
        return {
            'protein': {'low': 10, 'medium': 15, 'high': 20},       # grams per day
            'iron': {'low': 5, 'medium': 10, 'high': 15},           # mg per day
            'calcium': {'low': 400, 'medium': 600, 'high': 800},    # mg per day
            'vitamin_c': {'low': 30, 'medium': 50, 'high': 70},     # mg per day
            'calories': {'low': 1500, 'medium': 2000, 'high': 2500} # kcal per day
        }
    
    def analyze_meal_plan_nutrition(self, meal_plan):
        """Analyze nutritional content of a meal plan"""
        if not meal_plan:
            return {}
        
        total_nutrition = {
            'calories': 0,
            'protein': 0,
            'iron': 0,
            'calcium': 0,
            'vitamin_c': 0,
            'fat': 0,
            'carbohydrates': 0
        }
        
        for meal in meal_plan:
            for food_item in meal.get('foods', []):
                nutrition = self.get_food_nutrition(food_item)
                for nutrient in total_nutrition:
                    total_nutrition[nutrient] += nutrition.get(nutrient, 0)
        
        return total_nutrition
    
    def get_food_nutrition(self, food_item):
        """Get nutrition information for a specific food item"""
        if self.nutrition_data is not None:
            # Try to find the food in our database using Description column
            matches = self.nutrition_data[
                self.nutrition_data['Description'].str.contains(food_item, case=False, na=False)
            ]
            
            if not matches.empty:
                item = matches.iloc[0]
                return {
                    'calories': item.get('Data.Kilocalories', 0),
                    'protein': item.get('Data.Protein', 0),
                    'iron': item.get('Data.Major Minerals.Iron', 0),
                    'calcium': item.get('Data.Major Minerals.Calcium', 0),
                    'vitamin_c': item.get('Data.Vitamins.Vitamin C', 0),
                    'fat': item.get('Data.Fat.Total Lipid', 0),
                    'carbohydrates': item.get('Data.Carbohydrate', 0)
                }
        
        # Default nutrition values for common East African foods
        default_nutrition = {
            'ugali': {'calories': 378, 'protein': 8.1, 'iron': 1.2, 'calcium': 6, 'vitamin_c': 0, 'fat': 1.4, 'carbohydrates': 84},
            'beans': {'calories': 347, 'protein': 21.6, 'iron': 8.2, 'calcium': 143, 'vitamin_c': 0, 'fat': 1.2, 'carbohydrates': 63},
            'sukuma wiki': {'calories': 22, 'protein': 2.8, 'iron': 2.7, 'calcium': 212, 'vitamin_c': 60, 'fat': 0.3, 'carbohydrates': 4.3},
            'rice': {'calories': 365, 'protein': 7.1, 'iron': 0.8, 'calcium': 28, 'vitamin_c': 0, 'fat': 0.7, 'carbohydrates': 80},
            'meat': {'calories': 250, 'protein': 20, 'iron': 2.5, 'calcium': 10, 'vitamin_c': 0, 'fat': 15, 'carbohydrates': 0},
            'milk': {'calories': 61, 'protein': 3.2, 'iron': 0.03, 'calcium': 113, 'vitamin_c': 0, 'fat': 3.3, 'carbohydrates': 4.7},
            'eggs': {'calories': 155, 'protein': 13, 'iron': 1.2, 'calcium': 50, 'vitamin_c': 0, 'fat': 11, 'carbohydrates': 1.1}
        }
        
        return default_nutrition.get(food_item.lower(), {'calories': 100, 'protein': 5, 'iron': 1, 'calcium': 20, 'vitamin_c': 10, 'fat': 2, 'carbohydrates': 15})
    
    def predict_deficiency_risk(self, meal_plan, user_profile):
        """Predict risk of nutritional deficiencies"""
        nutrition_totals = self.analyze_meal_plan_nutrition(meal_plan)
        
        risks = {}
        
        # Adjust thresholds based on user profile
        adjusted_thresholds = self.risk_thresholds.copy()
        
        if user_profile.get('family_info', {}).get('pregnant'):
            adjusted_thresholds['iron']['medium'] += 10
            adjusted_thresholds['calcium']['medium'] += 200
            adjusted_thresholds['protein']['medium'] += 5
        
        if user_profile.get('family_info', {}).get('has_children'):
            adjusted_thresholds['calcium']['medium'] += 100
            adjusted_thresholds['vitamin_c']['medium'] += 20
        
        # Calculate risk for each nutrient
        for nutrient, thresholds in adjusted_thresholds.items():
            daily_intake = nutrition_totals.get(nutrient, 0)
            
            if daily_intake < thresholds['low']:
                risks[nutrient] = 'high'
            elif daily_intake < thresholds['medium']:
                risks[nutrient] = 'medium'
            else:
                risks[nutrient] = 'low'
        
        return risks
    
    def identify_at_risk_populations(self, user_profiles):
        """Identify populations at risk of nutritional deficiencies"""
        risk_groups = {
            'pregnant_women': [],
            'children': [],
            'elderly': [],
            'low_income': [],
            'vegetarians': []
        }
        
        for profile in user_profiles:
            if profile.get('family_info', {}).get('pregnant'):
                risk_groups['pregnant_women'].append(profile)
            
            if profile.get('family_info', {}).get('has_children'):
                risk_groups['children'].append(profile)
            
            if 'elderly' in profile.get('dietary_restrictions', []):
                risk_groups['elderly'].append(profile)
            
            if profile.get('budget', 0) < 1000:  # Low income threshold
                risk_groups['low_income'].append(profile)
            
            if 'vegetarian' in profile.get('dietary_restrictions', []):
                risk_groups['vegetarians'].append(profile)
        
        return risk_groups
    
    def suggest_nutrition_improvements(self, deficiency_risks):
        """Suggest foods to improve nutritional deficiencies"""
        suggestions = {}
        
        high_nutrient_foods = {
            'protein': ['beans', 'eggs', 'meat', 'fish', 'groundnuts'],
            'iron': ['spinach', 'beans', 'meat', 'sukuma wiki'],
            'calcium': ['milk', 'sukuma wiki', 'sesame seeds', 'fish'],
            'vitamin_c': ['oranges', 'tomatoes', 'sukuma wiki', 'cabbage'],
            'calories': ['ugali', 'rice', 'sweet potato', 'groundnuts']
        }
        
        for nutrient, risk_level in deficiency_risks.items():
            if risk_level in ['high', 'medium']:
                suggestions[nutrient] = {
                    'risk_level': risk_level,
                    'recommended_foods': high_nutrient_foods.get(nutrient, []),
                    'daily_target': self.risk_thresholds[nutrient]['high']
                }
        
        return suggestions

class EconomicImpactModeler:
    """Model economic impact of nutrition interventions"""
    
    def __init__(self):
        self.cost_savings = {}
        self.nutrition_improvements = {}
        self.community_metrics = {}
    
    def calculate_cost_savings(self, original_plan, optimized_plan):
        """Calculate cost savings from optimized meal planning"""
        original_cost = self.calculate_meal_plan_cost(original_plan)
        optimized_cost = self.calculate_meal_plan_cost(optimized_plan)
        
        savings = original_cost - optimized_cost
        savings_percentage = (savings / original_cost) * 100 if original_cost > 0 else 0
        
        return {
            'original_cost': original_cost,
            'optimized_cost': optimized_cost,
            'savings': savings,
            'savings_percentage': savings_percentage
        }
    
    def calculate_meal_plan_cost(self, meal_plan):
        """Calculate the cost of a meal plan"""
        if not meal_plan:
            return 0
        
        # Default costs in KES for common foods
        food_costs = {
            'ugali': 30,      # per serving
            'rice': 40,       # per serving
            'beans': 50,      # per serving
            'meat': 150,      # per serving
            'chicken': 120,   # per serving
            'fish': 100,      # per serving
            'sukuma wiki': 20, # per serving
            'spinach': 25,    # per serving
            'eggs': 15,       # per egg
            'milk': 50,       # per glass
            'bread': 60,      # per loaf
            'sweet potato': 30, # per serving
            'groundnuts': 80   # per serving
        }
        
        total_cost = 0
        
        for meal in meal_plan:
            for food_item in meal.get('foods', []):
                cost = food_costs.get(food_item.lower(), 50)  # Default cost
                total_cost += cost
        
        return total_cost
    
    def calculate_nutrition_improvement(self, before_nutrition, after_nutrition):
        """Calculate improvement in nutritional quality"""
        improvements = {}
        
        for nutrient in before_nutrition:
            before_value = before_nutrition[nutrient]
            after_value = after_nutrition[nutrient]
            
            if before_value > 0:
                improvement = ((after_value - before_value) / before_value) * 100
                improvements[nutrient] = improvement
            else:
                improvements[nutrient] = 100 if after_value > 0 else 0
        
        return improvements
    
    def measure_program_effectiveness(self, user_data):
        """Measure overall program effectiveness"""
        total_users = len(user_data)
        
        if total_users == 0:
            return {}
        
        total_savings = sum(user.get('cost_savings', 0) for user in user_data)
        users_with_improvements = sum(1 for user in user_data if user.get('nutrition_improved', False))
        
        average_savings = total_savings / total_users
        improvement_rate = (users_with_improvements / total_users) * 100
        
        return {
            'total_users_served': total_users,
            'total_cost_savings': total_savings,
            'average_savings_per_user': average_savings,
            'nutrition_improvement_rate': improvement_rate,
            'estimated_people_reached': total_users * 4  # Assuming 4 people per family
        }
    
    def calculate_roi(self, program_cost, benefits):
        """Calculate return on investment for nutrition programs"""
        if program_cost <= 0:
            return 0
        
        # Calculate total benefits
        total_benefits = (
            benefits.get('health_savings', 0) +
            benefits.get('productivity_gains', 0) +
            benefits.get('education_improvements', 0)
        )
        
        roi = ((total_benefits - program_cost) / program_cost) * 100
        
        return {
            'program_cost': program_cost,
            'total_benefits': total_benefits,
            'roi_percentage': roi,
            'payback_period': program_cost / (total_benefits / 12) if total_benefits > 0 else float('inf')
        }

class CulturalFoodMapper:
    """Map and preserve traditional nutrition knowledge"""
    
    def __init__(self):
        self.traditional_foods = {}
        self.cultural_practices = {}
        self.seasonal_foods = {}
        self.setup_cultural_mappings()
    
    def setup_cultural_mappings(self):
        """Setup mappings for traditional East African foods"""
        self.traditional_foods = {
            'Kenya': {
                'staples': ['ugali', 'rice', 'chapati', 'sukuma wiki'],
                'proteins': ['nyama choma', 'fish', 'eggs', 'beans'],
                'vegetables': ['sukuma wiki', 'spinach', 'cabbage', 'tomatoes'],
                'traditional_dishes': ['githeri', 'mukimo', 'nyama choma', 'pilau']
            },
            'Uganda': {
                'staples': ['posho', 'matoke', 'sweet potato', 'cassava'],
                'proteins': ['fish', 'chicken', 'beans', 'groundnuts'],
                'vegetables': ['dodo', 'nakati', 'jobyo'],
                'traditional_dishes': ['luwombo', 'malewa', 'eshabwe']
            },
            'Tanzania': {
                'staples': ['ugali', 'rice', 'mtindi', 'ndizi'],
                'proteins': ['nyama', 'samaki', 'maharage'],
                'vegetables': ['mboga', 'mchicha', 'kunde'],
                'traditional_dishes': ['pilau', 'wali wa nazi', 'mchuzi wa samaki']
            },
            'Ethiopia': {
                'staples': ['injera', 'rice', 'bread'],
                'proteins': ['doro', 'kitfo', 'shiro', 'fish'],
                'vegetables': ['gomen', 'misir', 'alicha'],
                'traditional_dishes': ['doro wat', 'kitfo', 'shiro wat', 'gomen']
            }
        }
        
        self.cultural_practices = {
            'fasting_foods': ['shiro', 'vegetables', 'lentils', 'beans'],
            'celebration_foods': ['doro wat', 'nyama choma', 'pilau', 'rice'],
            'child_foods': ['porridge', 'mashed banana', 'soft ugali', 'milk'],
            'pregnancy_foods': ['dark leafy greens', 'eggs', 'milk', 'meat'],
            'elderly_foods': ['soft ugali', 'porridge', 'soup', 'well-cooked vegetables']
        }
    
    def get_traditional_alternatives(self, modern_food, country='Kenya'):
        """Get traditional alternatives for modern foods"""
        alternatives = {
            'bread': ['chapati', 'mandazi', 'injera'],
            'pasta': ['ugali', 'rice', 'sweet potato'],
            'processed_meat': ['nyama choma', 'boiled meat', 'fish'],
            'soda': ['traditional_porridge', 'fruit_juice', 'tea'],
            'snacks': ['roasted_groundnuts', 'boiled_sweet_potato', 'fruits']
        }
        
        return alternatives.get(modern_food.lower(), [modern_food])
    
    def preserve_nutrition_knowledge(self, food_item, nutrition_info, cultural_context):
        """Preserve traditional nutrition knowledge"""
        knowledge_entry = {
            'food_name': food_item,
            'nutrition_info': nutrition_info,
            'cultural_context': cultural_context,
            'traditional_preparation': cultural_context.get('preparation', ''),
            'seasonal_availability': cultural_context.get('season', ''),
            'cultural_significance': cultural_context.get('significance', ''),
            'health_beliefs': cultural_context.get('health_beliefs', '')
        }
        
        return knowledge_entry
    
    def get_culturally_appropriate_meal(self, country, meal_type, dietary_restrictions):
        """Generate culturally appropriate meal suggestions"""
        if country not in self.traditional_foods:
            country = 'Kenya'  # Default
        
        country_foods = self.traditional_foods[country]
        
        # Build meal based on cultural patterns
        meal = {
            'staple': np.random.choice(country_foods['staples']),
            'protein': np.random.choice(country_foods['proteins']),
            'vegetable': np.random.choice(country_foods['vegetables'])
        }
        
        # Adjust for dietary restrictions
        if 'vegetarian' in dietary_restrictions:
            vegetarian_proteins = ['beans', 'groundnuts', 'eggs']
            meal['protein'] = np.random.choice(vegetarian_proteins)
        
        if 'no_dairy' in dietary_restrictions:
            # Remove dairy-based items
            pass
        
        return meal
