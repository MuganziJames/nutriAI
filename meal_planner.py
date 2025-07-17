# Meal Planning Engine for NutriAI East Africa
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from collections import defaultdict

class MealPlanningEngine:
    """Core engine for generating optimized meal plans"""
    
    def __init__(self, data_processor, nlp_processor, supply_predictor, nutrition_predictor):
        self.data_processor = data_processor
        self.nlp_processor = nlp_processor
        self.supply_predictor = supply_predictor
        self.nutrition_predictor = nutrition_predictor
        self.meal_templates = self.setup_meal_templates()
        self.optimization_weights = self.setup_optimization_weights()
    
    def setup_meal_templates(self):
        """Setup meal templates for different meal types"""
        return {
            'breakfast': {
                'structure': ['grain/starch', 'protein', 'beverage'],
                'common_combinations': [
                    ['ugali', 'eggs', 'chai'],
                    ['bread', 'milk', 'tea'],
                    ['porridge', 'milk', 'tea'],
                    ['sweet potato', 'groundnuts', 'tea']
                ],
                'nutritional_focus': ['energy', 'protein'],
                'typical_cost_range': [50, 150]
            },
            'lunch': {
                'structure': ['staple', 'protein', 'vegetable', 'fat'],
                'common_combinations': [
                    ['ugali', 'beans', 'sukuma wiki', 'oil'],
                    ['rice', 'meat', 'vegetables', 'oil'],
                    ['ugali', 'fish', 'spinach', 'oil'],
                    ['sweet potato', 'groundnuts', 'cabbage', 'oil']
                ],
                'nutritional_focus': ['protein', 'vitamins', 'minerals'],
                'typical_cost_range': [100, 300]
            },
            'dinner': {
                'structure': ['staple', 'protein', 'vegetable'],
                'common_combinations': [
                    ['rice', 'beef stew', 'vegetables'],
                    ['ugali', 'chicken', 'sukuma wiki'],
                    ['chapati', 'beans', 'cabbage'],
                    ['cassava', 'fish', 'spinach']
                ],
                'nutritional_focus': ['balanced', 'digestible'],
                'typical_cost_range': [120, 280]
            }
        }
    
    def setup_optimization_weights(self):
        """Setup weights for meal plan optimization"""
        return {
            'cost': 0.3,           # 30% weight on cost optimization
            'nutrition': 0.4,      # 40% weight on nutritional quality
            'availability': 0.2,   # 20% weight on food availability
            'cultural_fit': 0.1    # 10% weight on cultural appropriateness
        }
    
    def generate_meal_plan(self, user_preferences, days=7):
        """Generate a comprehensive meal plan"""
        # Process user preferences
        processed_prefs = self.nlp_processor.process_user_input(user_preferences.get('input_text', ''))
        
        # Combine with structured preferences
        combined_prefs = {**processed_prefs, **user_preferences}
        
        # Generate meal plan
        meal_plan = []
        
        for day in range(days):
            day_plan = {
                'day': day + 1,
                'date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
                'meals': {}
            }
            
            # Generate meals for each meal type
            for meal_type in ['breakfast', 'lunch', 'dinner']:
                meal = self.generate_single_meal(meal_type, combined_prefs, day)
                day_plan['meals'][meal_type] = meal
            
            meal_plan.append(day_plan)
        
        # Optimize the meal plan
        optimized_plan = self.optimize_meal_plan(meal_plan, combined_prefs)
        
        # Add nutritional analysis
        nutrition_analysis = self.analyze_meal_plan_nutrition(optimized_plan)
        
        return {
            'meal_plan': optimized_plan,
            'nutrition_analysis': nutrition_analysis,
            'user_preferences': combined_prefs,
            'cost_breakdown': self.calculate_cost_breakdown(optimized_plan),
            'shopping_list': self.generate_shopping_list(optimized_plan)
        }
    
    def generate_single_meal(self, meal_type, preferences, day):
        """Generate a single meal based on preferences"""
        template = self.meal_templates[meal_type]
        
        # Get available foods
        available_foods = self.get_available_foods(preferences.get('country', 'Kenya'))
        
        # Filter based on dietary restrictions
        filtered_foods = self.filter_foods_by_restrictions(available_foods, preferences.get('dietary_restrictions', []))
        
        # Select foods based on template structure
        selected_foods = []
        meal_cost = 0
        
        for component in template['structure']:
            suitable_foods = self.get_foods_by_component(filtered_foods, component)
            
            if suitable_foods:
                # Use AI to select best food based on multiple factors
                selected_food = self.select_optimal_food(suitable_foods, preferences, meal_type, day)
                selected_foods.append(selected_food)
                meal_cost += self.get_food_cost(selected_food)
        
        # Generate culturally appropriate description
        meal_description = self.nlp_processor.generate_meal_description(selected_foods, meal_type)
        
        return {
            'foods': selected_foods,
            'description': meal_description,
            'estimated_cost': meal_cost,
            'nutrition_score': self.calculate_nutrition_score(selected_foods),
            'cultural_score': self.calculate_cultural_score(selected_foods, preferences.get('country', 'Kenya')),
            'preparation_time': self.estimate_preparation_time(selected_foods)
        }
    
    def get_available_foods(self, country):
        """Get foods available in the specified country"""
        # Get foods from data processor
        east_african_foods = self.data_processor.get_east_african_foods()
        
        # Get seasonal availability
        seasonal_data = self.data_processor.get_seasonal_availability(country)
        
        # Combine with supply chain predictions
        availability_predictions = self.supply_predictor.predict_food_availability(country)
        
        # Create comprehensive food list
        available_foods = []
        
        # Add foods from our database
        if not east_african_foods.empty:
            for _, food in east_african_foods.iterrows():
                food_info = {
                    'name': food['food'],
                    'nutrition': food.to_dict(),
                    'availability_score': 70,  # Default availability
                    'seasonal_score': 60
                }
                available_foods.append(food_info)
        
        # Add common East African foods if database is empty
        if not available_foods:
            common_foods = [
                'ugali', 'rice', 'beans', 'sukuma wiki', 'sweet potato', 'cassava',
                'groundnuts', 'eggs', 'chicken', 'beef', 'fish', 'milk', 'spinach',
                'cabbage', 'tomatoes', 'onions', 'cooking oil', 'tea', 'bread'
            ]
            
            for food in common_foods:
                food_info = {
                    'name': food,
                    'nutrition': self.nutrition_predictor.get_food_nutrition(food),
                    'availability_score': 70,
                    'seasonal_score': 60
                }
                available_foods.append(food_info)
        
        return available_foods
    
    def filter_foods_by_restrictions(self, foods, restrictions):
        """Filter foods based on dietary restrictions"""
        filtered_foods = []
        
        for food in foods:
            food_name = food['name'].lower()
            
            # Check restrictions
            exclude_food = False
            
            if 'vegetarian' in restrictions:
                meat_keywords = ['meat', 'beef', 'chicken', 'fish', 'pork', 'mutton']
                if any(keyword in food_name for keyword in meat_keywords):
                    exclude_food = True
            
            if 'no_dairy' in restrictions:
                dairy_keywords = ['milk', 'cheese', 'yogurt', 'butter']
                if any(keyword in food_name for keyword in dairy_keywords):
                    exclude_food = True
            
            if 'gluten_free' in restrictions:
                gluten_keywords = ['wheat', 'bread', 'chapati', 'pasta']
                if any(keyword in food_name for keyword in gluten_keywords):
                    exclude_food = True
            
            if not exclude_food:
                filtered_foods.append(food)
        
        return filtered_foods
    
    def get_foods_by_component(self, foods, component):
        """Get foods that match a specific meal component"""
        component_mappings = {
            'grain/starch': ['ugali', 'rice', 'bread', 'chapati', 'sweet potato', 'cassava', 'porridge'],
            'staple': ['ugali', 'rice', 'sweet potato', 'cassava', 'bread', 'chapati'],
            'protein': ['beans', 'meat', 'chicken', 'fish', 'eggs', 'groundnuts', 'beef'],
            'vegetable': ['sukuma wiki', 'spinach', 'cabbage', 'tomatoes', 'onions', 'carrots'],
            'fat': ['cooking oil', 'groundnuts', 'avocado'],
            'beverage': ['tea', 'coffee', 'milk', 'water']
        }
        
        suitable_foods = []
        component_keywords = component_mappings.get(component, [])
        
        for food in foods:
            food_name = food['name'].lower()
            
            # Check if food matches component
            if any(keyword in food_name for keyword in component_keywords):
                suitable_foods.append(food)
        
        return suitable_foods
    
    def select_optimal_food(self, suitable_foods, preferences, meal_type, day):
        """Select optimal food using AI-based scoring"""
        if not suitable_foods:
            return 'ugali'  # Default fallback
        
        scores = []
        
        for food in suitable_foods:
            score = self.calculate_food_score(food, preferences, meal_type, day)
            scores.append((food['name'], score))
        
        # Sort by score and select best
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Add some randomness to avoid repetition
        if len(scores) > 1 and random.random() < 0.3:
            return scores[1][0]  # Sometimes select second best
        
        return scores[0][0]
    
    def calculate_food_score(self, food, preferences, meal_type, day):
        """Calculate optimization score for a food item"""
        # Cost score (lower cost = higher score)
        cost = self.get_food_cost(food['name'])
        budget = preferences.get('budget', 1000)
        cost_score = max(0, (budget - cost) / budget) * 100
        
        # Nutrition score
        nutrition_score = self.data_processor.get_nutrition_density_score(food['name'])
        
        # Availability score
        availability_score = food.get('availability_score', 70)
        
        # Cultural fit score
        cultural_score = self.calculate_cultural_score([food['name']], preferences.get('country', 'Kenya'))
        
        # Variety score (penalize repetition)
        variety_score = 100 - (day * 10)  # Slight penalty for later days to encourage variety
        
        # Weighted total score
        total_score = (
            cost_score * self.optimization_weights['cost'] +
            nutrition_score * self.optimization_weights['nutrition'] +
            availability_score * self.optimization_weights['availability'] +
            cultural_score * self.optimization_weights['cultural_fit'] +
            variety_score * 0.1
        )
        
        return total_score
    
    def get_food_cost(self, food_name):
        """Get estimated cost of a food item"""
        # Cost estimates in KES
        food_costs = {
            'ugali': 30, 'rice': 40, 'beans': 50, 'sukuma wiki': 20,
            'sweet potato': 25, 'cassava': 30, 'groundnuts': 80,
            'eggs': 15, 'chicken': 120, 'beef': 150, 'fish': 100,
            'milk': 50, 'spinach': 25, 'cabbage': 20, 'tomatoes': 30,
            'onions': 25, 'cooking oil': 60, 'tea': 10, 'bread': 60,
            'chapati': 20, 'porridge': 25, 'carrots': 30, 'avocado': 40
        }
        
        return food_costs.get(food_name.lower(), 50)
    
    def calculate_nutrition_score(self, foods):
        """Calculate overall nutrition score for a meal"""
        total_score = 0
        
        for food in foods:
            score = self.data_processor.get_nutrition_density_score(food)
            total_score += score
        
        return total_score / len(foods) if foods else 0
    
    def calculate_cultural_score(self, foods, country):
        """Calculate cultural appropriateness score"""
        cultural_foods = {
            'Kenya': ['ugali', 'sukuma wiki', 'beans', 'nyama choma', 'chai'],
            'Uganda': ['posho', 'matoke', 'groundnuts', 'fish'],
            'Tanzania': ['ugali', 'rice', 'samaki', 'mchicha'],
            'Ethiopia': ['injera', 'berbere', 'doro', 'shiro']
        }
        
        country_foods = cultural_foods.get(country, cultural_foods['Kenya'])
        
        score = 0
        for food in foods:
            if any(cultural_food in food.lower() for cultural_food in country_foods):
                score += 20
        
        return min(score, 100)
    
    def estimate_preparation_time(self, foods):
        """Estimate preparation time for a meal"""
        prep_times = {
            'ugali': 30, 'rice': 25, 'beans': 60, 'sukuma wiki': 15,
            'sweet potato': 20, 'cassava': 45, 'groundnuts': 10,
            'eggs': 10, 'chicken': 45, 'beef': 60, 'fish': 30,
            'milk': 5, 'spinach': 15, 'cabbage': 20, 'tomatoes': 10,
            'tea': 10, 'bread': 5, 'chapati': 20, 'porridge': 15
        }
        
        total_time = 0
        for food in foods:
            total_time += prep_times.get(food.lower(), 20)
        
        return max(total_time - 10, 15)  # Assume some parallel cooking
    
    def optimize_meal_plan(self, meal_plan, preferences):
        """Optimize the entire meal plan"""
        # Check budget constraints
        total_cost = sum(
            meal['estimated_cost'] 
            for day in meal_plan 
            for meal in day['meals'].values()
        )
        
        budget = preferences.get('budget', 1000) * 7  # Weekly budget
        
        if total_cost > budget:
            # Optimize for cost
            meal_plan = self.reduce_meal_plan_cost(meal_plan, budget)
        
        # Ensure nutritional balance
        meal_plan = self.balance_nutrition(meal_plan, preferences)
        
        # Add variety
        meal_plan = self.add_variety(meal_plan)
        
        return meal_plan
    
    def reduce_meal_plan_cost(self, meal_plan, budget):
        """Reduce meal plan cost while maintaining nutrition"""
        # Identify most expensive meals
        expensive_meals = []
        
        for day in meal_plan:
            for meal_type, meal in day['meals'].items():
                expensive_meals.append({
                    'day': day['day'],
                    'meal_type': meal_type,
                    'cost': meal['estimated_cost'],
                    'meal': meal
                })
        
        # Sort by cost and replace expensive items
        expensive_meals.sort(key=lambda x: x['cost'], reverse=True)
        
        # Replace with cheaper alternatives
        for expensive_meal in expensive_meals[:3]:  # Replace top 3 expensive meals
            cheaper_alternatives = self.get_cheaper_alternatives(expensive_meal['meal']['foods'])
            
            # Update meal plan
            day_idx = expensive_meal['day'] - 1
            meal_type = expensive_meal['meal_type']
            
            meal_plan[day_idx]['meals'][meal_type]['foods'] = cheaper_alternatives
            meal_plan[day_idx]['meals'][meal_type]['estimated_cost'] = sum(
                self.get_food_cost(food) for food in cheaper_alternatives
            )
        
        return meal_plan
    
    def get_cheaper_alternatives(self, foods):
        """Get cheaper alternatives for expensive foods"""
        alternatives = {
            'beef': 'chicken',
            'chicken': 'eggs',
            'fish': 'beans',
            'meat': 'beans',
            'milk': 'groundnuts',
            'cheese': 'beans',
            'bread': 'ugali',
            'rice': 'ugali'
        }
        
        cheaper_foods = []
        for food in foods:
            alternative = alternatives.get(food.lower(), food)
            cheaper_foods.append(alternative)
        
        return cheaper_foods
    
    def balance_nutrition(self, meal_plan, preferences):
        """Balance nutrition across the meal plan"""
        # Analyze current nutrition
        nutrition_totals = defaultdict(float)
        
        for day in meal_plan:
            for meal in day['meals'].values():
                for food in meal['foods']:
                    nutrition = self.nutrition_predictor.get_food_nutrition(food)
                    for nutrient, value in nutrition.items():
                        nutrition_totals[nutrient] += value
        
        # Check for deficiencies
        deficiency_risks = self.nutrition_predictor.predict_deficiency_risk(meal_plan, preferences)
        
        # Add foods to address deficiencies
        for nutrient, risk in deficiency_risks.items():
            if risk == 'high':
                # Add nutrient-rich foods
                suggestions = self.nutrition_predictor.suggest_nutrition_improvements({nutrient: risk})
                if nutrient in suggestions:
                    recommended_foods = suggestions[nutrient]['recommended_foods']
                    
                    # Add to meal plan
                    for day in meal_plan[:3]:  # Add to first 3 days
                        for meal_type in ['lunch', 'dinner']:
                            if recommended_foods:
                                day['meals'][meal_type]['foods'].append(recommended_foods[0])
                                day['meals'][meal_type]['estimated_cost'] += self.get_food_cost(recommended_foods[0])
        
        return meal_plan
    
    def add_variety(self, meal_plan):
        """Add variety to prevent meal repetition"""
        # Track food frequency
        food_frequency = defaultdict(int)
        
        for day in meal_plan:
            for meal in day['meals'].values():
                for food in meal['foods']:
                    food_frequency[food] += 1
        
        # Replace frequently repeated foods
        for food, frequency in food_frequency.items():
            if frequency > 3:  # Appears more than 3 times
                # Find alternatives
                alternatives = self.nlp_processor.suggest_cultural_alternatives(food)
                
                # Replace some occurrences
                replacements_made = 0
                for day in meal_plan:
                    if replacements_made >= frequency // 2:
                        break
                    
                    for meal_type, meal in day['meals'].items():
                        if food in meal['foods'] and alternatives:
                            meal['foods'] = [alternatives[0] if f == food else f for f in meal['foods']]
                            replacements_made += 1
                            break
        
        return meal_plan
    
    def analyze_meal_plan_nutrition(self, meal_plan):
        """Analyze nutritional content of the meal plan"""
        daily_nutrition = []
        
        for day in meal_plan:
            day_totals = defaultdict(float)
            
            for meal in day['meals'].values():
                for food in meal['foods']:
                    nutrition = self.nutrition_predictor.get_food_nutrition(food)
                    for nutrient, value in nutrition.items():
                        day_totals[nutrient] += value
            
            daily_nutrition.append({
                'day': day['day'],
                'nutrition': dict(day_totals)
            })
        
        # Calculate weekly averages
        weekly_averages = defaultdict(float)
        for day_nutrition in daily_nutrition:
            for nutrient, value in day_nutrition['nutrition'].items():
                weekly_averages[nutrient] += value
        
        for nutrient in weekly_averages:
            weekly_averages[nutrient] /= 7
        
        return {
            'daily_nutrition': daily_nutrition,
            'weekly_averages': dict(weekly_averages)
        }
    
    def calculate_cost_breakdown(self, meal_plan):
        """Calculate detailed cost breakdown"""
        cost_breakdown = {
            'daily_costs': [],
            'meal_type_costs': defaultdict(float),
            'food_category_costs': defaultdict(float),
            'total_weekly_cost': 0
        }
        
        for day in meal_plan:
            day_cost = 0
            
            for meal_type, meal in day['meals'].items():
                meal_cost = meal['estimated_cost']
                day_cost += meal_cost
                cost_breakdown['meal_type_costs'][meal_type] += meal_cost
                
                # Categorize food costs
                for food in meal['foods']:
                    food_cost = self.get_food_cost(food)
                    category = self.categorize_food(food)
                    cost_breakdown['food_category_costs'][category] += food_cost
            
            cost_breakdown['daily_costs'].append({
                'day': day['day'],
                'cost': day_cost
            })
            
            cost_breakdown['total_weekly_cost'] += day_cost
        
        return cost_breakdown
    
    def categorize_food(self, food):
        """Categorize food items"""
        categories = {
            'grains': ['ugali', 'rice', 'bread', 'chapati', 'porridge'],
            'proteins': ['beans', 'meat', 'chicken', 'fish', 'eggs', 'groundnuts'],
            'vegetables': ['sukuma wiki', 'spinach', 'cabbage', 'tomatoes', 'onions', 'carrots'],
            'dairy': ['milk', 'cheese', 'yogurt'],
            'oils': ['cooking oil', 'avocado'],
            'beverages': ['tea', 'coffee', 'water'],
            'tubers': ['sweet potato', 'cassava', 'potato']
        }
        
        food_lower = food.lower()
        
        for category, foods in categories.items():
            if any(f in food_lower for f in foods):
                return category
        
        return 'other'
    
    def generate_shopping_list(self, meal_plan):
        """Generate shopping list from meal plan"""
        shopping_list = defaultdict(int)
        
        for day in meal_plan:
            for meal in day['meals'].values():
                for food in meal['foods']:
                    shopping_list[food] += 1
        
        # Convert to list with quantities
        organized_list = []
        for food, quantity in shopping_list.items():
            organized_list.append({
                'item': food,
                'quantity': quantity,
                'estimated_cost': self.get_food_cost(food) * quantity,
                'category': self.categorize_food(food)
            })
        
        # Sort by category
        organized_list.sort(key=lambda x: x['category'])
        
        return organized_list
