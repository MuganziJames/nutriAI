# Data Processing Utilities for NutriAI East Africa
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from textblob import TextBlob

class DataProcessor:
    """Handle data loading and preprocessing for East African nutrition data"""
    
    def __init__(self):
        self.nutrition_data = None
        self.price_data = None
        self.food_groups = {}
        self.east_african_foods = {}
        self.cultural_mappings = {}
        self.load_data()
        self.setup_cultural_mappings()
    
    def load_data(self):
        """Load all CSV files and prepare data"""
        try:
            # Load main nutrition database
            self.nutrition_data = pd.read_csv('data/food.csv')
            
            # Load daily food nutrition dataset
            self.daily_nutrition = pd.read_csv('data/daily_food_nutrition_dataset.csv')
            
            # Load food groups
            for i in range(1, 6):
                group_data = pd.read_csv(f'data/FOOD-DATA-GROUP{i}.csv')
                self.food_groups[f'group_{i}'] = group_data
            
            # Load price data (WFP data)
            self.price_data = pd.read_csv('data/WLD_RTFP_country_2023-10-02.csv')
            
            # Focus on East African countries
            east_african_countries = ['Kenya', 'Uganda', 'Tanzania', 'Rwanda', 'Burundi', 'Ethiopia', 'Somalia', 'South Sudan']
            self.price_data = self.price_data[self.price_data['country'].isin(east_african_countries)]
            
            print("Data loaded successfully!")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create fallback data
            self.create_fallback_data()
    
    def create_fallback_data(self):
        """Create basic fallback data if files can't be loaded"""
        # Basic East African foods with nutrition info
        fallback_foods = {
            'food': ['ugali', 'sukuma wiki', 'beans', 'rice', 'maize', 'sweet potato', 'cassava', 'groundnuts', 'milk', 'eggs'],
            'Caloric Value': [378, 22, 347, 365, 365, 86, 160, 567, 61, 155],
            'Protein': [8.1, 2.8, 21.6, 7.1, 9.4, 1.6, 1.4, 25.8, 3.2, 13.0],
            'Carbohydrates': [84.0, 4.3, 63.0, 80.0, 74.3, 20.1, 38.1, 16.1, 4.7, 1.1],
            'Fat': [1.4, 0.3, 1.2, 0.7, 4.7, 0.1, 0.3, 49.2, 3.3, 11.0],
            'Calcium': [6, 212, 143, 28, 7, 30, 16, 92, 113, 50],
            'Iron': [1.2, 2.7, 8.2, 0.8, 2.7, 0.6, 0.3, 4.6, 0.03, 1.2],
            'Vitamin A': [0, 418, 0, 0, 11, 709, 13, 0, 46, 160],
            'Vitamin C': [0, 60, 0, 0, 0, 2.4, 20.6, 0, 0, 0]
        }
        
        self.nutrition_data = pd.DataFrame(fallback_foods)
        
        # Basic price data
        price_data = {
            'country': ['Kenya'] * 12,
            'date': pd.date_range('2023-01-01', periods=12, freq='M'),
            'Close': [50, 52, 48, 55, 60, 58, 62, 65, 63, 68, 70, 72]  # Sample prices in KES
        }
        
        self.price_data = pd.DataFrame(price_data)
    
    def setup_cultural_mappings(self):
        """Setup cultural food mappings for East African context"""
        self.cultural_mappings = {
            # Swahili names
            'ugali': ['maize flour', 'cornmeal', 'posho'],
            'sukuma wiki': ['collard greens', 'kale', 'spinach'],
            'nyama': ['meat', 'beef', 'chicken'],
            'samaki': ['fish', 'tilapia'],
            'maziwa': ['milk', 'dairy'],
            'mayai': ['eggs', 'egg'],
            'muhogo': ['cassava', 'tapioca'],
            'viazi': ['sweet potato', 'potato'],
            'maharage': ['beans', 'legumes'],
            'mchele': ['rice', 'grain'],
            'karanga': ['groundnuts', 'peanuts'],
            
            # Ethiopian foods
            'injera': ['teff', 'grain'],
            'berbere': ['spices', 'seasoning'],
            'doro': ['chicken', 'poultry'],
            'kitfo': ['beef', 'meat'],
            
            # Ugandan foods
            'posho': ['maize', 'ugali'],
            'matoke': ['banana', 'plantain'],
            'sim sim': ['sesame', 'seeds'],
            
            # Common meal patterns
            'breakfast': ['chai', 'mandazi', 'eggs', 'bread'],
            'lunch': ['ugali', 'sukuma wiki', 'beans', 'meat'],
            'dinner': ['rice', 'stew', 'vegetables']
        }
    
    def get_east_african_foods(self):
        """Get foods commonly available in East Africa"""
        if self.nutrition_data is not None:
            # Filter for foods commonly available in East Africa
            common_foods = ['maize', 'rice', 'beans', 'groundnuts', 'sweet potato', 'cassava', 
                          'banana', 'milk', 'eggs', 'chicken', 'beef', 'fish', 'spinach', 
                          'kale', 'tomato', 'onion', 'cabbage', 'carrot']
            
            # Try to match foods in our dataset using Description column
            available_foods = []
            for food in common_foods:
                matches = self.nutrition_data[self.nutrition_data['Description'].str.contains(food, case=False, na=False)]
                if not matches.empty:
                    # Create a standardized food entry
                    food_item = matches.iloc[0].copy()
                    food_item['food'] = food  # Add standardized food name
                    available_foods.append(food_item)
            
            return pd.DataFrame(available_foods)
        return pd.DataFrame()
    
    def get_seasonal_availability(self, country='Kenya'):
        """Predict seasonal food availability based on price patterns"""
        if self.price_data is not None and not self.price_data.empty:
            country_data = self.price_data[self.price_data['country'] == country].copy()
            
            if not country_data.empty:
                country_data['date'] = pd.to_datetime(country_data['date'])
                country_data['month'] = country_data['date'].dt.month
                
                # Calculate seasonal patterns
                seasonal_patterns = country_data.groupby('month')['Close'].agg(['mean', 'std']).reset_index()
                seasonal_patterns['availability_score'] = 100 - (seasonal_patterns['mean'] / seasonal_patterns['mean'].max() * 100)
                
                return seasonal_patterns
        
        # Fallback seasonal data for East Africa
        fallback_seasonal = {
            'month': list(range(1, 13)),
            'availability_score': [70, 65, 60, 55, 50, 45, 40, 45, 55, 65, 75, 80]  # Harvest seasons
        }
        
        return pd.DataFrame(fallback_seasonal)
    
    def get_nutrition_density_score(self, food_item):
        """Calculate nutrition density score for a food item"""
        if self.nutrition_data is not None:
            food_data = self.nutrition_data[self.nutrition_data['Description'].str.contains(food_item, case=False, na=False)]
            
            if not food_data.empty:
                item = food_data.iloc[0]
                
                # Calculate nutrition density (nutrients per calorie)
                calories = item.get('Data.Kilocalories', 100)
                protein = item.get('Data.Protein', 0)
                calcium = item.get('Data.Major Minerals.Calcium', 0)
                iron = item.get('Data.Major Minerals.Iron', 0)
                vitamin_c = item.get('Data.Vitamins.Vitamin C', 0)
                
                # Weighted nutrition density score
                density_score = (
                    (protein * 4) +  # Protein factor
                    (calcium * 0.01) +  # Calcium factor
                    (iron * 10) +  # Iron factor
                    (vitamin_c * 2)  # Vitamin C factor
                ) / max(calories, 1)
                
                return min(density_score * 10, 100)  # Scale to 0-100
        
        return 50  # Default medium score
    
    def search_culturally_appropriate_foods(self, user_input):
        """Search for foods based on cultural context and user input"""
        user_input = user_input.lower()
        found_foods = []
        
        # Search in cultural mappings
        for local_name, alternatives in self.cultural_mappings.items():
            if local_name in user_input:
                found_foods.extend(alternatives)
            for alt in alternatives:
                if alt in user_input:
                    found_foods.append(local_name)
        
        # Search in nutrition database using Description column
        if self.nutrition_data is not None:
            for food in found_foods:
                matches = self.nutrition_data[self.nutrition_data['Description'].str.contains(food, case=False, na=False)]
                if not matches.empty:
                    return matches.iloc[0]
        
        return None
