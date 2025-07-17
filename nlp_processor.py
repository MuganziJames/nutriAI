# NLP Processing for East African Food Context
import re
import nltk
from textblob import TextBlob
import pandas as pd

class NLPProcessor:
    """Handle natural language processing for food-related queries"""
    
    def __init__(self):
        self.setup_nltk()
        self.setup_food_vocabulary()
        self.setup_dietary_patterns()
    
    def setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def setup_food_vocabulary(self):
        """Setup vocabulary for East African foods and cooking terms"""
        self.food_vocabulary = {
            # Swahili food terms
            'ugali': ['ugali', 'posho', 'sima', 'cornmeal'],
            'sukuma wiki': ['sukuma wiki', 'collard greens', 'kale', 'spinach'],
            'nyama': ['nyama', 'meat', 'beef', 'goat', 'chicken'],
            'samaki': ['samaki', 'fish', 'tilapia'],
            'maharage': ['maharage', 'beans', 'kidney beans', 'black beans'],
            'mchele': ['mchele', 'rice', 'pilau'],
            'viazi': ['viazi', 'sweet potato', 'potato'],
            'muhogo': ['muhogo', 'cassava', 'tapioca'],
            'karanga': ['karanga', 'groundnuts', 'peanuts'],
            'maziwa': ['maziwa', 'milk', 'dairy'],
            'mayai': ['mayai', 'eggs', 'egg'],
            'chai': ['chai', 'tea', 'black tea'],
            'kahawa': ['kahawa', 'coffee'],
            'mandazi': ['mandazi', 'doughnut', 'fried bread'],
            'chapati': ['chapati', 'flatbread', 'roti'],
            'matoke': ['matoke', 'green banana', 'cooking banana'],
            'ndizi': ['ndizi', 'banana', 'ripe banana'],
            
            # Ethiopian foods
            'injera': ['injera', 'teff bread', 'sourdough flatbread'],
            'berbere': ['berbere', 'spice mix', 'chili spice'],
            'doro': ['doro', 'chicken'],
            'kitfo': ['kitfo', 'raw beef', 'steak tartare'],
            'shiro': ['shiro', 'chickpea powder', 'lentil stew'],
            
            # Ugandan foods
            'posho': ['posho', 'ugali', 'maize meal'],
            'sim sim': ['sim sim', 'sesame seeds'],
            'g-nuts': ['g-nuts', 'groundnuts', 'peanuts'],
            
            # Cooking methods
            'boiled': ['boiled', 'cooked in water'],
            'fried': ['fried', 'deep fried', 'pan fried'],
            'roasted': ['roasted', 'grilled'],
            'steamed': ['steamed', 'cooked with steam'],
            'stewed': ['stewed', 'cooked slowly'],
            'grilled': ['grilled', 'barbecued', 'roasted over fire']
        }
        
        self.dietary_keywords = {
            'vegetarian': ['vegetarian', 'no meat', 'plant based', 'vegan'],
            'no_dairy': ['no dairy', 'lactose intolerant', 'no milk'],
            'gluten_free': ['gluten free', 'no wheat', 'no gluten'],
            'diabetic': ['diabetic', 'low sugar', 'sugar free'],
            'low_salt': ['low salt', 'no salt', 'low sodium'],
            'pregnancy': ['pregnant', 'expecting', 'pregnancy'],
            'child': ['child', 'kid', 'toddler', 'baby'],
            'elderly': ['elderly', 'old', 'senior']
        }
    
    def setup_dietary_patterns(self):
        """Setup common dietary patterns in East Africa"""
        self.meal_patterns = {
            'breakfast': {
                'common_foods': ['chai', 'mandazi', 'eggs', 'bread', 'porridge'],
                'typical_combinations': [
                    ['chai', 'mandazi'],
                    ['eggs', 'bread'],
                    ['porridge', 'milk']
                ]
            },
            'lunch': {
                'common_foods': ['ugali', 'sukuma wiki', 'beans', 'meat', 'rice'],
                'typical_combinations': [
                    ['ugali', 'sukuma wiki', 'beans'],
                    ['rice', 'meat', 'vegetables'],
                    ['ugali', 'fish', 'vegetables']
                ]
            },
            'dinner': {
                'common_foods': ['rice', 'stew', 'vegetables', 'ugali', 'meat'],
                'typical_combinations': [
                    ['rice', 'beef stew'],
                    ['ugali', 'vegetable stew'],
                    ['chapati', 'beans']
                ]
            }
        }
    
    def process_user_input(self, user_text):
        """Process user input to extract food preferences and dietary requirements"""
        if not user_text:
            return {}
        
        user_text = user_text.lower()
        blob = TextBlob(user_text)
        
        result = {
            'foods_mentioned': [],
            'dietary_restrictions': [],
            'meal_type': None,
            'cooking_preferences': [],
            'family_info': {},
            'sentiment': blob.sentiment.polarity,
            'cultural_context': []
        }
        
        # Extract mentioned foods
        for food_type, variations in self.food_vocabulary.items():
            for variation in variations:
                if variation in user_text:
                    result['foods_mentioned'].append(food_type)
                    break
        
        # Extract dietary restrictions
        for restriction, keywords in self.dietary_keywords.items():
            for keyword in keywords:
                if keyword in user_text:
                    result['dietary_restrictions'].append(restriction)
                    break
        
        # Extract meal type
        if any(word in user_text for word in ['breakfast', 'morning', 'early']):
            result['meal_type'] = 'breakfast'
        elif any(word in user_text for word in ['lunch', 'midday', 'afternoon']):
            result['meal_type'] = 'lunch'
        elif any(word in user_text for word in ['dinner', 'evening', 'night']):
            result['meal_type'] = 'dinner'
        
        # Extract family information
        if 'family' in user_text or 'children' in user_text:
            result['family_info']['has_children'] = True
        if 'pregnant' in user_text or 'expecting' in user_text:
            result['family_info']['pregnant'] = True
        
        # Extract budget information
        budget_match = re.search(r'(\d+)\s*(ksh|kes|shilling)', user_text)
        if budget_match:
            result['budget'] = int(budget_match.group(1))
        
        # Extract cooking preferences
        cooking_methods = ['boiled', 'fried', 'roasted', 'steamed', 'stewed', 'grilled']
        for method in cooking_methods:
            if method in user_text:
                result['cooking_preferences'].append(method)
        
        return result
    
    def generate_meal_description(self, meal_components, meal_type='lunch'):
        """Generate culturally appropriate meal descriptions"""
        if not meal_components:
            return "A balanced meal with local ingredients"
        
        # Get cultural context
        cultural_names = []
        for component in meal_components:
            for local_name, alternatives in self.food_vocabulary.items():
                if component.lower() in alternatives:
                    cultural_names.append(local_name)
                    break
            else:
                cultural_names.append(component)
        
        # Generate description based on meal type
        if meal_type == 'breakfast':
            if 'chai' in cultural_names:
                return f"Traditional East African breakfast with {', '.join(cultural_names)}"
            else:
                return f"Nutritious morning meal featuring {', '.join(cultural_names)}"
        
        elif meal_type == 'lunch':
            if 'ugali' in cultural_names:
                return f"Classic East African lunch with {', '.join(cultural_names)}"
            else:
                return f"Hearty midday meal with {', '.join(cultural_names)}"
        
        elif meal_type == 'dinner':
            return f"Satisfying evening meal with {', '.join(cultural_names)}"
        
        return f"Balanced meal with {', '.join(cultural_names)}"
    
    def suggest_cultural_alternatives(self, unavailable_food):
        """Suggest culturally appropriate alternatives for unavailable foods"""
        alternatives = {
            'meat': ['beans', 'groundnuts', 'eggs', 'fish'],
            'rice': ['ugali', 'sweet potato', 'cassava'],
            'wheat': ['maize', 'rice', 'cassava'],
            'dairy': ['groundnuts', 'sesame seeds', 'beans'],
            'spinach': ['sukuma wiki', 'cabbage', 'amaranth leaves'],
            'potato': ['sweet potato', 'cassava', 'yam'],
            'oil': ['groundnuts', 'sesame seeds', 'avocado']
        }
        
        food_lower = unavailable_food.lower()
        
        for key, alts in alternatives.items():
            if key in food_lower:
                return alts
        
        return ['beans', 'maize', 'vegetables']  # Default alternatives
    
    def analyze_nutrition_needs(self, user_profile):
        """Analyze nutritional needs based on user profile"""
        needs = {
            'protein': 'medium',
            'iron': 'medium',
            'calcium': 'medium',
            'vitamin_c': 'medium',
            'calories': 'medium'
        }
        
        # Adjust based on user profile
        if 'pregnant' in user_profile.get('family_info', {}):
            needs['iron'] = 'high'
            needs['calcium'] = 'high'
            needs['protein'] = 'high'
        
        if 'has_children' in user_profile.get('family_info', {}):
            needs['calcium'] = 'high'
            needs['vitamin_c'] = 'high'
        
        if 'elderly' in user_profile.get('dietary_restrictions', []):
            needs['calcium'] = 'high'
            needs['protein'] = 'high'
        
        if 'diabetic' in user_profile.get('dietary_restrictions', []):
            needs['calories'] = 'low'
        
        return needs
    
    def generate_shopping_list(self, meal_plan, language='english'):
        """Generate shopping list in local language"""
        if language == 'swahili':
            translations = {
                'maize': 'mahindi',
                'beans': 'maharage',
                'meat': 'nyama',
                'fish': 'samaki',
                'milk': 'maziwa',
                'eggs': 'mayai',
                'rice': 'mchele',
                'vegetables': 'mboga',
                'oil': 'mafuta',
                'salt': 'chumvi',
                'sugar': 'sukari',
                'tea': 'chai'
            }
            
            translated_items = []
            for item in meal_plan:
                translated = translations.get(item.lower(), item)
                translated_items.append(translated)
            
            return translated_items
        
        return meal_plan
