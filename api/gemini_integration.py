import google.generativeai as genai
import os

class FashionPriceAPI:
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def get_price_and_stores(self, clothing_category, user_location=""):
            #Get price estimates and store suggestions
            prompt = f"""
            For {clothing_category} clothing:
            1. Provide average price range in KSH
            2. List 3-5 popular stores/brands where to buy
            3. Keep response concise and practical
            
            Location context: {user_location if user_location else "General"}"""

            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"Unable to fetch pricing info: {str(e)}"