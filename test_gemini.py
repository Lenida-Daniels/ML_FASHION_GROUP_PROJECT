import os
import sys
sys.path.append("api")

from gemini_integration import FashionPriceAPI
from dotenv import load_dotenv

load_dotenv()

def test_gemini_api():
    """Test Gemini API Intergaration"""
    print("===Testing Gemini API Intergration===")

    #Check if the gemini API is set
    api_key = os.getenv("GEMINI_API_KEY")                                  
    if not api_key:
        print("GEMINI_API_KEY not set in environment")
        print("set it with : export GEMINI_API_KEY = 'your-api-key'")
        return

    try:
        #Initialize API
        api = FashionPriceAPI()
        print("Gemini API initialized")

        #Test with simple category
        category = "Sneaker"
        location = "Nairboi, Kenya"

        print(f"\nTesting price lookup for : {category}")
        result = api.get_price_and_stores(category, location)

        print("API Response:")
        print("-" * 50)
        print(result)
        print("-" * 50)    


    except Exception as e:
        print(f"Gemini API test failed: {e}")


if __name__ == "__main__":
    test_gemini_api()    