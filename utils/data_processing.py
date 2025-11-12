from datetime import datetime
import pandas as pd

def load_emission_data(file_path):
    """Load emission quantification results from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading emission data: {e}")
        return pd.DataFrame()

def process_methane_facts(facts_file):
    """Load and process methane facts from a JSON file."""
    try:
        with open(facts_file, 'r') as f:
            facts = json.load(f)
        return facts
    except Exception as e:
        print(f"Error loading methane facts: {e}")
        return {}

def process_technology_specs(specs_file):
    """Load and process technology specifications from a JSON file."""
    try:
        with open(specs_file, 'r') as f:
            specs = json.load(f)
        return specs
    except Exception as e:
        print(f"Error loading technology specifications: {e}")
        return {}

def process_satellite_details(details_file):
    """Load and process satellite details from a JSON file."""
    try:
        with open(details_file, 'r') as f:
            details = json.load(f)
        return details
    except Exception as e:
        print(f"Error loading satellite details: {e}")
        return {}

def get_current_date():
    """Return the current date formatted as a string."""
    return datetime.now().strftime("%Y-%m-%d")