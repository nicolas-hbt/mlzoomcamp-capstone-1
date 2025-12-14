import requests

# The endpoint of the Flask service
url = 'http://localhost:9696/predict'

candidate_a = {
    "city": "city_115",
    "city_development_index": 0.789,
    "gender": None,  # Testing missing value handling
    "relevent_experience": "No relevent experience",
    "enrolled_university": None,
    "education_level": "Graduate",
    "major_discipline": "Business Degree",
    "experience": "<1",  # Ordinal string that needs cleaning
    "company_size": None,
    "company_type": "Pvt Ltd",
    "last_new_job": "never",
    "training_hours": 52
}

candidate_b = {
    "city": "city_40",
    "city_development_index": 0.776,
    "gender": "Male",
    "relevent_experience": "No relevent experience",
    "enrolled_university": "no_enrollment",
    "education_level": "Graduate",
    "major_discipline": "STEM",
    "experience": "15",
    "company_size": "50-99",
    "company_type": "Pvt Ltd",
    "last_new_job": ">4",
    "training_hours": 47
}


candidate_c = {
    "city": "city_11",
    "city_development_index": 0.550,  # Very low CDI is strongly correlated with Target 1
    "gender": "Male",
    "relevent_experience": "No relevent experience",
    "enrolled_university": "Full time course",
    "education_level": "Graduate",
    "major_discipline": "STEM",
    "experience": "2",
    "company_size": None,  # Often missing for job seekers
    "company_type": None,
    "last_new_job": "never",
    "training_hours": 20
}


candidate_d = {
    "city": "city_103",
    "city_development_index": 0.920,
    "gender": "Female",
    "relevent_experience": "Has relevent experience",
    "enrolled_university": "no_enrollment",
    "education_level": "Phd",
    "major_discipline": "STEM",
    "experience": ">20",
    "company_size": "10000+",
    "company_type": "Pvt Ltd",
    "last_new_job": ">4",
    "training_hours": 150
}


candidate_e = {
    "city": "city_21",
    "city_development_index": 0.624,
    "gender": "Male",
    "relevent_experience": "Has relevent experience",
    "enrolled_university": "no_enrollment",
    "education_level": "Masters",
    "major_discipline": "STEM",
    "experience": "5",
    "company_size": "50-99",
    "company_type": "Funded Startup",
    "last_new_job": "1",
    "training_hours": 45
}

# List of candidates to iterate through
candidates = [
    ("Candidate A (Missing Values / Junior)", candidate_a),
    ("Candidate B (Mid CDI / Experienced)", candidate_b),
    ("Candidate C (Low CDI / Junior)", candidate_c),
    ("Candidate D (High CDI / Senior)", candidate_d),
    ("Candidate E (Startup / Mid-level)", candidate_e)
]

print("-" * 30)
for name, data in candidates:
    print(f"Sending request for {name}...")
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("Response:", response.json())
        else:
            print("Error:", response.text)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Is it running?")
    print("-" * 30)