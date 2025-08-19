from django.shortcuts import render
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import numpy as np

# Dummy label encoder alternative using index mapping
crop_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
targets = ['rice', 'wheat', 'maize', 'cotton', 'barley',
           'millet', 'sugarcane', 'soybean', 'sunflower', 'pulses']

fertilizers = {
    ('rice', 0): 'Urea + DAP for Clay',
    ('rice', 1): 'Urea + DAP for Loamy',
    ('rice', 2): 'Urea + MOP for Sandy',
    ('wheat', 0): 'Urea + SSP for Clay',
    ('wheat', 1): 'Urea + DAP for Loamy',
    ('wheat', 2): 'NPK + Compost for Sandy',
    ('maize', 0): 'Urea + DAP for Clay',
    ('maize', 1): 'Urea + SSP for Loamy',
    ('maize', 2): 'Urea + MOP for Sandy',
    ('cotton', 0): 'Potash + FYM for Clay',
    ('cotton', 1): 'SSP + Urea for Loamy',
    ('cotton', 2): 'Urea + Potassium for Sandy',
    ('barley', 0): 'Ammonium Sulphate for Clay',
    ('barley', 1): 'Urea + DAP for Loamy',
    ('barley', 2): 'FYM + Potash for Sandy',
    ('millet', 0): 'Ammonium Nitrate for Clay',
    ('millet', 1): 'DAP + Organic for Loamy',
    ('millet', 2): 'MOP + FYM for Sandy',
    ('sugarcane', 0): 'SSP + Urea for Clay',
    ('sugarcane', 1): 'Compost + DAP for Loamy',
    ('sugarcane', 2): 'FYM + Urea for Sandy',
    ('soybean', 0): 'Phosphate-rich Manure for Clay',
    ('soybean', 1): 'Phosphate + Potash for Loamy',
    ('soybean', 2): 'SSP + Compost for Sandy',
    ('sunflower', 0): 'NPK + Potash for Clay',
    ('sunflower', 1): 'Urea + SSP for Loamy',
    ('sunflower', 2): 'Potash + SSP for Sandy',
    ('pulses', 0): 'DAP + Organic Manure for Clay',
    ('pulses', 1): 'Compost + NPK for Loamy',
    ('pulses', 2): 'Urea + FYM for Sandy',
}

def predict_crop_and_fertilizer(N, P, K, temperature, humidity, ph, rainfall, soil_type, algorithm='naive_bayes'):
    features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall, soil_type]],
                            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'soil_type'])

    # Dummy dataset
    crop_data = pd.DataFrame({
        'N': [50, 100, 150, 80, 120, 30, 90, 60, 140, 110],
        'P': [40, 60, 90, 70, 85, 20, 65, 45, 95, 75],
        'K': [60, 80, 100, 70, 90, 25, 85, 55, 110, 95],
        'temperature': [28, 32, 25, 30, 33, 20, 35, 27, 26, 31],
        'humidity': [80, 70, 60, 65, 75, 85, 55, 60, 70, 68],
        'ph': [6.5, 6.0, 7.0, 6.8, 6.2, 5.5, 7.5, 6.7, 6.1, 6.9],
        'rainfall': [100, 150, 120, 110, 130, 90, 200, 140, 160, 170],
        'soil_type': [0, 1, 2, 0, 1, 2, 0, 1, 2, 1]
    })

    # Labels
    crop_labels = list(range(10))

    # Choose model
    if algorithm == 'logistic_regression':
        model = LogisticRegression(max_iter=1000)
    else:
        model = GaussianNB()

    # Train and predict
    model.fit(crop_data, crop_labels)
    probs = model.predict_proba(features)[0]
    top_indices = np.argsort(probs)[::-1][:3]  # top 3 crops

    recommendations = []
    for idx in top_indices:
        crop_name = targets[idx]
        fert = fertilizers.get((crop_name, soil_type), 'General Fertilizer')
        recommendations.append({'crop': crop_name, 'fertilizer': fert})

    return recommendations

def home(request):
    if request.method == "POST":
        try:
            N = float(request.POST["N"])
            P = float(request.POST["P"])
            K = float(request.POST["K"])
            temperature = float(request.POST["temperature"])
            humidity = float(request.POST["humidity"])
            ph = float(request.POST["ph"])
            rainfall = float(request.POST["rainfall"])
            soil_type = int(request.POST["soil_type"])
            algorithm = request.POST.get("algorithm", "naive_bayes")
        except (ValueError, KeyError):
            return render(request, 'index.html', {'error': 'Invalid input. Please enter correct values.'})

        recommendations = predict_crop_and_fertilizer(
            N, P, K, temperature, humidity, ph, rainfall, soil_type, algorithm
        )

        return render(request, 'result.html', {
            'recommendations': recommendations
        })

    # For dropdowns: pass context with ranges
    context = {
        'range_0_200': range(0, 201, 10),
        'range_0_150': range(0, 151, 10),
        'range_10_45': range(10, 46),
        'range_20_100': range(20, 101, 5),
        'range_4_9': range(4, 10),
        'range_50_300': range(50, 301, 10),
    }

    return render(request, 'index.html', context)
