import pandas as pd
import joblib
from django.shortcuts import render
from .forms import CovertypePredictionForm

# Load the CSV file globally to avoid reloading it multiple times
csv_data = pd.read_csv('data&model/test_fat_data.csv')

def predict_fat(request):
    prediction = None
    target_value = None
    actual_target_value = None
    response_message = None
    default_values = csv_data.iloc[0].to_dict()  # Default to first row
    button_text = 'Load Second Data'  # Default button text

    if request.method == 'POST':
        if 'load_data' in request.POST:  # Check if the "Load Data" button is clicked
            if request.POST['load_data'] == 'Load Second Data':
                default_values = csv_data.iloc[1].to_dict()  # Get second row data
                button_text = 'Load First Data'  # Change button text to load first data
                response_message = 'Loaded second row data successfully.'
            else:
                default_values = csv_data.iloc[0].to_dict()  # Get first row data
                button_text = 'Load Second Data'  # Change button text back to load second data
                response_message = 'Loaded first row data successfully.'

            form = CovertypePredictionForm(initial=default_values)  # Pre-fill the form with row values
        else:
            form = CovertypePredictionForm(request.POST)
            if form.is_valid():
                input_data = form.cleaned_data
                model_input = [[
                    input_data['Age'],
                    input_data['Gender_CODE'],
                    input_data['Height'],
                    input_data['Weight'],
                    input_data['CALC_CODE'],
                    input_data['FAVC_CODE'],
                    input_data['FCVC'],
                    input_data['NCP'],
                    input_data['SCC_CODE'],
                    input_data['SMOKE_CODE'],
                    input_data['CH2O'],
                    input_data['FHWO_CODE'],
                    input_data['FAF'],
                    input_data['TUE'],
                    input_data['CAEC_CODE'],
                    input_data['MTRANS_CODE'],
                ]]

                # Load the model and make a prediction
                model = joblib.load('data&model/random_forest.sav')
                prediction = model.predict(model_input)

                # Get the actual target from the CSV
                target_from_csv = csv_data['Nobeyesdad__CODE'].values

                # Compare the prediction with the actual target
                if prediction is not None and len(prediction) > 0:
                    predicted_target = prediction[0]
                    target_value = predicted_target
                    
                    if predicted_target in target_from_csv:
                        actual_target_value = target_from_csv[target_from_csv.tolist().index(predicted_target)]
                        response_message = f"Predict correctly! (Prediction: {predicted_target}, Actual Target: {actual_target_value})"
                    else:
                        actual_target_value = target_from_csv[0]
                        response_message = f"Predict incorrectly! (Prediction: {predicted_target}, Actual Target: {actual_target_value})"
    else:
        form = CovertypePredictionForm(initial=default_values)

    return render(request, 'ML_app/fat_predict.html', {
        'form': form,
        'prediction': prediction,
        'target_value': target_value,
        'actual_target_value': actual_target_value,
        'response_message': response_message,
        'button_text': button_text  # Send button text to the template
    })
