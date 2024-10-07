import pandas as pd
from django import forms

# Load the CSV file to get default values for the form
test_first_row_data = pd.read_csv('data&model/test_fat_data.csv')
default_values = test_first_row_data.iloc[0].to_dict()

class CovertypePredictionForm(forms.Form):
    Age = forms.FloatField(label='Age', initial=default_values['Age'])
    Gender_CODE = forms.FloatField(label='Gender_CODE', initial=default_values['Gender_CODE'])
    Height = forms.FloatField(label='Height', initial=default_values['Height'])
    Weight = forms.FloatField(label='Weight', initial=default_values['Weight'])
    CALC_CODE = forms.FloatField(label='CALC_CODE', initial=default_values['CALC_CODE'])
    FAVC_CODE = forms.FloatField(label='FAVC_CODE', initial=default_values['FAVC_CODE'])
    FCVC = forms.FloatField(label='FCVC', initial=default_values['FCVC'])
    NCP = forms.FloatField(label='NCP', initial=default_values['NCP'])
    SCC_CODE = forms.FloatField(label='SCC_CODE', initial=default_values['SCC_CODE'])
    SMOKE_CODE = forms.FloatField(label='SMOKE_CODE', initial=default_values['SMOKE_CODE'])
    CH2O = forms.FloatField(label='CH2O', initial=default_values['CH2O'])
    FHWO_CODE = forms.FloatField(label='FHWO_CODE', initial=default_values['FHWO_CODE'])
    FAF = forms.FloatField(label='FAF', initial=default_values['FAF'])
    TUE = forms.FloatField(label='TUE', initial=default_values['TUE'])
    CAEC_CODE = forms.FloatField(label='CAEC_CODE', initial=default_values['CAEC_CODE'])
    MTRANS_CODE = forms.FloatField(label='MTRANS_CODE', initial=default_values['MTRANS_CODE'])
