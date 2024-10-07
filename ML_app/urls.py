from django.urls import path
from . import views
urlpatterns = [
 path('', views.predict_fat, name='predict_fat'),
]