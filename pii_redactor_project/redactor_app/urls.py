from django.urls import path
from . import views

urlpatterns = [
    # This route handles the main page load.
    path('', views.index, name='index'),
    
    # This route handles the file upload and redaction API call.
    path('redact/', views.redact_file_view, name='redact_file'),
]
