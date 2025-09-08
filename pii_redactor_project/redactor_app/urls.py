from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('redact/', views.redact_file_view, name='redact'),
]